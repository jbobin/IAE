"""
Metric Learning
"""

import pickle
from jax import grad, jit, lax
import jax.numpy as np
from jax.experimental.optimizers import adam, momentum, sgd, nesterov, adagrad, rmsprop
import numpy as onp
import time
import sys
from jax.nn import softplus,silu
from jax import random
from jax import vmap
import matplotlib.pyplot as plt

###################################################
# Elementary functions
###################################################

def load_model(fname):
    dataf = open(fname + '.pkl', 'rb')
    model = pickle.load(dataf)
    dataf.close()
    return model


############################################################
# Main code
############################################################

class IAE(object):
    """
    Model - input IAE model, overrides other parameters if provided (except the number of layers)
    fname - filename for the IAE model
    AnchorPoints - anchor points
    NSize - network structure (e.g. [8, 8, 8, 8] for a 3-layer neural network of size 8)
    active_forward - activation function in the encoder
    active_backward - activation function in the decoder
    res_factor - residual injection factor in the ResNet-like architecture
    reg_parameter - weighting constant to balance between the sample and transformed domains
    cost_weight - weighting constant to balance between the sample and transformed domains in the learning stage
    reg_inv - regularization term in the barycenter computation
    simplex - simplex constraint onto the barycentric coefficients
    nneg_weights - non-negative constraint onto the barycentric coefficients
    nneg_output - non-negative constraint onto the output
    noise_level - noise level in the learning stage as in the denoising autoencoder
    cost_type - cost function (not used)
    optim_learn - optimization algorithm in the learning stage
        (0: Adam, 1: Momentum, 2: RMSprop, 3: AdaGrad, 4: Nesterov, 5: SGD)
    optim_proj - optimization algorithm in the barycentric span projection
    step_size - step size of the optimization algorithms
    niter - number of iterations of the optimization algorithms
    eps_cvg - convergence tolerance
    verb - verbose mode
    """

    def __init__(self, Model=None, fname='IAE_model', AnchorPoints=None, NSize=None, active_forward='lRelu',
                 active_backward='lRelu', res_factor=0.1, reg_parameter=1000., cost_weight=None, reg_inv=1e-9,
                 simplex=False, nneg_weights=False, nneg_output=False, noise_level=None, cost_type=0, optim_learn=0,
                 optim_proj=3, step_size=1e-2, niter=5000, eps_cvg=1e-9, verb=False, enable_train_bn=True, dropout_rate=None,reg_parameter_schedule=0,learning_rate_schedule=0,noise_level_schedule=0,
                 code_version="version_2_June_7th_2021"):
        """
        Initialization
        """

        self.Model = Model
        self.fname = fname
        self.AnchorPoints = AnchorPoints
        self.num_anchor_points = None
        self.Params = {}
        self.PhiE = None
        self.NSize = NSize
        self.nlayers = None
        self.active_forward = active_forward
        self.active_backward = active_backward
        self.res_factor = res_factor
        self.ResParams = None
        self.reg_parameter = reg_parameter
        self.cost_weight = cost_weight
        self.reg_inv = reg_inv
        self.simplex = simplex
        self.nneg_weights = nneg_weights
        self.nneg_output = nneg_output
        self.noise_level = noise_level
        self.cost_type = cost_type
        self.optim_learn = optim_learn
        self.optim_proj = optim_proj
        self.step_size = step_size
        self.niter = niter
        self.eps_cvg = eps_cvg
        self.verb = verb
        self.code_version = code_version
        self.bn_param = {}
        self.enable_train_bn = enable_train_bn
        self.dropout_rate = dropout_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.noise_level_schedule = noise_level_schedule
        self.num_batches=1
        self.reg_parameter_schedule = reg_parameter_schedule

        self.init_parameters()

    def display(self,epoch,epoch_time,train_acc,rel_acc,pref="Learning stage - ",niter=None):

        if niter is None:
            niter = self.niter

        percent_time = epoch/(1e-12+niter)
        n_bar = 50
        bar = ' |'
        bar = bar + '█' * int(n_bar * percent_time)
        bar = bar + '-' * int(n_bar * (1-percent_time))
        bar = bar + ' |'
        bar = bar + onp.str(int(100 * percent_time))+'%'
        m, s = divmod(onp.int(epoch*epoch_time), 60)
        h, m = divmod(m, 60)
        time_run = ' [{:d}:{:02d}:{:02d}<'.format(h, m, s)
        m, s = divmod(onp.int((niter-epoch)*epoch_time), 60)
        h, m = divmod(m, 60)
        time_run += '{:d}:{:02d}:{:02d}]'.format(h, m, s)

        sys.stdout.write('\033[2K\033[1G')
        if epoch_time > 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +onp.str(niter)+ ' -- loss  = {0:e}'.format(onp.float(train_acc)) + ' -- loss rel. var. = {0:e}'.format(onp.float(rel_acc))+bar+time_run+'-{0:0.4} '.format(epoch_time)+' s/epoch', end="\r")
        if epoch_time < 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +onp.str(niter)+ ' -- loss  = {0:e}'.format(onp.float(train_acc)) + ' -- loss rel. var. = {0:e}'.format(onp.float(rel_acc))+bar+time_run+'-{0:0.4}'.format(1./epoch_time)+' epoch/s', end="\r")
        #sys.stdout.flush()

    def init_parameters(self):

        if self.verb:
            # HERE WE SHOULD LIST MOST OPTIONS
            print("code version : ",self.code_version)
            print("simplex constraint : ",self.simplex)

        if self.NSize is None:
            if self.Model is None:
                print("Hey, there's a problem, provide either NSize or Model !")
            else:
                self.NSize = self.Model["NSize"]
        self.nlayers = len(self.NSize) - 1

        if self.AnchorPoints is None:
            if self.Model is None:
                print("Hey, there's a problem, provide either AnchorPoints or Model !")
            else:
                self.AnchorPoints = self.Model["AnchorPoints"]

        for j in range(self.nlayers):
            W0 = onp.random.randn(self.NSize[j], self.NSize[j + 1])
            self.Params["Wt" + str(j)] = W0 / onp.linalg.norm(W0)
            self.Params["bt" + str(j)] = onp.zeros(self.NSize[j + 1])
            self.bn_param["mean_bn_" + str(j)] = np.zeros((1,))
            self.bn_param["std_bn_" + str(j)] = np.ones((1,))
            self.Params["mu" + str(j)] = np.zeros((1,))
            self.Params["std" + str(j)] = np.ones((1,))

        for j in range(self.nlayers):
            W0 = onp.random.randn(self.NSize[-j - 1], self.NSize[-j - 2])
            self.Params["Wp" + str(j)] = W0 / onp.linalg.norm(W0)
            self.Params["bp" + str(j)] = onp.zeros(self.NSize[-j - 2], )

        if self.Model is not None:

            if self.verb > 2:
                print("IAE model is given")

            if self.Model["code_version"] != self.code_version:
                print('Compatibility warning!')

            dL = self.nlayers - self.Model["nlayers"]
            for j in range(self.Model["nlayers"]):
                self.bn_param["mean_bn_" + str(j)] = self.Model["bn_param"]["mean_bn_" + str(j)]
                self.bn_param["std_bn_" + str(j)] = self.Model["bn_param"]["std_bn_" + str(j)]
                self.Params["Wt" + str(j)] = self.Model["Params"]["Wt" + str(j)]
                self.Params["bt" + str(j)] = self.Model["Params"]["bt" + str(j)]
                self.Params["mu" + str(j)] = self.Model["Params"]["mu" + str(j)]
                self.Params["std" + str(j)] = self.Model["Params"]["std" + str(j)]
            for j in range(self.Model["nlayers"]):
                self.Params["Wp" + str(j + dL)] = self.Model["Params"]["Wp" + str(j)]
                self.Params["bp" + str(j + dL)] = self.Model["Params"]["bp" + str(j)]

            #self.fname = self.Model["fname"]
            self.AnchorPoints = self.Model["AnchorPoints"]
            self.active_forward = self.Model["active_forward"]
            self.active_backward = self.Model["active_backward"]
            self.res_factor = self.Model["res_factor"]
            self.reg_parameter = self.Model["reg_parameter"]
            self.cost_weight = self.Model["cost_weight"]
            self.simplex = self.Model["simplex"]
            self.nneg_output = self.Model["nneg_output"]
            self.nneg_weights = self.Model["nneg_weights"]
            self.noise_level = self.Model["noise_level"]
            self.reg_inv = self.Model["reg_inv"]
            self.cost_type = self.Model["cost_type"]
            self.optim_learn = self.Model["optim_learn"]
            self.step_size = self.Model["step_size"]
            self.niter = self.Model["niter"]
            self.eps_cvg = self.Model["eps_cvg"]
            self.verb = self.Model["verb"]
            self.code_version = self.Model["code_version"]
            self.enable_train_bn = self.Model["enable_train_bn"]

        self.ResParams = self.res_factor * (2 ** (onp.arange(self.nlayers) / self.nlayers) - 1)
        self.num_anchor_points = onp.shape(self.AnchorPoints)[0]
        self.encode_anchor_points()

    def update_parameters(self, Params):
        """
        Update the parameters from learnt params
        """

        for j in range(self.nlayers):
            self.Params["Wt" + str(j)] = Params["Wt" + str(j)]
            self.Params["bt" + str(j)] = Params["bt" + str(j)]
            self.Params["Wp" + str(j)] = Params["Wp" + str(j)]
            self.Params["bp" + str(j)] = Params["bp" + str(j)]
            self.Params["mu" + str(j)] = Params["mu" + str(j)]
            self.Params["std" + str(j)] = Params["std" + str(j)]

    def learnt_params_init(self):
        """
        Update the parameters from learnt params
        """

        Params = {}

        for j in range(self.nlayers):
            Params["Wt" + str(j)] = self.Params["Wt" + str(j)]
            Params["bt" + str(j)] = self.Params["bt" + str(j)]
            Params["Wp" + str(j)] = self.Params["Wp" + str(j)]
            Params["bp" + str(j)] = self.Params["bp" + str(j)]
            Params["mu" + str(j)] = self.Params["mu" + str(j)]
            Params["std" + str(j)] = self.Params["std" + str(j)]
            #Params["mean_bn_" + str(j)] = self.Params["mean_bn_" + str(j)]
            #Params["std_bn_" + str(j)] = self.Params["std_bn_" + str(j)]

        return Params

    def batch_normalization(self,batch,epoch,m_glob,s_glob,apply_only=False):

        if apply_only is False:

            m_loc = np.mean(batch)
            s_loc = np.std(batch)

            #m_glob = 1/(epoch+1)*m_loc + epoch/(epoch+1)*m_glob
            #s_glob = 1/(epoch+1)*s_loc + epoch/(epoch+1)*s_glob

        return (batch-m_glob)/s_glob,m_glob,s_glob

    def de_batch_normalization(self,batch,m_glob,s_glob):

        return batch*s_glob + m_glob

    def dropout_layer(self,batch,epoch):
        if epoch is None:
            epoch = 0
        key = random.PRNGKey(epoch)
        return random.bernoulli(key,self.dropout_rate,batch.shape)*batch

    def get_learning_rate_schedule(self,epoch=0):
        if self.learning_rate_schedule ==1:
            return np.exp(np.log(self.step_size)- (np.log(self.step_size) - np.log(self.step_size/100))/(self.num_batches*self.niter)*epoch)
        else:
            return self.step_size

    def get_noise_level_schedule(self,epoch=0):
        if self.noise_level_schedule ==1:
            return np.exp(np.log(self.noise_level)- (np.log(self.noise_level) - np.log(self.noise_level/100))/(self.num_batches*self.niter)*epoch)
        else:
            return self.noise_level

    def get_reg_parameter_schedule(self,epoch=0):
        if self.reg_parameter_schedule ==1:
            return np.exp(np.log(self.reg_parameter)- (np.log(self.reg_parameter) - np.log(self.reg_parameter/100))/(self.num_batches*self.niter)*epoch)
        else:
            return self.reg_parameter

    def save_model(self):

        Model = {"fname": self.fname,
                 "AnchorPoints": self.AnchorPoints,
                 "Params": self.Params,
                 "NSize": self.NSize,
                 "nlayers": self.nlayers,
                 "active_forward": self.active_forward,
                 "active_backward": self.active_backward,
                 "res_factor": self.res_factor,
                 "reg_parameter": self.reg_parameter,
                 "cost_weight": self.cost_weight,
                 "simplex": self.simplex,
                 "nneg_output": self.nneg_output,
                 "nneg_weights": self.nneg_weights,
                 "noise_level": self.noise_level,
                 "reg_inv": self.reg_inv,
                 "cost_type": self.cost_type,
                 "optim_learn": self.optim_learn,
                 "step_size": self.step_size,
                 "niter": self.niter,
                 "eps_cvg": self.eps_cvg,
                 "bn_param": self.bn_param,
                 "verb": self.verb,
                 "code_version": self.code_version,
                 "enable_train_bn":self.enable_train_bn}
        outfile = open(self.fname + '.pkl', 'wb')
        pickle.dump(Model, outfile)
        outfile.close()

    def get_optimizer(self, optim=None, stage='learn', step_size=None):

        if optim is None:
            if stage == 'learn':
                optim = self.optim_learn
            else:
                optim = self.optim_proj
        if step_size is None:
            step_size = self.get_learning_rate_schedule #self.step_size

        if optim == 1:
            if self.verb > 2:
                print("With momentum optimizer")
            opt_init, opt_update, get_params = momentum(step_size, mass=0.95)
        elif optim == 2:
            if self.verb > 2:
                print("With rmsprop optimizer")
            opt_init, opt_update, get_params = rmsprop(step_size, gamma=0.9, eps=1e-8)
        elif optim == 3:
            if self.verb > 2:
                print("With adagrad optimizer")
            opt_init, opt_update, get_params = adagrad(step_size, momentum=0.9)
        elif optim == 4:
            if self.verb > 2:
                print("With Nesterov optimizer")
            opt_init, opt_update, get_params = nesterov(step_size, 0.9)
        elif optim == 5:
            if self.verb > 2:
                print("With SGD optimizer")
            opt_init, opt_update, get_params = sgd(step_size)
        else:
            if self.verb > 2:
                print("With adam optimizer")
            opt_init, opt_update, get_params = adam(step_size)

        return opt_init, opt_update, get_params

    def encoder(self, X, W=None,epoch=None,InAP=None):

        if W is None:
            W = self.Params
        if epoch is None:
            apply_only = True
        else:
            apply_only = False

        PhiX = X
        ResidualX = X
        if InAP is not None:
            PhiE = InAP
            ResidualE = InAP
        else:
            PhiE = self.AnchorPoints
            ResidualE = self.AnchorPoints

        for l in range(self.nlayers):

            if self.dropout_rate is not None:   # Only on phiX???
                PhiX = self.dropout_layer(PhiX,epoch)

            PhiX,mg,sg = self.batch_normalization(PhiX,epoch,W["mu" + str(l)],W["std" + str(l)],apply_only=True) # Rescaling

            PhiX = self.activation_function(np.dot(PhiX, W["Wt" + str(l)]) + W["bt" + str(l)], direction='forward')
            PhiX += self.ResParams[l] * ResidualX

            #PhiE,mg,sg = self.batch_normalization(PhiE,epoch,self.bn_param["mean_bn_" + str(l)],self.bn_param["std_bn_" + str(l)],apply_only=True)
            PhiE,mg,sg = self.batch_normalization(PhiE,epoch,W["mu" + str(l)],W["std" + str(l)],apply_only=True) # Rescaling
            PhiE = self.activation_function(np.dot(PhiE, W["Wt" + str(l)]) + W["bt" + str(l)], direction='forward')
            PhiE += self.ResParams[l] * ResidualE

            ResidualX = PhiX
            ResidualE = PhiE

        return PhiX, PhiE

    def encode_anchor_points(self):

        X0 = onp.ones((1, onp.shape(self.AnchorPoints)[1]))  # arbitrary X, but necessary to use encoder method
        _, self.PhiE = self.encoder(X0)

    def decoder(self, B, W=None,epoch=None):

        if W is None:
            W = self.Params
        if epoch is None:
            epoch=0

        XRec = B
        ResidualR = B

        for l in range(self.nlayers):

            if self.dropout_rate is not None:   # Only on phiX???
                XRec = self.dropout_layer(XRec,epoch)

            XRec = self.activation_function(np.dot(XRec, W["Wp" + str(l)]) + W["bp" + str(l)], direction='backward')

            XRec += self.ResParams[-(l + 1)] * ResidualR

            ResidualR = XRec

        if self.nneg_output:
            XRec = XRec * (XRec > 0)

        return XRec

    def activation_function(self, X, direction='forward'):

        if direction == 'forward':
            active = self.active_forward
        else:
            active = self.active_backward

        if active == 'linear':
            Y = X
        elif active == 'Relu':
            Y = X * (X > 0)
        elif active == 'lRelu':
            Y1 = ((X > 0) * X)
            Y2 = ((X <= 0) * X * 0.01)  # with epsilon = 0.01
            Y = Y1 + Y2
        elif active == 'silu':
            Y = silu(X)
        elif active == 'softplus':
            Y = softplus(X)
        elif active == 'mish':
            Y = X*np.tanh(np.log(1.+np.exp(X)))
        else:
            Y = np.tanh(X)

        return Y

    def interpolator(self, PhiX, PhiE):

        Lambda = np.dot(PhiX, np.dot(PhiE.T, np.linalg.inv(np.dot(PhiE, PhiE.T) + self.reg_inv * onp.eye(self.num_anchor_points))))

        if self.simplex:
            Lambda = Lambda / (np.sum(Lambda, axis=1)[:, np.newaxis] + 1e-3)  # not really a projection on the simplex

        B = Lambda @ PhiE

        return B, Lambda

#################################
# Learning stage / main code
#################################

    def learning_stage(self, X, XValidation=None, niter=None,batch_size=None):
        """
        Learning the parameters
        """

        if batch_size is None:
            num_batches = 1
            batch_size = X.shape[0]

        num_complete_batches, leftover = divmod(X.shape[0], batch_size)
        num_batches = num_complete_batches + bool(leftover)
        self.num_batches = num_batches

        if XValidation is None:
            num_batches_test = num_batches
        else:
            num_complete_batches_test, leftover_test = divmod(XValidation.shape[0], batch_size)
            num_batches_test = num_complete_batches_test + bool(leftover_test)

        # Learning objective
        def learning_objective(W, XBatch,epoch):

            rng = random.PRNGKey(epoch)

            if self.noise_level is not None:
                batch = XBatch + self.get_noise_level_schedule(epoch) * random.normal(rng,shape=XBatch.shape) # We might need to change this
            else:
                PhiX, PhiE = self.encoder(XBatch, W=W,epoch=epoch)

            # Encode data and anchor points
            PhiX, PhiE = self.encoder(XBatch, W=W,epoch=epoch)

            # Define the barycenter
            B, Lambda = self.interpolator(PhiX, PhiE)

            # Decode the barycenter
            XRec = self.decoder(B, W=W,epoch=epoch)

            # Define the cost function - We could also consider others

            if self.cost_weight is None:
                cost1 = np.linalg.norm(PhiX - B)
                cost2 = self.get_reg_parameter_schedule(epoch) * np.linalg.norm(XRec - XBatch)
                cost = cost1 + cost2
            else:
                cost1 = np.linalg.norm(PhiX - B)
                cost2 = self.get_reg_parameter_schedule(epoch) * np.linalg.norm((XRec - XBatch) / self.cost_weight)
                cost = cost1 + cost2

            return cost, cost1, cost2

        # Learning stage

        opt_init, opt_update, get_params = self.get_optimizer(stage='learn')

        def get_batch(i,X):
            i = i % num_batches
            return lax.dynamic_slice_in_dim(X, i * self.batch_size, self.batch_size)

        def cost_objective(params, XBatch,epoch=0):
            cost, _, _ = learning_objective(params, XBatch,epoch)
            return cost

        @jit
        def update(it, XBatch, optstate,epoch):  # We could also use random batches as well
            params = get_params(optstate)
            return opt_update(it, grad(cost_objective)(params, XBatch,epoch), optstate)

        # Initializing the parameters
        initP = self.learnt_params_init()

        opt_state = opt_init(initP)

        out_val = []
        out_val1 = []
        out_val2 = []
        rel_acc = 0

        ste_time = time.time()
        average_epoch = onp.inf

        if niter is None:
            niter = self.niter

        for epoch in range(niter):
            # We should use vmap ...
            UPerm = onp.random.permutation(X.shape[0])  # For batch-based optimization
            cum_epoch = 0 # reset for each epoch
            for b in range(num_batches):
                batch = X[UPerm[b * batch_size:(b + 1) * batch_size], :]
                opt_state = update(epoch, batch, opt_state,cum_epoch)
                Params = get_params(opt_state)
                cum_epoch +=1
            if XValidation is not None:
                train_acc, train_acc1, train_acc2 = learning_objective(Params, XValidation,cum_epoch)
            else:
                train_acc, train_acc1, train_acc2 = learning_objective(Params, X,cum_epoch)
            out_val.append(train_acc)
            out_val1.append(train_acc1)
            out_val2.append(train_acc2)

            if epoch > 50:
                average_epoch = onp.mean(out_val[len(out_val) - 100:len(out_val) - 50])
                rel_acc = (abs(average_epoch - onp.mean(out_val[len(out_val) - 50::])) / (average_epoch + 1e-16))

            if onp.mod(epoch, 100) == 0:
                epoch_time = 0.01*(time.time()-ste_time) # Averaged over 100 it.
                ste_time = time.time()
                self.display(epoch,epoch_time,average_epoch,rel_acc)

        self.update_parameters(Params)
        if self.fname is not None:
            if self.verb > 2:
                print('Saving model...')
            self.save_model()
        self.encode_anchor_points()

        out_curves = {"total_cost": out_val, "trans_cost": out_val1, "samp_cost": out_val2}

        return self.Params, out_curves

    def fast_interpolation(self, X, Amplitude=None):

        """
        Quick forward-interpolation-backward estimation
        """

        if Amplitude is None:
            Amplitude = onp.sum(onp.abs(X), axis=1) / onp.mean(onp.sum(onp.abs(self.AnchorPoints), axis=1))
            estimate_amplitude = True
        else:
            estimate_amplitude = False
            if not hasattr(Amplitude, "__len__"):
                Amplitude = onp.ones(len(X)) * Amplitude

        # Encode data
        PhiX, _ = self.encoder(X / Amplitude[:, onp.newaxis])

        # Define the barycenter
        B, Lambda = self.interpolator(PhiX, self.PhiE)

        # Decode the barycenter
        XRec = self.decoder(B)

        if estimate_amplitude:
            Amplitude = onp.sum(XRec * X, axis=1) / onp.maximum(onp.sum(XRec ** 2, axis=1), 1e-3)

        XRec = XRec * Amplitude[:, onp.newaxis]

        Output = {"PhiX": PhiX, "PhiE": self.PhiE, "Barycenter": B, "Lambda": Lambda, "XRec": XRec,
                  "Amplitude": Amplitude}

        return Output

    ##############
    # Projection onto the barycentric span
    ##############

    def barycentric_span_projection(self, X, Amplitude=None, Lambda0=None, Amplitude0=None, niter=None, optim=None, step_size=None):

        """
        Project on the barycentric span.
        """

        if Lambda0 is None or (Amplitude0 is None and Amplitude is None):
            output = self.fast_interpolation(X=X, Amplitude=Amplitude)
            if Lambda0 is None:
                Lambda0 = output["Lambda"]
            if Amplitude0 is None and Amplitude is None:
                Amplitude0 = output["Amplitude"]
            if not hasattr(Amplitude0, "__len__") and Amplitude is None:
                Amplitude0 = onp.ones(len(X)) * Amplitude0
        if Amplitude is not None and not hasattr(Amplitude, "__len__"):
            Amplitude = onp.ones(len(X)) * Amplitude
        if niter is None:
            niter = self.niter
        if step_size is None:
            step_size = self.step_size

        Params = {}
        if not self.simplex:
            Params["Lambda"] = Lambda0
        else:  # if simplex constraint, optimization is performed on first dimensions of barycentric weights
            Params["Lambda"] = Lambda0[:, :-1]
        if Amplitude is None:
            Params["Amplitude"] = Amplitude0.copy()

        def get_cost(params):

            # Define the barycenter

            if not self.simplex:
                B = params["Lambda"] @ self.PhiE
            else:
                B = np.hstack((params["Lambda"], 1 - np.sum(params["Lambda"], axis=1)[:, np.newaxis])) @ self.PhiE

            XRec = self.decoder(B)

            if Amplitude is None:
                XRec = params["Amplitude"][:, np.newaxis] * XRec
            else:
                XRec = Amplitude[:, np.newaxis] * XRec

            return np.linalg.norm(XRec - X) ** 2

        opt_init, opt_update, get_params = self.get_optimizer(stage="project",step_size=step_size)

        @jit
        def update(i, OptState):
            params = get_params(OptState)
            return opt_update(i, grad(get_cost)(params), OptState)

        opt_state = opt_init(Params)
        train_acc_old = 1e32

        #t = trange(niter, desc='Projection - loss = %g, loss rel. var. = %g - ' % (0., 0.), disable=not self.verb)
        ste_time = time.time()

        for epoch in range(niter):

            opt_state = update(epoch, opt_state)
            Params = get_params(opt_state)
            train_acc = get_cost(Params)
            rel_acc = abs(train_acc_old - train_acc) / (train_acc_old + 1e-16)
            if rel_acc < self.eps_cvg:
                break
            train_acc_old = train_acc

            if onp.mod(epoch, 100) == 0:
                epoch_time = 0.01*(time.time() - ste_time)
                ste_time = time.time()
                self.display(epoch,epoch_time,train_acc,rel_acc,pref="BS projection - ")

        if self.verb:
            print("Finished in %i it. - loss = %g, loss rel. var. = %g " % (epoch, train_acc, rel_acc))

        if self.simplex:
            Params['Lambda'] = onp.hstack((Params["Lambda"], 1 - onp.sum(Params["Lambda"], axis=1)[:, onp.newaxis]))
        if Amplitude is not None:
            Params['Amplitude'] = Amplitude
        Params['XRec'] = self.get_barycenter(Params['Lambda'], Params['Amplitude'])

        return Params

        ####

    def get_barycenter(self, Lambda, Amplitude=None):

        """
        Get barycenter for a fixed Lambda
        """

        # Get barycenter
        B = Lambda @ self.PhiE

        # Decode barycenter
        XRec = self.decoder(B)

        if Amplitude is not None:
            XRec = Amplitude[:, onp.newaxis] * XRec

        return XRec

    def grad_AP(self, X, AP=None):
        """
        Learning the parameters
        """

        # Learning objective

        if AP is None:
            params = {"AP":self.AnchorPoints}
        else:
            params = {"AP":AP}

        def cost_objective(W):
            PhiX, PhiE = self.encoder(X,InAP=W["AP"])
            B, Lambda = self.interpolator(PhiX, PhiE)
            return np.linalg.norm(self.decoder(B) - X)**2

        return grad(cost_objective)(params)

    def grad_AP_latent(self, X, AP=None):
        """
        Learning the parameters
        """

        # Learning objective

        if AP is None:
            params = {"AP":self.AnchorPoints}
        else:
            params = {"AP":AP}

        def cost_objective(W):
            PhiX, PhiE = self.encoder(X,InAP=W["AP"])
            B, Lambda = self.interpolator(PhiX, PhiE)
            return np.linalg.norm(B - PhiX)**2

        return grad(cost_objective)(params)

#####################################
#
# Unrolling for fast BSP
#
#####################################

    def learn_unrolled_bsp(self, X, NLayers=5,Amplitude=None, Lambda0=None, Amplitude0=None, niter=None, optim=None, step_size=None):

        """
        Project on the barycentric span.
        """

        if Lambda0 is None or (Amplitude0 is None and Amplitude is None):
            output = self.fast_interpolation(X=X, Amplitude=Amplitude)
            if Lambda0 is None:
                Lambda0 = output["Lambda"]
            if Amplitude0 is None and Amplitude is None:
                Amplitude0 = output["Amplitude"]
            if not hasattr(Amplitude0, "__len__") and Amplitude is None:
                Amplitude0 = onp.ones(len(X)) * Amplitude0

        if Amplitude is not None and not hasattr(Amplitude, "__len__"):
            Amplitude = onp.ones(len(X)) * Amplitude
        if niter is None:
            niter = self.niter
        if step_size is None:
            step_size = self.step_size

        Params = {}
        if not self.simplex:
            Params["Lambda"] = Lambda0
        else:  # if simplex constraint, optimization is performed on first dimensions of barycentric weights
            Params["Lambda"] = Lambda0[:, :-1]
        if Amplitude is None:
            Params["Amplitude"] = Amplitude0.copy()

        def get_cost_bsp(params):

            # Define the barycenter

            lamb,amp = params

            if not self.simplex:
                B = lamb @ self.PhiE
            else:
                B = np.hstack((lamb, 1 - np.sum(lamb, axis=1)[:, np.newaxis])) @ self.PhiE

            XRec = self.decoder(B)

            if Amplitude is None:
                XRec = amp[:, np.newaxis] * XRec
            else:
                XRec = Amplitude[:, np.newaxis] * XRec

            return np.linalg.norm(XRec - X) ** 2

        get_grad = lambda lamb,amp: grad(get_cost_bsp)([lamb,amp])

        ### Now define the parameters and cost function for the unrolling:

        params_unroll = {}
        for r in range(NLayers):
            W0 = onp.random.rand(Lambda0.shape[1],Lambda0.shape[1])
            params_unroll["W"+onp.str(r)] = W0/onp.linalg.norm(W0) # Hessian-like
            params_unroll["alphaL"+onp.str(r)] = step_size # step size Lambda
            params_unroll["alphaA"+onp.str(r)] = step_size # step size Amplitude

        def unroll_model(X,Params_bsp,Params,Pgrad=None):
            amp = Params_bsp["Amplitude"]
            lamb = Params_bsp["Lambda"]
            for l in range(NLayers):
                # gradient
                if Pgrad is None:
                    g_lamb,g_amp = get_grad(lamb,amp)
                else:
                    g_lamb,g_amp = Pgrad[l+1]["grad"]
                # Update
                amp -= Params["alphaA"+onp.str(l)]*g_amp
                lamb -= Params["alphaL"+onp.str(l)]*np.dot(g_lamb,Params["W"+onp.str(l)])
            return amp,lamb

        def get_unroll_values(X,Params_bsp,Params):
            amp = Params_bsp["Amplitude"]
            lamb = Params_bsp["Lambda"]
            Pout=[]
            Q = {}
            Q["Amplitude"] = amp
            Q["Lambda"] = lamb
            Q["grad"] = []

            Pout.append(Q)
            for l in range(NLayers):
                # gradient
                g_lamb,g_amp = get_grad(lamb,amp)
                # Update
                amp -= 1./Params["alphaA"+onp.str(l)]*g_amp
                lamb -= 1./Params["alphaL"+onp.str(l)]*np.dot(g_lamb,Params["W"+onp.str(l)])
                Q = {}
                Q["Amplitude"] = amp
                Q["Lambda"] = lamb
                Q["grad"] = g_lamb,g_amp
                Pout.append(Q)
            return Pout

        # Cost for the unrolling

        @jit
        def cost_unrolling(params_unroll,X,Params_bsp,Pgrad):
            amp,lamb = unroll_model(X,Params_bsp,params_unroll,Pgrad)
            XRec = self.get_barycenter(lamb, amp)
            return np.linalg.norm(XRec - X) ** 2

        Pgrad = get_unroll_values(X,Params,params_unroll)
        g = grad(cost_unrolling)(params_unroll,X,Params,Pgrad)

        opt_init, opt_update, get_params = self.get_optimizer(stage="project",step_size=step_size)

        @jit
        def update(i, OptState,batch,Params,Pgrad):
            params = get_params(OptState)
            return opt_update(i, grad(cost_unrolling)(params,batch,Params,Pgrad), OptState)

        opt_state = opt_init(params_unroll)
        train_acc_old = 1e32

        #t = trange(niter, desc='Projection - loss = %g, loss rel. var. = %g - ' % (0., 0.), disable=not self.verb)
        ste_time = time.time()
        loss = []
        Pgrad = None

        for epoch in range(niter):
            # We should use batches here

            opt_state = update(epoch, opt_state,X,Params,Pgrad)
            P = get_params(opt_state)
            train_acc = cost_unrolling(P,X,Params,Pgrad)
            #Pgrad = get_unroll_values(X,Params,P)

            rel_acc = abs(train_acc_old - train_acc) / (train_acc_old + 1e-16)
            if rel_acc < self.eps_cvg:
                break
            train_acc_old = train_acc
            loss.append(train_acc)

            if onp.mod(epoch, 100) == 0:
                epoch_time = 0.01*(time.time() - ste_time)
                ste_time = time.time()
                self.display(epoch,epoch_time,train_acc,rel_acc,pref="BS projection - ")

        # # if self.verb:
        # #     print("Finished in %i it. - loss = %g, loss rel. var. = %g " % (epoch, train_acc, rel_acc))
        # #
        # # if self.simplex:
        # #     Params['Lambda'] = onp.hstack((Params["Lambda"], 1 - onp.sum(Params["Lambda"], axis=1)[:, onp.newaxis]))
        # # if Amplitude is not None:
        # #     Params['Amplitude'] = Amplitude
        # # Params['XRec'] = self.get_barycenter(Params['Lambda'], Params['Amplitude'])
        #
        return P,loss

    def barycentric_span_projection_fast(self, X, params_unroll=None,NLayers=5, Amplitude=None, Lambda0=None, Amplitude0=None):

        """
        Project on the barycentric span.
        """

        if Lambda0 is None or (Amplitude0 is None and Amplitude is None):
            output = self.fast_interpolation(X=X, Amplitude=Amplitude)
            if Lambda0 is None:
                Lambda0 = output["Lambda"]
            if Amplitude0 is None and Amplitude is None:
                Amplitude0 = output["Amplitude"]
            if not hasattr(Amplitude0, "__len__") and Amplitude is None:
                Amplitude0 = onp.ones(len(X)) * Amplitude0
        if Amplitude is not None and not hasattr(Amplitude, "__len__"):
            Amplitude = onp.ones(len(X)) * Amplitude

        Params = {}
        if not self.simplex:
            Params["Lambda"] = Lambda0
        else:  # if simplex constraint, optimization is performed on first dimensions of barycentric weights
            Params["Lambda"] = Lambda0[:, :-1]
        if Amplitude is None:
            Params["Amplitude"] = Amplitude0.copy()

        def get_cost_bsp(params):

            # Define the barycenter

            lamb,amp = params

            if not self.simplex:
                B = lamb @ self.PhiE
            else:
                B = np.hstack((lamb, 1 - np.sum(lamb, axis=1)[:, np.newaxis])) @ self.PhiE

            XRec = self.decoder(B)

            if Amplitude is None:
                XRec = amp[:, np.newaxis] * XRec
            else:
                XRec = Amplitude[:, np.newaxis] * XRec

            return np.linalg.norm(XRec - X) ** 2

        get_grad = lambda lamb,amp: grad(get_cost_bsp)([lamb,amp])

        def unroll_model(X,Params_bsp,Params):
            amp = Params_bsp["Amplitude"]
            lamb = Params_bsp["Lambda"]
            for l in range(NLayers):
                # gradient
                g_lamb,g_amp = get_grad(lamb,amp)
                # Update
                amp -= Params["alphaA"+onp.str(l)]*g_amp
                lamb -= Params["alphaL"+onp.str(l)]*np.dot(g_lamb,Params["W"+onp.str(l)])
            return amp,lamb

        Amp,Lamb = unroll_model(X,Params,params_unroll)

        XRec = self.get_barycenter(Lamb, Amp)

        return XRec,Lamb,Amp

#####################################
#
# An attempt to optimize the anchor points
#
#####################################

def optimize_anchor_points(X,Latent=False,Ortho=False,NumSearch=None,Model=None,niter=1000,step_size=1e-2,rho=1e-5,Strategy=None):

    lf = IAE(Model=Model)
    num_ap = lf.AnchorPoints.shape[0]

    for index in range(num_ap): # We need to do better than

        encode = lambda x: lf.encoder(x)[0]
        if NumSearch is not None:
            IndSearch = onp.random.permutation(X.shape[0])
            IndSearch = IndSearch[0:NumSearch]
            PhiX = encode(X[IndSearch,:])
        else:
            PhiX = encode(X)

        if Latent:
            g = lf.grad_AP_latent(X)["AP"][index,:].reshape(1,-1)
        else:
            g = lf.grad_AP(X)["AP"][index,:].reshape(1,-1)

        g = g/np.linalg.norm(g)
        AP = lf.AnchorPoints[index,:].reshape(1,-1)
        P = {"rho":rho}

        Q = np.dot(np.dot(np.transpose(lf.AnchorPoints),np.linalg.inv(np.dot(lf.AnchorPoints,np.transpose(lf.AnchorPoints)))),lf.AnchorPoints)

        def cost_optim(P):
            alpha = P["rho"]
            alpha = alpha*(alpha > 0)

            if Latent:
                a = encode(AP) - alpha*g
            else:
                a = encode(AP - alpha*g)

            #a = a/np.linalg.norm(a) # we need a normalization
            R = PhiX-a
            if Ortho:
                R = np.dot(R,Q)

            if Strategy is None:
                return np.min(np.linalg.norm(R,axis=1)**2,axis=0)
            elif Strategy == "mean":
                return np.mean(np.linalg.norm(R,axis=1)**2,axis=0)
            elif Strategy == "median":
                return np.median(np.linalg.norm(R,axis=1)**2,axis=0)

        opt_init, opt_update, get_params = lf.get_optimizer(stage="project",step_size=step_size)

        @jit
        def update(i, OptState):
            params = get_params(OptState)
            return opt_update(i, grad(cost_optim)(params), OptState)

        opt_state = opt_init(P)
        train_acc_old = 1e32

        #t = trange(niter, desc='Projection - loss = %g, loss rel. var. = %g - ' % (0., 0.), disable=not self.verb)
        ste_time = time.time()

        for epoch in range(niter):

            opt_state = update(epoch, opt_state)
            params = get_params(opt_state)
            train_acc = cost_optim(params)
            rel_acc = abs(train_acc_old - train_acc) / (train_acc_old + 1e-16)
            if rel_acc < lf.eps_cvg:
                break
            train_acc_old = train_acc

            if onp.mod(epoch, 100) == 0:
                epoch_time = 0.01*(time.time() - ste_time)
                ste_time = time.time()
                lf.display(epoch,epoch_time,train_acc,rel_acc,pref="AP optimisation - ")

        d = np.linalg.norm(PhiX-encode(AP - params["rho"]*g),axis=1)
        if index == 0:
            if NumSearch is not None:
                APout = X[IndSearch[onp.where(d == np.min(d))[0]],:]
            else:
                APout = X[onp.where(d == np.min(d))[0],:]
        else:
            if NumSearch is not None:
                APout = np.concatenate((APout,X[IndSearch[onp.where(d == np.min(d))[0]],:]),axis=0)
            else:
                APout = np.concatenate((APout,X[onp.where(d == np.min(d))[0],:]),axis=0)

    return APout

def dist_matrix(X):
    Y = X.T/onp.linalg.norm(X.T,axis=0)
    d = abs(onp.arccos(Y.T@Y*(1-1e-9)))
    return d

def medoid(X):
    sd = onp.sum(dist_matrix(X),axis=1)
    I = onp.where(sd == onp.min(sd))[0]
    return X[I,:]

def add_anchor_points(X,strategy=None,Model=None,perc = 10):

    lf = IAE(Model=Model)
    PhiX,PhiE = lf.encoder(X)
    Residual = PhiX - PhiX@PhiE.T@np.linalg.inv(PhiE@PhiE.T)@PhiE
    d = np.linalg.norm(Residual,axis=1)

    if strategy is None:
        AddAP = X[onp.where(d == np.max(d))[0],:]
    elif strategy == 'perc_mean':
        e_perc = np.percentile(d,1-perc) # We take the perc percent higher
        D = X[onp.where(d > e_perc)[0],:]
        AddAP = medoid(D)

    return AddAP # we could also take some k-mean

#####################################
#
# Learning with layer-wise pre-training
#
#####################################

def layer_wise_pretrain_IAE(X_train=None,XValidation=None,Model=None, fname='IAE_model', AnchorPoints=None, NSize=None, active_forward='lRelu',
             active_backward='tanH', res_factor=0.1, reg_parameter=1., cost_weight=None, reg_inv=1e-8,
             simplex=False, nneg_weights=False, nneg_output=False, noise_level=None, cost_type=0, optim_learn=0,
             optim_proj=3, step_size=1e-3,batch_size=100, niter=10000, eps_cvg=1e-9, verb=False,niter_lpw=2500,enable_train_bn=True):

    Model=None
    for nl in range(1,len(NSize)):
        lwp_NSize = NSize[0:nl+1]
        lwp_fname = fname+"temp_lwp" # At some point, we should remove these files
        print("#----------- nl: %d -----------#" % nl)
        learnfunc = IAE(Model=Model, fname=lwp_fname, AnchorPoints=AnchorPoints, NSize=lwp_NSize, active_forward=active_forward,
                     active_backward=active_backward, res_factor=res_factor, reg_parameter=reg_parameter, cost_weight=cost_weight, reg_inv=reg_inv,
                     simplex=simplex, nneg_weights=nneg_weights, nneg_output=nneg_output, noise_level=noise_level, cost_type=optim_learn, optim_learn=optim_learn,
                     optim_proj=optim_proj, step_size=step_size, niter=niter, eps_cvg=eps_cvg, verb=verb,enable_train_bn=enable_train_bn)
        out = learnfunc.learning_stage(X_train,XValidation=XValidation,niter=niter_lpw,batch_size=batch_size)
        Model = load_model(lwp_fname)

    Model["step_size"]=step_size/10.
    Model["optim_learn"]=3
    learnfunc = IAE(Model=Model, fname=fname, AnchorPoints=AnchorPoints, NSize=NSize, active_forward=active_forward,
                 active_backward=active_backward, res_factor=res_factor, reg_parameter=reg_parameter, cost_weight=cost_weight, reg_inv=reg_inv,
                 simplex=simplex, nneg_weights=nneg_weights, nneg_output=nneg_output, noise_level=noise_level, cost_type=optim_learn, optim_learn=optim_learn,
                 optim_proj=optim_proj, step_size=step_size, niter=niter, eps_cvg=eps_cvg, verb=verb,enable_train_bn=enable_train_bn)
    out = learnfunc.learning_stage(X_train,XValidation=XValidation,niter=niter,batch_size=batch_size)

    return out

#####################################
#
# Restart
#
#####################################

def IAE_Restart(X_train,X_test,X_dico,reg_parameter_schedule=None,learning_rate_schedule=None,num_steps=5,AnchorPoints=None,NL=2,NSize=None,niter=1000,name='test',step_size=1e-3,ShowResults=True,reg_parameter=1e3,Strategy=None):

    MSE_mean = []
    MSE_med = []
    Model=None

    for r in range(num_steps):

        fname=name
        learnfunc = IAE(Model=Model,reg_parameter_schedule=reg_parameter_schedule,learning_rate_schedule=learning_rate_schedule,step_size=step_size,reg_inv=1e-9, active_forward='mish',active_backward='sft',AnchorPoints=AnchorPoints, NSize=NSize,simplex=False, noise_level=1e-3, cost_type=0,niter=niter, eps_cvg=1e-9, verb=False,fname=fname,reg_parameter=reg_parameter)
        outref = learnfunc.learning_stage(X_train, XValidation=X_train,batch_size=100)


        outrec = learnfunc.barycentric_span_projection(X_test,niter=1000)["XRec"]
        e_ref = -20*onp.log10(onp.linalg.norm(outrec-X_test,axis=1)/onp.linalg.norm(X_test,axis=1))
        MSE_mean.append(onp.mean(e_ref))
        MSE_med.append(onp.median(e_ref))

        if ShowResults:
            plt.figure()
            plt.loglog(outref[1]["total_cost"],label=fname)
            plt.legend()
            plt.figure()
            out=plt.hist(e_ref,bins=100,label=fname)
            plt.title(onp.str(onp.mean(e_ref))+" / "+onp.str(onp.median(e_ref)))
            plt.legend()

        Model = load_model(fname)

    return MSE_mean,MSE_med

#####################################
#
# Anchor points optimization
#
#####################################

def IAE_OptimizedAP(X_train,X_test,X_dico,Ortho=False,NumSearch=20,simplex=False,reg_parameter_schedule=None,learning_rate_schedule=None,num_steps=5,AnchorPoints=None,NL=2,NSize=None,niter=1000,niter_in=1000,name='test',step_size=1e-3,ShowResults=True,rho=1e-3,reg_parameter=1e3,Strategy=None):


    OutAP=[]
    OutAP.append(AnchorPoints)
    MSE_mean = []
    MSE_med = []
    Model=None

    for r in range(num_steps):

        fname=name+"_step_"+onp.str(r)
        learnfunc = IAE(Model=Model,reg_parameter_schedule=reg_parameter_schedule,learning_rate_schedule=learning_rate_schedule,step_size=step_size,reg_inv=1e-9, active_forward='mish',active_backward='sft',AnchorPoints=AnchorPoints, NSize=NSize,simplex=simplex, noise_level=1e-3, cost_type=0,niter=niter, eps_cvg=1e-9, verb=False,fname=fname,reg_parameter=reg_parameter)
        outref = learnfunc.learning_stage(X_train, XValidation=X_train,batch_size=100)


        outrec = learnfunc.barycentric_span_projection(X_test,niter=1000)["XRec"]
        e_ref = -20*onp.log10(onp.linalg.norm(outrec-X_test,axis=1)/onp.linalg.norm(X_test,axis=1))
        MSE_mean.append(onp.mean(e_ref))
        MSE_med.append(onp.median(e_ref))

        if ShowResults:
            plt.figure()
            plt.loglog(outref[1]["total_cost"],label=fname)
            plt.legend()
            plt.figure()
            out=plt.hist(e_ref,bins=100,label=fname)
            plt.title(onp.str(onp.mean(e_ref))+" / "+onp.str(onp.median(e_ref)))
            plt.legend()

        AnchorPoints = optimize_anchor_points(X_dico,Ortho=Ortho,NumSearch=NumSearch,Model=load_model(fname),niter=niter_in,step_size=step_size,rho=rho,Strategy=Strategy)
        OutAP.append(AnchorPoints)
        Model = load_model(fname)
        Model["AnchorPoints"] = AnchorPoints

    return OutAP,MSE_mean,MSE_med

#####################################
#
# Anchor points optimization
#
#####################################

def IAE_OptimizedAP_andADD(X_train,X_test,X_dico,strategy=None,num_add=3,num_steps=3,noise_level=None,AnchorPoints=None,NL=2,NSize=None,niter=1000,name='test',step_size=1e-3,ShowResults=True,rho=1e-3,reg_parameter=1e3,nneg_output=False,learning_rate_schedule=None,reg_parameter_schedule=None):

    OutAP=[]
    OutAP.append(AnchorPoints)
    Model=None
    dim = AnchorPoints.shape[0]

    for d in range(num_add):
        for r in range(num_steps):

            fname=name+"_dim"+onp.str(dim)
            learnfunc = IAE(Model=Model,step_size=step_size,reg_inv=1e-9, active_forward='mish',active_backward='mish',AnchorPoints=AnchorPoints, NSize=NSize,simplex=False,nneg_output=nneg_output, noise_level=noise_level, cost_type=0,reg_parameter_schedule=reg_parameter_schedule,learning_rate_schedule=learning_rate_schedule,niter=niter, eps_cvg=1e-9, verb=False,fname=fname,reg_parameter=reg_parameter)
            outref = learnfunc.learning_stage(X_train, XValidation=X_train,batch_size=100)

            if ShowResults:
                plt.figure()
                plt.loglog(outref[1]["total_cost"],label=fname)
                plt.legend()
                outrec = learnfunc.barycentric_span_projection(X_test,niter=1000)["XRec"]
                e_ref = -20*onp.log10(onp.linalg.norm(outrec-X_test,axis=1)/onp.linalg.norm(X_test,axis=1))
                plt.figure()
                out=plt.hist(e_ref,bins=100,label=fname)
                plt.title(onp.str(onp.mean(e_ref))+" / "+onp.str(onp.median(e_ref)))
                plt.legend()

            AnchorPoints = optimize_anchor_points(X_dico,Model=load_model(fname),niter=1000,step_size=step_size,rho=rho)
            OutAP.append(AnchorPoints)
            Model = load_model(fname)
            Model["AnchorPoints"] = AnchorPoints

        AddAP = add_anchor_points(X_dico,Model=Model,strategy=strategy)
        Model["AnchorPoints"] = np.concatenate((AnchorPoints,AddAP),axis=0)
        OutAP.append(Model["AnchorPoints"] )
        dim += 1

    return OutAP
