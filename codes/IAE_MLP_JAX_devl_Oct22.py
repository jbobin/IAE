"""
IAE - generic - tensor mix or not ("couple dictionary learning") and potentially different sizes for the reconstruction

version 3 - November 22nd, 2021
author - J.Bobin
CEA-Saclay
"""

import pickle
from jax import grad, jit, lax
import jax.numpy as np
from jax.experimental.optimizers import adam, momentum, sgd, nesterov, adagrad, rmsprop
import numpy as onp
import time
import sys
import jax
from jax.nn import softplus,silu,elu
from jax import random
from jax import vmap
import matplotlib.pyplot as plt
from jax import jit, grad, lax, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, FanOut, Relu, Softplus


###################################################
# Elementary functions
###################################################

def load_model(fname):
    dataf = open(fname + '.pkl', 'rb')
    model = pickle.load(dataf)
    dataf.close()
    return model

def _normalize(X,norm='1',log=False):

    if len(X.shape) < 3:
        if log:
            Y = np.log10(X.T)
        else:
            Y = X.T
        if norm == '1':
            Y = Y/np.sum(abs(Y),axis=0)
        if norm == '2':
            Y = Y/np.sqrt(np.sum(Y**2,axis=0))
        if norm == 'inf':
            Y = 0.5*Y/np.max(Y,axis=0)
        return Y.T
    else:
        if log:
            Y = np.log10(np.swapaxes(X,0,2))
        else:
            Y = np.swapaxes(X,0,2)
        if norm == '1':
            Y = Y/np.sum(abs(Y),axis=(0,1))
        if norm == '2':
            Y = Y/np.sqrt(np.sum(Y**2,axis=(0,1)))
        if norm == 'inf':
            Y = 0.5*Y/np.max(abs(Y),axis=(0,1))
        if norm == 'glob':
            Y = Y/np.max(np.sum(abs(Y),axis=(0,1))) # global normalisation
        return np.swapaxes(Y,0,2)

############################################################
# Main code
############################################################

class IAE(object):
    """
    Model - input IAE model, overrides other parameters if provided (except the number of layers)

    fname - filename for the IAE model

    AnchorPoints - anchor points

    NSizeIn - encoder network structure (e.g. [8, 8, 8, 8] for a 3-layer neural network of size 8)
    NSizeOut - decoder network structure; if not set, NSizeOut=NSizeIn

    active_forward - activation function in the encoder
    active_backward - activation function in the decoder

    fulltensor - if True, implements a dense tensor network
    resnet - if True, implements a residual network
    res_factor - residual injection factor in the ResNet-like architecture
    sparse_code - if set, enforces the sparsity of the code space
    niter_sparse - number of iterations of the sparse approximation step (as part of the interpolator) - default is 10

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
    batch_norm - if set, applies batch normalization
    learn_scaling - learn an affine batch transform (part of batch normalization)
    init_weights - option for the weight initialization, if 1, l2 normalization, if 2 Xavier initialization
    dropout_rate - if set, applies dropout with rate given by dropout_rate
    reg_parameter_schedule - scheduler for the regularization parameter (last if 1e-2xfirts)
    learning_rate_schedule - scheduler for the learning rate (last if 1e-2xfirts)
    noise_level_schedule- scheduler for the noise level (last if 1e-2xfirts)
    eps_cvg - convergence tolerance w.r.t. the validation set

    verb - verbose mode
    code_version - current version of the code
    """

    ####
    #--- Displaying the loss value
    ####

    # TBD-TBD --> change the parsing procedure to make it simpler

    def __init__(self,fulltensor=False,resnet=False, Model=None, fname='IAE_model', AnchorPoints=None, Normalisation = 'inf', NSizeIn=None, NSizeOut=None, active_forward='mish',
                 active_backward='mish', res_factor=0.1, reg_parameter=1000., cost_weight=None, reg_inv=1e-9,
                 simplex=False, nneg_weights=False, nneg_output=False, noise_level=None, cost_type=0, optim_learn=0,
                 optim_proj=3,learn_scaling=True,init_weights=1,sparse_code=False,niter_sparse=10, step_size=1e-2, niter=5000, eps_cvg=1e-9, verb=False, batch_norm=True, dropout_rate=None,reg_parameter_schedule=0,learning_rate_schedule=0,noise_level_schedule=0,
                 code_version="version_3_NOV_22th_2021"):
        """
        Initialization
        """

        self.Model = Model
        self.fulltensor = fulltensor
        self.resnet = resnet
        self.fname = fname
        self.AnchorPoints = AnchorPoints
        self.num_anchor_points = None
        self.Params = {}
        self.PhiE = None
        self.NSizeIn = NSizeIn
        if NSizeIn is None:
            self.NSizeOut = NSizeIn # Pick the same as NSizeIn if not given
        else:
            self.NSizeOut = NSizeOut
        self.nlayersIn = None
        self.nlayersOut = None
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
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.noise_level_schedule = noise_level_schedule
        self.num_batches=1
        self.reg_parameter_schedule = reg_parameter_schedule
        self.sparse_code = sparse_code
        self.niter_sparse = niter_sparse
        self.init_weights = init_weights
        self.dim = None
        self.learn_scaling=learn_scaling
        self.batch_size=100
        self.BatchStats = {}
        self.DefNetworkParams = ["fname",
                 "AnchorPoints",
                 "Params",
                 "NSizeIn",
                 "NSizeOut",
                 "nlayersIn",
                 "nlayersOut",
                 "fulltensor",
                 "active_forward",
                 "active_backward",
                 "res_factor",
                 "reg_parameter",
                 "cost_weight",
                 "simplex",
                 "nneg_output",
                 "nneg_weights",
                 "noise_level",
                 "reg_inv",
                 "cost_type",
                 "optim_learn",
                 "step_size",
                 "niter",
                 "eps_cvg",
                 "verb",
                 "code_version",
                 "batch_norm",
                 "sparse_code",
                 "niter_sparse",
                 "BatchStats",
                 "learn_scaling"]

        self.init_parameters()
        self.Normalisation = Normalisation

    ####
    #--- Displaying the loss value
    ####

    def display(self,epoch,epoch_time,name_acc,train_acc,rel_acc,pref="Learning stage - ",niter=None):

        """
        Function to display losses
        """

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

        loss_val = ''
        for r in range(len(name_acc)):
            loss_val+='-- '+name_acc[r]+' = {0:e}'.format(onp.float(train_acc[r]))

        sys.stdout.write('\033[2K\033[1G')
        if epoch_time > 1:
            #print(pref+'epoch {0}'.format(epoch)+'/' +onp.str(niter)+ ' -- loss  = {0:e}'.format(onp.float(train_acc)) + ' -- loss rel. var. = {0:e}'.format(onp.float(rel_acc))+bar+time_run+'-{0:0.4} '.format(epoch_time)+' s/epoch', end="\r")
            print(pref+'epoch {0}'.format(epoch)+'/' +onp.str(niter)+ loss_val + ' -- loss rel. var. = {0:e}'.format(onp.float(rel_acc))+bar+time_run+'-{0:0.4} '.format(epoch_time)+' s/epoch', end="\r")
        if epoch_time < 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +onp.str(niter)+ loss_val + ' -- loss rel. var. = {0:e}'.format(onp.float(rel_acc))+bar+time_run+'-{0:0.4}'.format(1./epoch_time)+' epoch/s', end="\r")

    ####
    #--- Initializing the parameters
    ####

    def init_parameters(self):
        """
        Initialize the parameters
        """

        if self.verb:
            # HERE WE SHOULD LIST MOST OPTIONS
            print("code version : ",self.code_version)
            print("fname = ",self.fname)
            print("Encoder arch.:")
            print(self.NSizeIn)
            print("Decoder arch.:")
            print(self.NSizeOut)
            if self.fulltensor:
                print("Full dense tensor")
            if self.resnet:
                print("Residual network")
            print("simplex constraint : ",self.simplex)
            print("Activation - forward: ",self.active_forward)
            print("Activation - backward: ",self.active_backward)
            print("Reg. parameter:",self.reg_parameter)
            print("Non-negative weights:",self.nneg_weights)
            print("Non-negative weights:",self.nneg_output)
            print("Noise level:",self.noise_level)
            print("Optim. strat. at training",self.optim_learn)
            print("Optim. strat. at proj.",self.optim_proj)
            print("Step size:",self.step_size)
            print("Number of iterations",self.niter)
            print("Batch norm.",self.batch_norm)
            print("Learn affine transf. in BN:",self.learn_scaling)
            if self.sparse_code:
                print("Sparse code",self.sparse_code)
                print("Nb of it. for sparse coding",self.niter_sparse)
            print("Init. weights:",self.init_weights)

        ## --- NSize

        if self.NSizeIn is None:
            if self.Model is None:
                print("Hey, there's a problem, provide either NSizeIn or Model !")
            else:
                self.NSizeIn = self.Model["NSizeIn"]
        self.nlayersIn = self.NSizeIn.shape[0]- 1
        self.dim = self.NSizeIn[0,1] # just the third dim of the input (channel length

        if self.NSizeOut is None:
            if self.Model is None:
                self.NSizeOut = self.NSizeIn #Take the same as NSizeIn
            else:
                self.NSizeOut = self.Model["NSizeOut"]
        self.nlayersOut = self.NSizeOut.shape[0] - 1

        ## --- The AnchorPoints

        if self.AnchorPoints is None:
            if self.Model is None:
                print("Hey, there's a problem, provide either AnchorPoints or Model !")
            else:
                self.AnchorPoints = self.Model["AnchorPoints"]
        self.num_anchor_points = self.AnchorPoints.shape[0]

        ## -- Initialize the weights
        ## ENCODER

        for j in range(self.nlayersIn):

            if self.fulltensor:
                W0 = onp.random.randn(self.NSizeIn[j,0], self.NSizeIn[j + 1,0],self.NSizeIn[j,1], self.NSizeIn[j + 1,1])
                b0 = onp.zeros((self.NSizeIn[j + 1,0],self.NSizeIn[j + 1,1]))
            else:
                W0 = onp.random.randn(self.NSizeIn[j,0], self.NSizeIn[j + 1,0],self.dim)
                b0 = onp.zeros((self.NSizeIn[j + 1,0],self.dim))

            if self.init_weights==1:
                W0 = W0 / onp.linalg.norm(W0)
            elif self.init_weights==2:
                W0 = onp.sqrt(2./(self.NSizeIn[j,0]*self.NSizeIn[j+1,1] + self.NSizeIn[j+1,0]*self.NSizeIn[j+1,1]))*W0

            self.Params["Wt" + str(j)] = W0
            self.Params["bt" + str(j)] = b0

            ## batch normalization parameters
            self.BatchStats["mean_bn_t" + str(j)] = np.zeros((1,))
            self.BatchStats["std_bn_t" + str(j)] = np.ones((1,))
            self.Params["mut" + str(j)] = np.zeros((1,))
            self.Params["stdt" + str(j)] = np.ones((1,))

        ## -- Initialize the weights
        ## DECODER

        for j in range(self.nlayersOut):

            if self.fulltensor:
                W0 = onp.random.randn(self.NSizeOut[-j - 1,0], self.NSizeOut[-j - 2,0],self.NSizeOut[-j - 1,1], self.NSizeOut[-j - 2,1])
                b0 = onp.zeros((self.NSizeOut[-j - 2,0], self.NSizeOut[-j - 2,1]))
            else:
                W0 = onp.random.randn(self.NSizeOut[-j - 1,0], self.NSizeOut[-j - 2,0],self.dim)
                b0 = onp.zeros((self.NSizeOut[-j - 2,0],self.dim))

            if self.init_weights==1:
                W0 = W0 / onp.linalg.norm(W0)
            elif self.init_weights==2:
                W0 = onp.sqrt(2./(self.NSizeOut[j,0]*self.NSizeOut[j+1,1] + self.NSizeOut[j+1,0]*self.NSizeOut[j+1,1]))*W0

            self.Params["Wp" + str(j)] = W0
            self.Params["bp" + str(j)] = b0

            ## batch normalization parameters
            self.BatchStats["mean_bn_p" + str(j)] = np.zeros((1,))
            self.BatchStats["std_bn_p" + str(j)] = np.ones((1,))
            self.Params["mup" + str(j)] = np.zeros((1,))
            self.Params["stdp" + str(j)] = np.ones((1,))

        ## --- Parameters for sparse coding

        if self.sparse_code:
            self.Params["thd"] =  -6*np.ones((self.num_anchor_points ,))
            for i in range(self.niter_sparse):
                self.Params["step_size_"+onp.str(i)] =  np.zeros((1 ,))

        if self.Model is not None: #--- Load from an existing model

            if self.verb > 2:
                print("Load from an existing model")

            if self.Model["code_version"] != self.code_version:
                print('Compatibility warning!')

            dL = self.nlayersIn - self.Model["nlayersIn"]
            for j in range(self.Model["nlayersIn"]):
                self.BatchStats["mean_bn_t" + str(j)] = self.Model["BatchStats"]["mean_bn_t" + str(j)]
                self.BatchStats["std_bn_t" + str(j)] = self.Model["BatchStats"]["std_bn_t" + str(j)]
                self.Params["Wt" + str(j)] = self.Model["Params"]["Wt" + str(j)]
                self.Params["bt" + str(j)] = self.Model["Params"]["bt" + str(j)]
                self.Params["mut" + str(j)] = self.Model["Params"]["mut" + str(j)]
                self.Params["stdt" + str(j)] = self.Model["Params"]["stdt" + str(j)]

            dL = self.nlayersOut - self.Model["nlayersOut"]
            for j in range(self.Model["nlayersOut"]):
                self.BatchStats["mean_bn_p" + str(j)] = self.Model["BatchStats"]["mean_bn_p" + str(j)]
                self.BatchStats["std_bn_p" + str(j)] = self.Model["BatchStats"]["std_bn_p" + str(j)]
                self.Params["Wp" + str(j + dL)] = self.Model["Params"]["Wp" + str(j)]
                self.Params["bp" + str(j + dL)] = self.Model["Params"]["bp" + str(j)]
                self.Params["mup" + str(j)] = self.Model["Params"]["mup" + str(j)]
                self.Params["stdp" + str(j)] = self.Model["Params"]["stdp" + str(j)]

            if self.sparse_code:
                self.Params["thd"] =   self.Model["Params"]["thd"] # May not be possible if we restart from a different dimensionality
                for i in range(self.niter_sparse):
                    self.Params["step_size_"+onp.str(i)] =  self.Model["Params"]["step_size_"+onp.str(i)]

            for key in self.DefNetworkParams:  # Get all the other parameters
                if (key != 'Params') or (key != 'BatchStats'):
                    setattr(self,key,self.Model[key])

        self.ResParams = self.res_factor * (2 ** (onp.arange(self.nlayersIn) / self.nlayersIn) - 1) ## We should check that as well

        self.encode_anchor_points() #--- Encode the anchor points

        if self.Model is None:  #--- create the model if needed
            self.Model = {}
            for key in self.DefNetworkParams:
                self.Model[key] = getattr(self,key)

    def save_model(self):
        """
        Saving the model
        """

        outfile = open(self.fname + '.pkl', 'wb')
        pickle.dump(self.Model, outfile)
        outfile.close()

    def update_parameters(self, Params):
        """
        Update the trainable parameters
        """

        for j in range(self.nlayersIn):
            self.Params["Wt" + str(j)] = Params["Wt" + str(j)]
            self.Params["bt" + str(j)] = Params["bt" + str(j)]
            if self.learn_scaling:
                self.Params["mut" + str(j)] = Params["mut" + str(j)]
                self.Params["stdt" + str(j)] = Params["stdt" + str(j)]

        for j in range(self.nlayersOut):
            self.Params["Wp" + str(j)] = Params["Wp" + str(j)]
            self.Params["bp" + str(j)] = Params["bp" + str(j)]
            if self.learn_scaling:
                self.Params["mup" + str(j)] = Params["mup" + str(j)]
                self.Params["stdp" + str(j)] = Params["stdp" + str(j)]

        if self.sparse_code:
            self.Params["thd"] = Params["thd"]
            for i in range(self.niter_sparse):
                self.Params["step_size_"+onp.str(i)] = Params["step_size_"+onp.str(i)]

    def learnt_params_init(self):
        """
        Initialize the trainable parameters  // this could be updating based on what we want to learn (e.g. greedy layerwise training)
        """

        Params = {}

        for j in range(self.nlayersIn):
            Params["Wt" + str(j)] = self.Params["Wt" + str(j)]
            Params["bt" + str(j)] = self.Params["bt" + str(j)]
            if self.learn_scaling:
                Params["mut" + str(j)] = self.Params["mut" + str(j)]
                Params["stdt" + str(j)] = self.Params["stdt" + str(j)]

        for j in range(self.nlayersOut):
            Params["Wp" + str(j)] = self.Params["Wp" + str(j)]
            Params["bp" + str(j)] = self.Params["bp" + str(j)]
            if self.learn_scaling:
                Params["mup" + str(j)] = self.Params["mup" + str(j)]
                Params["stdp" + str(j)] = self.Params["stdp" + str(j)]

        if self.sparse_code:
            Params["thd"] = self.Params["thd"]
            for i in range(self.niter_sparse):
                Params["step_size_"+onp.str(i)] = self.Params["step_size_"+onp.str(i)]

        return Params

    ###
    #- Model-related functions
    ###

    def dropout_layer(self,batch,epoch):  # Useful ????
        """
        Dropout code
        """
        if epoch is None:
            epoch = 0
        key = random.PRNGKey(epoch)
        return random.bernoulli(key,self.dropout_rate,batch.shape)*batch

    ### Below are defined schedulers

    def get_learning_rate_schedule(self,epoch=0):
        """
        Learning rate scheduler
        """
        if self.learning_rate_schedule ==1:
            return np.exp(np.log(self.step_size)- (np.log(self.step_size) - np.log(self.step_size/100))/(self.num_batches*self.niter)*epoch)
        else:
            return self.step_size

    def get_noise_level_schedule(self,epoch=0):
        """
        Noise level scheduler
        """
        if self.noise_level_schedule ==1:
            return np.exp(np.log(self.noise_level)- (np.log(self.noise_level) - np.log(self.noise_level/100))/(self.num_batches*self.niter)*epoch)
        else:
            return self.noise_level

    def get_reg_parameter_schedule(self,epoch=0):
        """
        Regularization parameter scheduler
        """
        if self.reg_parameter_schedule ==1:
            return np.exp(np.log(self.reg_parameter)- (np.log(self.reg_parameter) - np.log(self.reg_parameter/100))/(self.num_batches*self.niter)*epoch)
        else:
            return self.reg_parameter

    ##-- Defining the optimizer

    def get_optimizer(self, optim=None, stage='learn', step_size=None):

        """
        Defines the optimizer
        """

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

    def encoder(self, X, W=None,epoch=None,in_AnchorPoints=None):  # We should concatenate PhiX and PhiE

        """
        ENCODER
        """

        if W is None:
            W = self.Params
        if epoch is None:
            apply_only = True
        else:
            apply_only = False

        if in_AnchorPoints is not None:
            PhiXE = np.concatenate((X, in_AnchorPoints), axis=0)
        else:
            PhiXE = np.concatenate((X, self.AnchorPoints), axis=0)

        if self.resnet:
            ResidualXE = PhiXE

        for l in range(self.nlayersIn):

            # Batch normalization

            if self.batch_norm:

                if apply_only:
                    mean = self.BatchStats["mean_bn_t" + str(l)]
                    std = self.BatchStats["std_bn_t" + str(l)]
                else:
                    mean = np.mean(PhiXE, axis=(0, 1))  # mean and var calculated over PhiX only
                    std = np.sqrt(np.var(PhiXE, axis=(0, 1)))
                    self.BatchStats["mean_bn_t" + str(l)] = mean
                    self.BatchStats["std_bn_t" + str(l)] = std

                PhiXE = (PhiXE - mean) /(std + 1e-9)

                if self.learn_scaling:
                    PhiXE = PhiXE * W["stdt" + str(l)] + W["mut" + str(l)]

            # Dropout ????

            if self.dropout_rate is not None:   # Only on phiX???
                PhiXE = self.dropout_layer(PhiXE,epoch)

            # Update PhiXE

            if self.fulltensor:
                PhiXE = self.activation_function(np.einsum('ijk,jlkh->ilh', PhiXE, W["Wt" + str(l)])+ W["bt" + str(l)], direction='forward')
            else:
                PhiXE = self.activation_function(np.einsum('ijk,jlk->ilk', PhiXE, W["Wt" + str(l)])+ W["bt" + str(l)], direction='forward')

            if self.resnet:
                PhiXE += self.ResParams[l] * ResidualXE
                ResidualXE = PhiXE

        return PhiXE[:len(X), :, :], PhiXE[len(X):, :, :]
        #return PhiX, PhiE

    def decoder(self, B, W=None,epoch=None):

        """
        DECODER
        """

        if W is None:
            W = self.Params
        if epoch is None:
            epoch=0
            apply_only = True
        else:
            apply_only = False

        XRec = B
        if self.resnet:
            ResidualR = B

        for l in range(self.nlayersOut):

            # Batch normalization

            if self.batch_norm:

                if apply_only:
                    mean = self.BatchStats["mean_bn_p" + str(l)]
                    std = self.BatchStats["std_bn_p" + str(l)]
                else:
                    mean = np.mean(XRec, axis=(0, 1))  # mean and var calculated over PhiX only
                    std = np.sqrt(np.var(XRec, axis=(0, 1)))
                    self.BatchStats["mean_bn_p" + str(l)] = mean
                    self.BatchStats["std_bn_p" + str(l)] = std

                XRec = (XRec - mean) /(std + 1e-9)

                if self.learn_scaling:
                    XRec = XRec * W["stdp" + str(l)] + W["mup" + str(l)]

            if self.dropout_rate is not None:   # Only on phiX???
                XRec = self.dropout_layer(XRec,epoch)

            if self.fulltensor:
                XRec = self.activation_function(np.einsum('ijk,jlkh->ilh', XRec, W["Wp" + str(l)])+ W["bp" + str(l)], direction='backward')
            else:
                XRec = self.activation_function(np.einsum('ijk,jlk->ilk', XRec, W["Wp" + str(l)])+ W["bp" + str(l)], direction='backward')

            if self.resnet:
                XRec += self.ResParams[-(l + 1)] * ResidualR
                ResidualR = XRec

        if self.nneg_output:
            XRec = XRec * (XRec > 0)

        return XRec

    def activation_function(self, X, direction='forward'):

        """
        Activation functions
        """

        if direction == 'forward':
            active = self.active_forward
        else:
            active = self.active_backward

        if active == 'linear':
            return X
        elif active == 'Relu':
            return X * (X > 0)
        elif active == 'lRelu':
            Y1 = ((X > 0) * X)
            Y2 = ((X <= 0) * X * 0.01)  # with epsilon = 0.01
            return Y1 + Y2
        elif active == 'silu':
            return silu(X)
        elif active == 'softplus':
            return softplus(X)
        elif active == 'mish':
            return X*np.tanh(np.log(1.+np.exp(X)))
        elif active == 'sft':
            epsilon = 0.7
            return epsilon*(X**2)*np.tanh(X)+(1-epsilon)*X
        elif active == 'elu':
            return elu(X)
        else:
            return np.tanh(X)

    def interpolator(self, PhiX, PhiE,W=None):

        PhiE2 = np.einsum('ijk,ljk -> il',PhiE, PhiE)

        #if self.sparse_code:
        Not = False
        if Not:

            Z = np.einsum('ijk,ljk,lm', PhiX, PhiE, np.diag(1./np.diag(PhiE2))) # Not very clean
            if W is None:
                for r in range(self.niter_sparse):
                    Z = Z - np.exp(self.Params["step_size_"+onp.str(r)])/np.linalg.norm(PhiE2)*np.einsum('ijk,ljk',np.tensordot(Z, PhiE, axes=(1, 0))-PhiX,PhiE) # Pas forcément top, au minimum faire un FISTA
                Z1 = Z-np.exp(self.Params["thd"])
                Z2  = -Z-np.exp(self.Params["thd"])
                Lambda = Z1*(Z1 > 0) - Z2*(Z2>0)
            else:
                for r in range(self.niter_sparse):
                    Z = Z - np.exp(W["step_size_"+onp.str(r)])/np.linalg.norm(PhiE2)*np.einsum('ijk,ljk',np.tensordot(Z, PhiE, axes=(1, 0))-PhiX,PhiE) # Pas forcément top, au minimum faire un FISTA
                Z1 = Z-np.exp(W["thd"])
                Z2  = -Z-np.exp(W["thd"])
                Lambda = Z1*(Z1 > 0) - Z2*(Z2>0)
            if self.nneg_weights:
                Lambda = Lambda*(Lambda>0)
            if self.simplex:
                Lambda = Lambda / (np.sum(np.abs(Lambda), axis=1)[:, np.newaxis] + 1e-3)  # not really a projection on the simplex

        else:

            iPhiE = np.linalg.inv(PhiE2 + self.reg_inv * onp.eye(self.num_anchor_points))
            Lambda = np.einsum('ijk,ljk,lm', PhiX, PhiE, iPhiE)
            ones = np.ones((1,Lambda.shape[1]))

            if self.sparse_code:
                u = np.sqrt(np.sum(np.square(Lambda),0))
                if W is None:
                    w = np.maximum(1 - self.Params["thd"]/(u+1e-16),0)
                else:
                    w = np.maximum(1 - np.exp(W["thd"])/(u+1e-16),0)
                Lambda = Lambda*w

            if self.nneg_weights:

                mu = np.max(Lambda,1)-1.0
    ##
                for i in range(self.num_anchor_points) :
                    F = np.sum(np.maximum(Lambda,mu.reshape(-1,1)), 1) - self.num_anchor_points*mu-1
                    mu = mu + 2/self.num_anchor_points*F

                Lambda = np.maximum(Lambda,mu.reshape(-1,1))-mu.reshape(-1,1)

            elif self.simplex:

                ones = np.ones_like(Lambda)
                mu = (1 - np.sum(Lambda,1))/np.sum(iPhiE)
                Lambda = Lambda +  np.einsum('ij,i -> ij',np.einsum('ij,jk -> ik', ones, iPhiE),mu)

        return np.tensordot(Lambda, PhiE, axes=(1, 0)),Lambda

    def encode_anchor_points(self):

        """
        Encode the AnchorPoints only to get PhiE
        """

        X0 = onp.ones((1, onp.shape(self.AnchorPoints)[1],self.dim))  # arbitrary X, but necessary to use encoder method
        _, self.PhiE = self.encoder(X0)

#################################
# Learning stage / main code
#################################

    def learning_stage(self, X, XValidation=None, niter=None,batch_size=None,logloss=False):
        """
        Learning the parameters // Xout are the data to be reconstructed
        -> TBD define inputs/outputs
        """

        if batch_size is None:
            num_batches = 1
            batch_size = X.shape[0]

        num_complete_batches, leftover = divmod(X.shape[0], batch_size)
        num_batches = num_complete_batches + bool(leftover)
        self.num_batches = num_batches
        self.batch_size = batch_size

        if XValidation is None:
            num_batches_test = num_batches
        else:
            num_complete_batches_test, leftover_test = divmod(XValidation.shape[0], batch_size)
            num_batches_test = num_complete_batches_test + bool(leftover_test)

        # Learning objective
        def learning_objective(W, XBatch,epoch):

            rng = random.PRNGKey(epoch)

            # Encode data and anchor points
            if self.noise_level is not None:
                batch = XBatch + self.get_noise_level_schedule(epoch) * random.normal(rng,shape=XBatch.shape) # We might need to change this
                PhiX, PhiE = self.encoder(batch, W=W,epoch=epoch)
            else:
                PhiX, PhiE = self.encoder(XBatch, W=W,epoch=epoch)

            # Define the barycenter
            B, Lambda = self.interpolator(PhiX, PhiE,W=W)

            # Decode the barycenter
            XRec = self.decoder(B, W=W,epoch=epoch)

            # Define the cost function - We could also consider others

            if logloss:
                if self.cost_weight is None:
                    cost1 = np.log(np.square(np.linalg.norm(PhiX - B)))
                    cost2 =np.log(np.square(np.linalg.norm(XRec - XBatch)))
                    cost = (cost1 + cost2)
                else:
                    cost1 = np.log(np.square(np.linalg.norm(PhiX - B)))
                    cost2 = np.log(np.square(np.linalg.norm((XRec - XBatch) / self.cost_weight)))
                    cost = (cost1 + cost2)
            else:
                if self.cost_weight is None:
                    cost1 = np.linalg.norm(PhiX - B)
                    cost2 = self.get_reg_parameter_schedule(epoch) * np.linalg.norm(XRec - XBatch)
                    cost = (cost1 + cost2)/(1+self.get_reg_parameter_schedule(epoch))
                else:
                    cost1 = np.linalg.norm(PhiX - B)
                    cost2 = self.get_reg_parameter_schedule(epoch) * np.linalg.norm((XRec - XBatch) / self.cost_weight)
                    cost = (cost1 + cost2)/(1+self.get_reg_parameter_schedule(epoch))

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

        #######

        out_val2 = []
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
                batch = X[UPerm[b * batch_size:(b + 1) * batch_size], :,:]
                opt_state = update(epoch, batch, opt_state,cum_epoch)
                cum_epoch +=1

            Params = get_params(opt_state)

            train_loss, train_acc1, train_acc2 = learning_objective(Params, X,0)

            if XValidation is not None:
                train_acc, train_acc1, train_acc2 = learning_objective(Params, XValidation,0)
            else:
                train_acc = train_loss

            out_val.append(train_acc)
            out_val1.append(train_acc1)
            out_val2.append(train_acc2)

            if epoch > 50:
                average_epoch = onp.mean(out_val[len(out_val) - 100:len(out_val) - 50])
                rel_acc = (abs(average_epoch - onp.mean(out_val[len(out_val) - 50::])) / (average_epoch + 1e-16))

            if onp.mod(epoch, 100) == 0:
                epoch_time = 0.01*(time.time()-ste_time) # Averaged over 100 it.
                ste_time = time.time()
                self.display(epoch,epoch_time,["training loss","validation loss"],[train_loss,average_epoch],rel_acc)

        self.update_parameters(Params)
        if self.fname is not None:
            if self.verb > 2:
                print('Saving model...')
            self.save_model()
        self.encode_anchor_points()

        out_curves = {"total_cost": out_val, "trans_cost": out_val1, "samp_cost": out_val2}

        return self.Params, out_curves

    ###
    #- Below are estimation codes / once the model has been trained
    ###

    def fast_interpolation(self, X, Amplitude=None):

        """
        Quick forward-interpolation-backward estimation
        """

        norm = self.Normalisation

        if Amplitude is None: # Quick amplitude estimator // needs to be updated
            Amplitude = self.Init_Amplitude(X,norm=norm)
        else:
            if not hasattr(Amplitude, "__len__"):
                Amplitude = onp.ones(len(X)) * Amplitude

        # Encode data
        PhiX, _ = self.encoder(X / Amplitude[:, onp.newaxis, onp.newaxis])

        # Define the barycenter
        B, Lambda = self.interpolator(PhiX, self.PhiE)

        # Decode the barycenter
        XRec = self.decoder(B)

        Amplitude = np.sum(XRec*X,(1,2))/np.sum(XRec*XRec,(1,2))

        XRec = XRec * Amplitude[:, onp.newaxis, onp.newaxis]

        Output = {"PhiX": PhiX, "PhiE": self.PhiE, "Barycenter": B, "Lambda": Lambda, "XRec": XRec,
                  "Amplitude": Amplitude}

        return Output

    ##############
    # Init_amplitude
    ##############

    def Init_Amplitude(self,X,norm='1'):

        if norm == '1':
            Amplitude = np.sum(np.abs(X),(1,2))#.squeeze()
        elif norm =='2':
            Amplitude = np.sqrt(np.sum(np.square(X),(1,2)))#.squeeze())
        elif norm=='inf':
            Amplitude = 2*np.max(np.abs(X),(1,2))#.squeeze()

        return Amplitude

    ##############
    # Projection onto the barycentric span
    ##############

    def barycentric_span_projection(self, X, Lambda=None, Amplitude=None, niter=None, optim=None, step_size=None):

        """
        Project on the barycentric span.
        """

        if Lambda is None or Amplitude is None:
            output = self.fast_interpolation(X=X, Amplitude=Amplitude) # Could be replaced by some mean/median value or sampling from the training set or K-medoid?
            if Lambda is None:
                Lambda = output["Lambda"]
            if Amplitude is None:
                Amplitude = output["Amplitude"]

        if niter is None:
            niter = self.niter
        if step_size is None:
            step_size = self.step_size

        Params = {}
        Params["Lambda"] = Lambda
        Params["Amplitude"] = Amplitude

        PhiE2 = np.einsum('ijk,ljk -> il',self.PhiE, self.PhiE)
        iPhiE = np.linalg.inv(PhiE2 + self.reg_inv * onp.eye(self.num_anchor_points))

        def get_cost(params):

            # Define the barycenter

            Lambda = params["Lambda"]

            if self.nneg_weights:
                mu = np.max(Lambda,1)-1.0
                for i in range(self.num_anchor_points) :
                    F = np.sum(np.maximum(Lambda,mu.reshape(-1,1)), 1) - self.num_anchor_points*mu-1
                    mu = mu + 2/self.num_anchor_points*F
                Lambda = np.maximum(Lambda,mu.reshape(-1,1))-mu.reshape(-1,1)

            elif self.simplex:

                ones = np.ones_like(Lambda)
                mu = (1 - np.sum(Lambda,1))/np.sum(iPhiE)
                Lambda = Lambda +  np.einsum('ij,i -> ij',np.einsum('ij,jk -> ik', ones, iPhiE),mu)

            B = np.tensordot(Lambda, self.PhiE, axes=(1, 0))

            XRec = self.decoder(B)

            if Amplitude is None:
                XRec = params["Amplitude"][:, np.newaxis, onp.newaxis] * XRec
            else:
                XRec = Amplitude[:, np.newaxis, onp.newaxis] * XRec

            return np.linalg.norm(XRec - X) ** 2

        opt_init, opt_update, get_params = self.get_optimizer(stage="project",step_size=step_size)

        @jit
        def update(i, OptState):
            params = get_params(OptState)
            return opt_update(i, grad(get_cost)(params), OptState)

        opt_state = opt_init(Params)
        train_acc_old = 1e32
        ste_time = time.time()

        for epoch in range(niter):  ## We might include batches + vmap

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
                self.display(epoch,epoch_time,["Loss"],[train_acc],rel_acc,pref="BSP -- ")

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
        B = np.tensordot(Lambda, self.PhiE, axes=(1, 0))

        # Decode barycenter
        XRec = self.decoder(B)

        if Amplitude is not None:
            XRec = Amplitude[:, onp.newaxis, onp.newaxis] * XRec

        return XRec
