"""
IAE - generic - tensor mix or not ("couple dictionary learning") and potentially different sizes for the reconstruction

version 3 - January, 28 2021
author - J.Bobin
CEA-Saclay

What changes:
    - Add an extra dimension reduction

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
from jax.example_libraries import stax
#from jax.experimental import stax

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

    NSize - encoder network structure (e.g. [8, 8, 8, 8] for a 3-layer neural network of size 8)
    NSizeOut - decoder network structure; if not set, NSizeOut=NSize

    active_forward - activation function in the encoder
    active_backward - activation function in the decoder

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

    def __init__(self,input_dim = None,rho_latcon=0.5, Model=None, fname='IAE_model', AnchorPoints=None, NSize=None, active_forward='mish',
                 active_backward='mish', reg_parameter=1000., cost_weight=None, reg_inv=1e-9,
                 simplex=False, nneg_weights=False, nneg_output=False, noise_level=None, cost_type=0, optim_learn=0,
                 optim_proj=3,init_weights=1,sparse_code=False,niter_sparse=10, step_size=1e-2, niter=5000, eps_cvg=1e-9, verb=False, batch_norm=True, dropout_rate=None,reg_parameter_schedule=0,learning_rate_schedule=0,noise_level_schedule=0,
                 code_version="version_3_Dec_13th_2021"):
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
        self.nlayers = None
        self.active_forward = active_forward
        self.active_backward = active_backward
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
        self.input_dim = None
        self.batch_size=100
        self.BatchStats = {}
        self.Dims = None
        self.Padding = None
        self.params_count = None
        self.encode = None
        self.decode = None
        self.encode_lat = None
        self.decode_lat = None
        self.encode_arch = None
        self.decode_arch = None
        self.encode_lat_arch = None
        self.decode_lat_arch = None
        self.rho_latcon = rho_latcon
        self.input_dim = input_dim
        self.initP = None
        self.DimsEnc=None
        self.bsp_stat = None
        self.fast_bsp_params = None
        self.DefNetworkParams = ["fname",
                 "rho_latcon",
                 "AnchorPoints",
                 "Params",
                 "NSize",
                 "nlayers",
                 "active_forward",
                 "active_backward",
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
                 "Dims",
                 "DimsEnc",
                 "Padding"]

        self.init_parameters()

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
            print('-- RUNNING THE IAE INITIALIZATION --')
            print(' ')

        ## --- NSize

        if self.NSize is None:
            if self.Model is None:
                print("Hey, there's a problem, provide either NSize or Model !")
            else:
                self.NSize = self.Model["NSize"]
        self.nlayers = self.NSize.shape[0]

        if not hasattr(self.rho_latcon, "__len__"): # if is a scalar
            self.rho_latcon = self.rho_latcon*onp.ones((self.nlayers-1,))

        ## --- The AnchorPoints

        if self.AnchorPoints is None:
            if self.Model is None:
                print("Hey, there's a problem, provide either AnchorPoints or Model !")
            else:
                self.AnchorPoints = self.Model["AnchorPoints"]
        self.num_anchor_points = self.AnchorPoints.shape[0]
        self.generate_padding()

        ## --- Input dimension

        self.input_dim = [onp.shape(self.AnchorPoints)[1],onp.shape(self.AnchorPoints)[2]]

        ## -- Initialize the weights
        ## ENCODER

        def create_block(filter_size,nfilters,stride,padding="SAME",transpose=False,batchnorm=True,ndim=None): ## TO BE CLEANED
            if transpose:
                if batchnorm:
                    return stax.serial(stax.GeneralConvTranspose(('NHC', 'HIO', 'NHC'),nfilters,  (filter_size,), strides=(1,),  padding="SAME"),stax.BatchNorm(),stax.Elu)
                else:
                    return stax.serial(stax.GeneralConvTranspose(('NHC', 'HIO', 'NHC'),nfilters,  (filter_size,), strides=(1,), padding="SAME"),stax.Elu)
            else:
                if batchnorm:
                    if stride > 1:
                        return stax.serial(stax.GeneralConv(('NHC', 'HIO', 'NHC'),nfilters,  (filter_size,), strides=(stride,), padding=padding),stax.BatchNorm(),stax.MaxPool((stride,)))
                    else:
                        return stax.serial(stax.GeneralConv(('NHC', 'HIO', 'NHC'),nfilters,  (filter_size,), strides=(stride,), padding=padding),stax.BatchNorm(),stax.Elu)
                else:
                    if stride > 1:
                        return stax.serial(stax.GeneralConv(('NHC', 'HIO', 'NHC'),nfilters,  (filter_size,), strides=(stride,), padding=padding),stax.BatchNorm(),stax.MaxPool((stride,)))
                    else:
                        return stax.serial(stax.GeneralConv(('NHC', 'HIO', 'NHC'),nfilters,  (filter_size,), strides=(stride,), padding=padding),stax.BatchNorm(),stax.Elu)

        # ENCODER ARCHITECTURE

        encode_arch = []
        for j in range(self.nlayers):
            f = create_block(self.NSize[j,0],self.NSize[j,2],self.NSize[j,1],"SAME")
            encode_arch.append(f)

        # DECODER ARCHITECTURE

        decode_arch = []
        j=0
        decode_arch.append(stax.serial(stax.GeneralConvTranspose(('NHC', 'HIO', 'NHC'),self.NSize[self.nlayers-j-2,2],  (self.NSize[self.nlayers-j-1,0],), strides=(1,), padding="SAME"),stax.Elu))

        for j in range(1,self.nlayers-1):
            f = create_block(self.NSize[self.nlayers-j-1,0],self.NSize[self.nlayers-j-2,2],1,"SAME",transpose=True)
            decode_arch.append(f)

        j = self.nlayers-1
        decode_arch.append(stax.serial(stax.GeneralConvTranspose(('NHC', 'HIO', 'NHC'),self.input_dim[1],  (self.NSize[self.nlayers-j-1,0],), strides=(1,), padding="SAME"),stax.Elu))

        # LATERAL CONNECTIONS ARCHITECTURE

        encode_lat_arch=[]
        for j in range(1,self.nlayers):
            f = create_block(self.NSize[j,0],self.NSize[j,2],self.NSize[j,1],"SAME")
            encode_lat_arch.append(f)

        decode_lat_arch=[]
        for j in range(self.nlayers-1):
            f = create_block(self.NSize[self.nlayers-j-1,0],self.NSize[self.nlayers-j-2,2],self.NSize[self.nlayers-j-1,1],"SAME",transpose=True)
            decode_lat_arch.append(f)

        self.encode_arch = encode_arch
        self.decode_arch = decode_arch
        self.encode_lat_arch = encode_lat_arch
        self.decode_lat_arch = decode_lat_arch

        ## --- Parameters for sparse coding

        self.Params={}
        #if self.sparse_code:
        for i in range(self.niter_sparse):
            self.Params["step_size_"+onp.str(i)] =  -1.*np.ones((1 ,))
            self.Params["thd_"+onp.str(i)] =  -3*np.ones((self.num_anchor_points ,)) # Very small threshold at the beginning

        encode = []
        init_encoder_params = []
        encode_lat = []
        init_encoder_lat_params = []
        decode = []
        init_decoder_params = []
        decode_lat = []
        init_decoder_lat_params = []

        ndims = []
        dims = [1,self.input_dim[0],self.input_dim[1]]
        DimsEnc = [self.input_dim[0]]

        if self.verb:
            print('-- IAE architecture --')

        for r in range(0,self.nlayers):
            # encoder
            rng = random.PRNGKey(0)
            f_init, f = self.encode_arch[r]
            encode.append(f)
            _, p = f_init(rng, (dims[0],dims[1],dims[2]))
            init_encoder_params.append(p)
            temp = f(p,np.zeros(dims))
            dims = np.shape(temp)
            DimsEnc.append(dims[1])
            if self.verb:
                print("encoder out #",r,' - ',dims)
            if r < self.nlayers-1:
                #enc_lat_con
                f_init, f = self.encode_lat_arch[r]
                encode_lat.append(f)
                _, p = f_init(rng, dims)
                init_encoder_lat_params.append(p)
                temp = f(p,np.zeros((dims[0],dims[1],dims[2])))
                if self.verb:
                    print("encoder lat out #",r,' - ',np.shape(temp))
                ndims.append(np.shape(temp))
        ndims = ndims[::-1]
        DimsEnc = DimsEnc[::-1]
        self.DimsEnc = DimsEnc

        for r in range(0,self.nlayers):

            # decoder
            rng = random.PRNGKey(0)
            f_init, f = self.decode_arch[r]
            decode.append(f)
            _, p = f_init(rng, (dims[0],dims[1],dims[2]))
            init_decoder_params.append(p)
            temp = f(p,np.zeros(dims))
            dims = np.shape(temp)
            if self.verb:
                print("decoder out #",r,' - ',np.shape(temp))

            if r < self.nlayers-1:
                #dec_lat_con
                f_init, f = self.decode_lat_arch[r]
                decode_lat.append(f)
                _, p = f_init(rng, (ndims[r][0],ndims[r][1],ndims[r][2]))
                init_decoder_lat_params.append(p)
                temp = f(p,np.zeros((ndims[r][0],ndims[r][1],ndims[r][2])))
                if self.verb:
                    print("decoder lat out #",r,' - ',np.shape(temp))

        self.encode=encode
        self.decode=decode
        self.encode_lat=encode_lat
        self.decode_lat=decode_lat

        self.initP = init_encoder_params, init_decoder_params,init_encoder_lat_params, init_decoder_lat_params,self.learnt_params_init()
        self.Params = self.initP

        if self.Model is not None: #--- Load from an existing model

            if self.verb > 2:
                if self.verb:
                    print("Load from an existing model")

            if self.Model["code_version"] != self.code_version:
                print('Compatibility warning!')

            self.Params = self.Model["Params"]
            self.initP = self.Model["Params"]

            for key in self.DefNetworkParams:  # Get all the other parameters
                if (key != 'Params'):
                    setattr(self,key,self.Model[key])

        if self.Model is None:  #--- create the model if needed
            self.Model = {}
            for key in self.DefNetworkParams:
                self.Model[key] = getattr(self,key)

        if self.verb:
            # HERE WE SHOULD LIST MOST OPTIONS
            print(' ')
            print('-- LIST OF OPTIONS --')
            print("code version : ",self.code_version)
            print("fname = ",self.fname)
            print("Encoder arch.:")
            print("Filter lengths:",self.NSize[:,0])
            print("Stride:",self.NSize[:,1])
            print("Filter numbers:",self.NSize[:,2])
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
            if self.sparse_code:
                print("Sparse code",self.sparse_code)
                print("Nb of it. for sparse coding",self.niter_sparse)
            print("Init. weights:",self.init_weights)
            print(' ')

    def generate_padding(self):
        """
        Fill self.Padding array according to the network architecture. This method is called by the constructor.

        Returns
        -------
        None
        """
        dim = [onp.shape(self.AnchorPoints)[1]]
        for r in range(self.nlayers):
            dim.append(int(onp.ceil(dim[-1] / self.NSize[r,1])))
        dim = dim[::-1]
        self.Dims = dim

        self.Padding = []
        for r in range(self.nlayers):
            if dim[r] == dim[r + 1]:
                self.Padding.append('SAME')
            else:
                p = int(dim[r + 1] - self.NSize[-(r + 1),1] * (dim[r] - 1) + self.NSize[-(r + 1),0] - 2)
                if p % 2 == 0:
                    self.Padding.append(((p // 2, p // 2),))
                else:
                    self.Padding.append((((p + 1) // 2, (p - 1) // 2),))


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
        self.Params = Params
        self.Model["Params"] = Params

    def learnt_params_init(self):
        """
        Initialize the trainable parameters  // this could be updating based on what we want to learn (e.g. greedy layerwise training)
        """

        Params = {}
        for i in range(self.niter_sparse):
            Params["step_size_"+onp.str(i)] = self.Params["step_size_"+onp.str(i)]
            Params["thd_"+onp.str(i)] = self.Params["thd_"+onp.str(i)]

        return Params

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

    ###
    #- Model-related functions
    ###

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
            step_size = 1e-3 #self.step_size

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
            We,_,Wlat,_,_ = self.Params
        else:
            We,Wlat = W

        if in_AnchorPoints is not None:
            PhiXE = np.concatenate((X, in_AnchorPoints), axis=0)
        else:
            PhiXE = np.concatenate((X, self.AnchorPoints), axis=0)

        PhiX_lat = []
        PhiE_lat = []

        for r in range(self.nlayers-1):
            #PhiXE = self.activation_function(self.encode[r](We[r],PhiXE),direction='forward')
            #lPhiXE = self.activation_function(self.encode_lat[r](Wlat[r],PhiXE),direction='forward')
            PhiXE = self.encode[r](We[r],PhiXE)
            lPhiXE = self.encode_lat[r](Wlat[r],PhiXE)
            PhiX_lat.append(lPhiXE[:len(X), :, :])
            PhiE_lat.append(lPhiXE[len(X):, :, :])
        #PhiXE = self.activation_function(self.encode[self.nlayers-1](We[self.nlayers-1],PhiXE),direction='forward')
        PhiXE = self.encode[self.nlayers-1](We[self.nlayers-1],PhiXE)
        PhiX_lat.append(PhiXE[:len(X), :, :])
        PhiE_lat.append(PhiXE[len(X):, :, :])

        return PhiX_lat, PhiE_lat

    def decoder(self, B, W=None,epoch=None):

        """
        DECODER
        """
        if W is None:
            _,Wd,_,Wlat,_ = self.Params
        else:
            Wd,Wlat=W
        if epoch is None:
            epoch=0
            apply_only = True
        else:
            apply_only = False

        XRec = B[0]

        for r in range(self.nlayers-1):
            temp = jax.image.resize(XRec, (XRec.shape[0],self.DimsEnc[r+1],XRec.shape[2]), "nearest")
            #XRec = self.activation_function(self.decode[r](Wd[r],temp),direction='backward')
            XRec = self.decode[r](Wd[r],temp)
            temp = jax.image.resize(XRec, (B[r+1].shape[0],self.DimsEnc[r+1],B[r+1].shape[2]), "nearest")
            #lXRec = self.activation_function(self.decode_lat[r](Wlat[r],temp),direction='backward')
            lXRec = self.decode_lat[r](Wlat[r],temp)
            XRec += self.rho_latcon[r]*lXRec # Add what goes through the lateral connection
        temp = jax.image.resize(XRec, (XRec.shape[0],self.DimsEnc[self.nlayers],XRec.shape[2]), "nearest")
        XRec = self.decode[self.nlayers-1](Wd[self.nlayers-1],temp)

        return XRec

    def interpolator(self, PhiX, PhiE,W=None):

        if W is None:
            _,_,_,_,W = self.Params

        L = []

        for r in range(1,self.nlayers+1):

            PhiE2 = np.einsum('ijk,ljk -> il',PhiE[self.nlayers-r], PhiE[self.nlayers-r])
            iPhiE = np.linalg.inv(PhiE2 + self.reg_inv * onp.eye(self.num_anchor_points))
            Lambda = np.einsum('ijk,ljk,lm', PhiX[self.nlayers-r], PhiE[self.nlayers-r], iPhiE)

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

            L.append(Lambda)

        B = [np.tensordot(L[r-1], PhiE[self.nlayers-r], axes=(1, 0)) for r in range(1,self.nlayers+1)]

        return B,L

    def encode_anchor_points(self,W):

        """
        Encode the AnchorPoints only to get PhiE
        """

        X0 = onp.ones((1, onp.shape(self.AnchorPoints)[1],self.input_dim[1]))  # arbitrary X, but necessary to use encoder method
        _, self.PhiE = self.encoder(W,X0)

#################################
# Learning stage / main code
#################################

    def learning_stage(self, X, X_out=None, XValidation=None,XValidation_out=None, niter=None,batch_size=None,get_bsp_stat=True):
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

        # Initializing

        initP = self.initP #init_encoder_params, init_decoder_params,init_encoder_lat_params, init_decoder_lat_params,self.learnt_params_init()

        # Learning objective
        def learning_objective(W, XBatch,epoch):

            rng = random.PRNGKey(epoch)

            encoder_params, decoder_params,encoder_lat_params,decoder_lat_params,Wint = W
            Wenc = encoder_params,encoder_lat_params
            Wdec = decoder_params,decoder_lat_params

            # Encode data and anchor points
            if self.noise_level is not None:
                batch = XBatch + self.noise_level * random.normal(rng,shape=XBatch.shape) # We might need to change this
                PhiX, PhiE= self.encoder(batch, W=Wenc,epoch=epoch)
            else:
                PhiX, PhiE = self.encoder(XBatch, W=Wenc,epoch=epoch)

            # Define the barycenter
            B,_ = self.interpolator(PhiX, PhiE,W=Wint)

            # Decode the barycenter
            XRec = self.decoder(B, W=Wdec,epoch=epoch)

            # Define the cost function - We could also consider others

            if self.cost_weight is None:
                cost1 = self.reg_parameter/(1+self.reg_parameter)*np.linalg.norm(PhiX[self.nlayers-1] - B[0])/XBatch.shape[0]
                for r in range(1,self.nlayers):
                    cost1 += self.rho_latcon[r-1]*self.reg_parameter/(1+self.reg_parameter)*np.linalg.norm(PhiX[self.nlayers-r-1] - B[r])/XBatch.shape[0]
                cost2 = 1/(1+self.reg_parameter) * np.linalg.norm(XRec - XBatch)/XBatch.shape[0]
                cost = (cost1 + cost2)
            else:
                cost1 = self.reg_parameter/(1+self.reg_parameter)*np.linalg.norm(PhiX[self.nlayers-1] - B[0])/XBatch.shape[0]
                for r in range(1,self.nlayers):
                    cost1 += self.rho_latcon[r-1]*self.reg_parameter/(1+self.reg_parameter)*np.linalg.norm(PhiX[self.nlayers-r-1] - B[r])/XBatch.shape[0]
                cost2 = 1/(1+self.reg_parameter) * np.linalg.norm((XRec - XBatch)/ self.cost_weight)/XBatch.shape[0]
                cost = (cost1 + cost2)

            return cost, cost1, cost2

        # Learning stage

        opt_init, opt_update, get_params = self.get_optimizer(stage='learn')

        def cost_objective(params, XBatch):
            cost, _, _ = learning_objective(params, XBatch,0)
            return cost

        @jit
        def update(it, XBatch, optstate,epoch):  # We could also use random batches as well
            params = get_params(optstate)
            g = grad(cost_objective)(params, XBatch)
            return opt_update(it, g, optstate),g

        opt_state = opt_init(initP)

        out_val2 = []
        out_val = []
        out_val1 = []
        out_val2 = []
        rel_acc = 0

        ste_time = time.time()
        average_epoch = onp.inf

        if niter is None:
            niter = self.niter

        epoch = 0
        GoOn = True

        while GoOn:
            epoch += 1
            if epoch > niter:
                GoOn=False
            #for epoch in range(niter):
            # We should use vmap ...
            UPerm = onp.random.permutation(X.shape[0])  # For batch-based optimization
            cum_epoch = 0 # reset for each epoch

            for b in range(num_batches):
                batch = X[UPerm[b * batch_size:(b + 1) * batch_size], :,:]
                opt_state,_ = update(epoch, batch, opt_state,cum_epoch)
                cum_epoch +=1

            Params = get_params(opt_state)

            if X_out is None:
                train_loss, train_acc1, train_acc2 = learning_objective(Params, X,0)
            else:
                train_loss, train_acc1, train_acc2 = learning_objective(Params, X,0)

            if XValidation is not None:
                if XValidation_out is not None:
                    train_acc, train_acc1, train_acc2 = learning_objective(Params, XValidation,0)
                else:
                    train_acc, train_acc1, train_acc2 = learning_objective(Params, XValidation,0)
            else:
                train_acc = train_loss

            out_val.append(train_acc)
            out_val1.append(train_acc1)
            out_val2.append(train_acc2)

            if onp.isnan(train_acc):
                print("Stop learning because encountered NaNs")
                GoOn = False

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
        self.PhiE = self.encoder(self.AnchorPoints)

        if get_bsp_stat:
            print("Get BSP stats")
            Lambda = self.barycentric_span_projection(X,niter=2500,optim=4)["Lambda"]
            self.bsp_stat = {"mean":onp.mean(Lambda,axis=0),"median":onp.median(Lambda,axis=0),"std":onp.std(Lambda,axis=0),"min":onp.min(Lambda,axis=0),"max":onp.max(Lambda,axis=0)}

        out_curves = {"total_cost": out_val, "trans_cost": out_val1, "samp_cost": out_val2}

        return self.Params, out_curves

    ###
    #- Below are estimation codes / once the model has been trained
    ###

    def fast_interpolation(self, X, Amplitude=None,norm='1'):

        """
        Quick forward-interpolation-backward estimation
        """

        if Amplitude is None: # Quick amplitude estimator // needs to be updated
            Amplitude = self.Init_Amplitude(X,norm=norm)
        else:
            if not hasattr(Amplitude, "__len__"):
                Amplitude = onp.ones(len(X)) * Amplitude

        # Encode data

        self.PhiE,_ = self.encoder(self.AnchorPoints)

        PhiX, _ = self.encoder(X / Amplitude[:, onp.newaxis, onp.newaxis])

        # Define the barycenter
        B,Lambda= self.interpolator(PhiX, self.PhiE)

        # Decode the barycenter
        XRec = self.decoder(B)

        #if estimate_amplitude:  #-- We might need to re-estimate the amplitude
        #    Amplitude = onp.sum(XRec * X, axis=(1,2)) / onp.maximum(onp.sum(XRec ** 2, axis=(1,2)), 1e-3)

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

    def barycentric_span_projection(self, X, Amplitude=None, Lambda=None, niter=None, optim=None, step_size=None,norm='1'):

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

        if Amplitude is None:
            Params["Amplitude"] = Amplitude
        self.PhiE,_ = self.encoder(self.AnchorPoints)

        PhiE2 = [np.einsum('ijk,ljk -> il',r,r) for r in self.PhiE]
        iPhiE = [np.linalg.inv(r + self.reg_inv * onp.eye(self.num_anchor_points)) for r in PhiE2]

        def get_cost(params):

            # Define the barycenter

            L = params["Lambda"]

            if self.nneg_weights:
                for r in range(len(L)):
                    Lambda = L[r]
                    mu = np.max(Lambda,1)-1.0
                    for i in range(self.num_anchor_points) :
                        F = np.sum(np.maximum(Lambda,mu.reshape(-1,1)), 1) - self.num_anchor_points*mu-1
                        mu = mu + 2/self.num_anchor_points*F
                    Lambda = np.maximum(Lambda,mu.reshape(-1,1))-mu.reshape(-1,1)
                    L[r] = Lambda

            elif self.simplex:
                for r in range(len(L)):
                    Lambda = L[r]
                    ones = np.ones_like(Lambda)
                    mu = (1 - np.sum(Lambda,1))/np.sum(iPhiE)
                    Lambda = Lambda +  np.einsum('ij,i -> ij',np.einsum('ij,jk -> ik', ones, iPhiE),mu)
                    L[r] = Lambda

            B = [np.tensordot(L[r], self.PhiE[self.nlayers-r-1], axes=(1, 0)) for r in range(self.nlayers)]

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

        self.PhiE,_ = self.encoder(self.AnchorPoints)

        # Get barycenter
        B = [np.tensordot(Lambda[r], self.PhiE[self.nlayers-r-1], axes=(1, 0)) for r in range(self.nlayers)]

        # Decode barycenter
        XRec = self.decoder(B)

        if Amplitude is not None:
            XRec = Amplitude[:, onp.newaxis, onp.newaxis] * XRec

        return XRec
