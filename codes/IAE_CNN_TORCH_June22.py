"""
Metric Learning

TBD :
    - Add striding and padding
    - Add lateral connections
    - Add positivity
"""

import pickle
import numpy as np
import time
import sys
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Adam,NAdam,AdamW, SGD, Adagrad, LBFGS,ASGD
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from ray import tune
from ray.tune.schedulers import ASHAScheduler

############################################################
# Main code
############################################################

def _get_optimizer(Optimizer,parameters,learning_rate=1e-4):

    if Optimizer == 0:
        print("Adam")
        optimizer = Adam(parameters, lr=learning_rate)
    elif Optimizer == 1:
        print("AdamW")
        optimizer = AdamW(parameters, lr=learning_rate)
    elif Optimizer == 2:
        print("NAdam")
        optimizer = NAdam(parameters, lr=learning_rate)
    elif Optimizer == 3:
        print("Adagrad")
        optimizer = Adagrad(parameters, lr=learning_rate, weight_decay=1e-5)
    elif Optimizer == 4:
        print("SGD")
        optimizer = SGD(parameters, lr=learning_rate)
    elif Optimizer == 5:
        print("ASGD")
        optimizer = ASGD(parameters, lr=learning_rate)

    return optimizer

############################################################
# Saving model from dict
############################################################

def save_model(model,fname='test'):

    params = {"AnchorPoints":model.anchorpoints, "NSize":model.NSize,"reg_parameter":model.reg_parameter,"rho_latcon":model.rho_latcon,"noise_level":model.noise_level,"niter_sparse":model.niter_sparse,"sparse_code":model.sparse_code,"simplex" : model.simplex,"device":model.device,"GaussNoise":model.GaussNoise,"PositiveWeights":model.PositiveWeights}

    torch.save({"model":model.state_dict(),"iae_params":params}, fname+".pth")

############################################################
# Loading model from dict
############################################################

def load_model(fname,device="cpu"):

    model_in = torch.load(fname+".pth", map_location=device)
    params = model_in["iae_params"]
    model_state = model_in["model"]

    iae = IAE(AnchorPoints=params["AnchorPoints"], NSize=params["NSize"],reg_parameter=params["reg_parameter"],rho_latcon=params["rho_latcon"], noise_level=params["noise_level"],niter_sparse = params["niter_sparse"],sparse_code = params["sparse_code"],simplex = params["simplex"],device=device,GaussNoise=params["GaussNoise"],PositiveWeights=params["PositiveWeights"])

    iae.load_state_dict(model_state)

    if device=='cuda':
        iae = iae.cuda()

    return iae

############################################################
# Main code
############################################################

class IAE(torch.nn.Module):
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

    def __init__(self, fname='IAE_model', device='cpu', AnchorPoints=None, NSize=None,  reg_parameter=1000.,
                 simplex=False, noise_level=None,GaussNoise=True, PositiveWeights=False, rho_latcon=None,sparse_code=False,niter_sparse=10, warmup_iter=15000, verb=False):
        """
        Initialization
        """
        super(IAE, self).__init__()

        self.fname = fname
        NLayers = len(NSize[:,0])
        self.NLayers = NLayers
        self.NSize = NSize
        self.noise_level = noise_level
        #self.anchorpoints = Parameter(AnchorPoints, requires_grad=True)
        self.anchorpoints = AnchorPoints
        self.reg_parameter = reg_parameter
        self.verb = verb
        self.niter_sparse = niter_sparse
        self.sparse_code = sparse_code
        self.simplex = simplex
        self.num_ap = AnchorPoints.shape[0]
        self.Lin = AnchorPoints.shape[1]
        self.PhiE = None
        self.device=device
        self.warmup_iter = warmup_iter
        self.GaussNoise = GaussNoise
        self.PositiveWeights = PositiveWeights

        if rho_latcon is None:
            self.rho_latcon = np.ones((self.NLayers,))
        else:
            self.rho_latcon = rho_latcon

        dim = []
        dim.append(self.Lin)
        Lin = self.Lin

        for r in range(self.NLayers):

            if r ==0:
                Nch_in = AnchorPoints.shape[2]
            else:
                Nch_in = NSize[r-1,0]
            Nch_out = NSize[r,0]
            kern_size = NSize[r,1]
            stride = NSize[r,2]
            Lout = np.int(np.floor(1+1/stride*(Lin-kern_size)))
            dim.append(Lout)
            Lin = Lout

            encoder = []
            encoder.append(torch.nn.Conv1d(Nch_in, Nch_out,kern_size,stride=stride,bias=False))
            encoder.append(torch.nn.BatchNorm1d(Nch_out))
            encoder.append(torch.nn.ELU())  # This could be changed based

            setattr(self,'encoder'+str(r+1),torch.nn.Sequential(*encoder))

            # For the lateral connection

            if r >0 and r < self.NLayers:

                encoder = []
                encoder.append(torch.nn.Conv1d(Nch_in, Nch_out,kern_size,stride=stride,bias=False))
                encoder.append(torch.nn.BatchNorm1d(Nch_out))
                encoder.append(torch.nn.ELU())  # This could be changed based

                setattr(self,'encoder_lat'+str(r),torch.nn.Sequential(*encoder))

        self.dim = dim

        for r in range(1,self.NLayers+1):
            if r == NLayers:
                Nch_out = AnchorPoints.shape[2]
            else:
                Nch_out = NSize[self.NLayers-r-1,0]
            Nch_in = NSize[self.NLayers-r,0]
            kern_size = NSize[self.NLayers-r,1]
            stride = NSize[self.NLayers-r,2]

            decoder = []
            decoder.append(torch.nn.ConvTranspose1d(Nch_in, Nch_out,kern_size,bias=False))

            if stride == 1:
                decoder.append(torch.nn.ELU())  # This could be changed based

            setattr(self,'decoder'+str(r),torch.nn.Sequential(*decoder))

            # For the lateral connection
            if r < self.NLayers:

                decoder = []
                decoder.append(torch.nn.ConvTranspose1d(Nch_in, Nch_out,kern_size,bias=False))

                if stride == 1:
                    decoder.append(torch.nn.ELU())  # This could be changed based

                setattr(self,'decoder_lat'+str(r),torch.nn.Sequential(*decoder))

        self.thrd = 0
        if self.sparse_code:
            self.thrd = Parameter(torch.zeros(1,self.num_ap).to(self.device), requires_grad=True)

    ###
    #  DISPLAY
    ###

    def display(self,epoch,epoch_time,train_acc,rel_acc,pref="Learning stage - ",niter=None):

        if niter is None:
            niter = self.niter

        percent_time = epoch/(1e-12+niter)
        n_bar = 50
        bar = ' |'
        bar = bar + '█' * int(n_bar * percent_time)
        bar = bar + '-' * int(n_bar * (1-percent_time))
        bar = bar + ' |'
        bar = bar + np.str(int(100 * percent_time))+'%'
        m, s = divmod(np.int(epoch*epoch_time), 60)
        h, m = divmod(m, 60)
        time_run = ' [{:d}:{:02d}:{:02d}<'.format(h, m, s)
        m, s = divmod(np.int((niter-epoch)*epoch_time), 60)
        h, m = divmod(m, 60)
        time_run += '{:d}:{:02d}:{:02d}]'.format(h, m, s)

        sys.stdout.write('\033[2K\033[1G')
        if epoch_time > 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +np.str(niter)+ ' -- loss  = {0:e}'.format(np.float(train_acc)) + ' -- validation loss = {0:e}'.format(np.float(rel_acc))+bar+time_run+'-{0:0.4} '.format(epoch_time)+' s/epoch', end="\r")
        if epoch_time < 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +np.str(niter)+ ' -- loss  = {0:e}'.format(np.float(train_acc)) + ' -- validation loss = {0:e}'.format(np.float(rel_acc))+bar+time_run+'-{0:0.4}'.format(1./epoch_time)+' epoch/s', end="\r")

    ###
    #  ADDING CORRUPTION
    ###

    def corrupt(self, x): # Corrupting the data // THIS COULD BE CHANGED // should be tested

        if self.GaussNoise:
            noise = self.noise_level*torch.randn_like(x).to(self.device)
            result = x.clone()
            result.data = result.data + noise
        else:
            noise = torch.bernoulli(self.noise_level*torch.ones_like(x.data)).to(self.device)
            result = x.clone()
            result.data = result.data * noise # We could put additive noise

        return result

    ###
    #  ENCODE
    ###

    def encode(self, X):
        if self.noise_level is not None:
            tilde_x = self.corrupt(X)
        else:
            tilde_x = X.clone()

        PhiX_lat = []
        PhiE_lat = []

        PhiX = getattr(self,'encoder'+str(1))(torch.swapaxes(tilde_x,1,2))
        PhiE = getattr(self,'encoder'+str(1))(torch.swapaxes(self.anchorpoints.clone(),1,2))

        for r in range(1,self.NLayers):

            if r < self.NLayers:
                PhiX_lat.append(torch.swapaxes(getattr(self,'encoder_lat'+str(r))(PhiX),1,2))
                PhiE_lat.append(torch.swapaxes(getattr(self,'encoder_lat'+str(r))(PhiE),1,2))

            PhiX = getattr(self,'encoder'+str(r+1))(PhiX)
            PhiE = getattr(self,'encoder'+str(r+1))(PhiE)

        PhiX_lat.append(torch.swapaxes(PhiX,1,2))
        PhiE_lat.append(torch.swapaxes(PhiE,1,2))

        return PhiX_lat,PhiE_lat

    def decode(self, B):

        Xrec = torch.swapaxes(B[0],1,2)

        for r in range(self.NLayers-1):

            #up = torch.nn.Upsample(size=self.dim[self.NLayers-r-1], mode='linear', align_corners=True)
            #Xtemp = up(Xrec)
            #Btemp = up(torch.swapaxes(B[r+1],1,2))

            Xtemp = Xrec
            Btemp = torch.swapaxes(B[r+1],1,2)

            up = torch.nn.Upsample(size=self.dim[self.NLayers-r-1], mode='linear', align_corners=True)
            Xrec = up(getattr(self,'decoder'+str(r+1))(Xtemp) + self.rho_latcon[r]*getattr(self,'decoder_lat'+str(r+1))(Btemp))

        Xrec = getattr(self,'decoder'+str(self.NLayers))(Xrec)
        up = torch.nn.Upsample(size=self.dim[0], mode='linear', align_corners=True)
        Xrec = up(Xrec)

        return torch.swapaxes(Xrec,1,2)

    def interpolator(self, PhiX, PhiE):

        L = []
        B = []

        for r in range(self.NLayers):
            PhiE2 = torch.einsum('ijk,ljk -> il',PhiE[self.NLayers-r-1], PhiE[self.NLayers-r-1])
            iPhiE = torch.linalg.inv(PhiE2 + 1e-6*torch.linalg.norm(PhiE2)* torch.eye(self.anchorpoints.shape[0],device=self.device))
            Lambda = torch.einsum('ijk,ljk,lm', PhiX[self.NLayers-r-1], PhiE[self.NLayers-r-1], iPhiE)

            if self.PositiveWeights:
                mu = torch.max(Lambda,dim=1).values-1.0
                for i in range(Lambda.shape[1]) : # or maybe more
                    F = torch.sum(torch.maximum(Lambda,mu.reshape(-1,1)), dim=1) - Lambda.shape[1]*mu-1
                    mu = mu + 2/Lambda.shape[1]*F
                Lambda = torch.maximum(Lambda,mu.reshape(-1,1))-mu.reshape(-1,1)

            elif self.simplex:
                ones = torch.ones_like(Lambda,device=self.device)
                mu = (1 - torch.sum(Lambda,dim=1))/torch.sum(iPhiE)
                Lambda = Lambda +  torch.einsum('ij,i -> ij',torch.einsum('ij,jk -> ik', ones, iPhiE),mu)

            L.append(Lambda)
            B.append(torch.einsum('ik,kjl->ijl', Lambda, PhiE[self.NLayers-r-1]))

        return B, L

    def fast_interpolation(self, X, Amplitude=None,norm='1'):

        # Estimating the amplitude

        if norm == '1':
            Amplitude = torch.sum(torch.abs(X),(1,2)).squeeze()
        elif norm =='2':
            Amplitude = torch.sqrt(torch.sum(torch.square(X),(1,2)).squeeze())
        elif norm=='inf':
            Amplitude = 2*torch.max(torch.max(torch.abs(X),2)[0],1)[0].squeeze() # Not quite clean

        # Encode data
        PhiX,PhiE = self.encode(torch.einsum('ijk,i->ijk',X,1./Amplitude))

        # Define the barycenter
        B, Lambda = self.interpolator(PhiX,PhiE)

        # Decode the barycenter
        XRec = self.decode(B)

        Output = {"PhiX": PhiX, "PhiE": PhiE, "Barycenter": B, "Lambda": Lambda, "Amplitude": Amplitude, "XRec": XRec}

        return Output

    def get_barycenter(self, Lambda, Amplitude=None):

        _,PhiE = self.encode(self.anchorpoints)

        if Amplitude is None:
            print("To be done")
            Amplitude = torch.ones(Lambda[0].shape[0],).to(self.device)

        B = []
        for r in range(self.NLayers):
            B.append(torch.einsum('ik,kjl->ijl', Lambda[r], PhiE[self.NLayers-r-1]))

        # Decode the barycenter
        XRec = torch.einsum('ijk,i -> ijk',self.decode(B),Amplitude)

        return XRec

    def train(self, data_loader, optimizer, epochs,validation_loader=None):

        ste_time = time.time()
        cost_train = []
        cost_samp = []
        cost_rec = []
        cost_val = []

        for e in range(epochs):

            agg_cost = 0.
            agg_cost1 = 0.
            agg_cost2 = 0.
            num_batches = 0.

            for k,X in enumerate(data_loader):

                optimizer.zero_grad()

                Z,Ze = self.encode(X)
                B,Lambda = self.interpolator(Z,Ze)
                Xrec = self.decode(B)

                loss = torch.sum(torch.square(Xrec-X), 1)
                cost1 = torch.mean(loss)
                cost2 = 0

                for r in range(self.NLayers):
                    loss = torch.sum(torch.square(Z[self.NLayers-r-1]-B[r]), 1)
                    cost2 += self.reg_parameter*self.rho_latcon[r]*torch.mean(loss)
                cost = cost1 + cost2

                cost.backward(retain_graph=True) #backward autograd
                optimizer.step()

                agg_cost += cost
                agg_cost1 += cost1
                agg_cost2 += cost2
                num_batches += 1

            agg_cost /= num_batches
            agg_cost1 /= num_batches
            agg_cost2 /= num_batches
            cost_train.append(agg_cost.cpu().data)
            cost_samp.append(agg_cost2.cpu().data)
            cost_rec.append(agg_cost1.cpu().data)

            if validation_loader is not None:
                val_cost=0.
                for k,X in enumerate(validation_loader):

                    Z,Ze = self.encode(X)
                    B,Lambda = self.interpolator(Z,Ze)
                    Xrec = self.decode(B)

                    loss = torch.sum(torch.square(Xrec-X), 1)
                    cost1 = torch.mean(loss)
                    loss = torch.sum(torch.square(Z-B), 1)
                    cost2 = self.reg_parameter*torch.mean(loss)
                    cost = cost1 + cost2

                    val_cost += cost
                    num_batches += 1

                val_cost /= num_batches
                cost_val.append(val_cost.cpu().data)
            else:
                val_cost = agg_cost.clone()
                cost_val.append(0)

            if self.verb and (np.mod(e,100)==0):
                epoch_time = 0.01*(time.time()-ste_time) # Averaged over 100 it.
                ste_time = time.time()
                self.display(e,epoch_time,agg_cost.data,val_cost.data,niter=epochs) # To be updated

            # LR schedule

        self.PhiE = Ze

        return {"cost_train": cost_train, "cost_rec": cost_rec, "cost_samp": cost_samp,'cost_val':cost_val}

###############################################################################################################################################

class BSP(torch.nn.Module):
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

    def __init__(self,size=None,model=None,fname='IAE_model',Amplitude=None,Lambda=None,verb=None,device='cpu'):
        """
        Initialization
        """
        super(BSP, self).__init__()

        if model is None:
            model = load_model(fname,device=device)
        self.encode = model.encode
        self.decode = model.decode
        self.interpolator = model.interpolator
        self.NLayers = model.NLayers
        self.simplex = model.simplex
        self.Xrec = None
        self.verb = verb
        self.device=device
        PhiE,_=self.encode(model.anchorpoints)
        self.PhiE = PhiE


        if Lambda is None:
            print("To be done")
            for r in range(self.NLayers):
                setattr(self,'Lambda'+str(r+1),Parameter(Variable(torch.zeros(size,model.anchorpoints.shape[0]).to(self.device)), requires_grad=True))
        else:
            for r in range(self.NLayers):
                setattr(self,'Lambda'+str(r+1),Parameter(Variable(Lambda[r]), requires_grad=True))

        if Amplitude is None:
            self.Amplitude = Parameter(torch.ones(size,).to(self.device), requires_grad=False)
        else:
            self.Amplitude = Parameter(Amplitude, requires_grad=False)

    def display(self,epoch,epoch_time,train_acc,rel_acc,pref="Learning stage - ",niter=None):

        if niter is None:
            niter = self.niter

        percent_time = epoch/(1e-12+niter)
        n_bar = 50
        bar = ' |'
        bar = bar + '█' * int(n_bar * percent_time)
        bar = bar + '-' * int(n_bar * (1-percent_time))
        bar = bar + ' |'
        bar = bar + np.str(int(100 * percent_time))+'%'
        m, s = divmod(np.int(epoch*epoch_time), 60)
        h, m = divmod(m, 60)
        time_run = ' [{:d}:{:02d}:{:02d}<'.format(h, m, s)
        m, s = divmod(np.int((niter-epoch)*epoch_time), 60)
        h, m = divmod(m, 60)
        time_run += '{:d}:{:02d}:{:02d}]'.format(h, m, s)

        sys.stdout.write('\033[2K\033[1G')
        if epoch_time > 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +np.str(niter)+ ' -- loss  = {0:e}'.format(np.float(train_acc)) + ' -- loss rel. var. = {0:e}'.format(np.float(rel_acc))+bar+time_run+'-{0:0.4} '.format(epoch_time)+' s/epoch', end="\r")
        if epoch_time < 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +np.str(niter)+ ' -- loss  = {0:e}'.format(np.float(train_acc)) + ' -- loss rel. var. = {0:e}'.format(np.float(rel_acc))+bar+time_run+'-{0:0.4}'.format(1./epoch_time)+' epoch/s', end="\r")


    def solve(self,X,optimizer,epochs):

        ste_time = time.time()
        cost_train = []

        for e in range(epochs):

            optimizer.zero_grad()

            B = []
            for r in range(self.NLayers):
                if self.simplex:
                    L = torch.einsum('ij,i -> ij',getattr(self,'Lambda'+str(r+1)),1/(1e-6+torch.sum(abs(getattr(self,'Lambda'+str(r+1))),1)))
                else:
                    L = getattr(self,'Lambda'+str(r+1))
                B.append(torch.einsum('ik,kjl->ijl', L, self.PhiE[self.NLayers-r-1]))

            Xrec = torch.einsum('ijk,i -> ijk',self.decode(B),self.Amplitude)

            cost = torch.mean(torch.sum(torch.square(Xrec-X), (1,2)))

            cost.backward(retain_graph=True) #backward autograd
            optimizer.step()

            cost_train.append(cost.cpu().data)

            if self.verb and (np.mod(e,100)==0):
                epoch_time = 0.01*(time.time()-ste_time) # Averaged over 100 it.
                ste_time = time.time()
                self.display(e,epoch_time,cost.cpu().data,cost.cpu().data,niter=epochs) # To be updated

        self.Xrec = Xrec
        self.Barycenter = B

        return {"cost_train": cost_train}

###############################################################################################################################################

def learning_stage(XTrain,AnchorPoints,Xvalidation=None,logloss=False,Optimizer=2,batch_size=50,rho_latcon=None,rho_sparse=1,decay_epoch=2000,learning_rate=1e-3,reg_parameter=1e-3,epochs=2000,NSize=None,niter_sparse = 10,GaussNoise=True,loss_cut=1e-4,warmup_epochs=15000,sparse_code = False,simplex = False,fname='test',noise_level=None,display=False,restart=False,verb=True,tune=False,Weight=None,PositiveWeights=False):

    if torch.cuda.is_available():
        device = 'cuda'
        kwargs = {}
    else:
        device = 'cpu'
        kwargs = {}

    print("device USED: ",device)

    ###
    ###
    ###

    if NSize is None:
        NSize = [XTrain.shape[1],XTrain.shape[1],XTrain.shape[1]]
    if Weight is not None:
        print("With weighting")

    ###
    ###
    ###

    if device == 'cuda': # if GPU

        torch.backends.cudnn.benchmark = True

        # Transfer the data to the GPU

        XTrain = torch.as_tensor(XTrain.astype('float32')).cuda()
        AnchorPoints = torch.as_tensor(AnchorPoints.astype('float32')).cuda()

        # Initialize the model

        if restart:
            print("restarting with model ",fname)
            iae = torch.load(fname+'.pth')
            iae.anchorpoints = AnchorPoints
            iae.sparse_code = sparse_code # Only add sparsity
            fname += '_restart'
        else:
            iae = IAE(AnchorPoints=AnchorPoints, NSize=NSize,reg_parameter=reg_parameter,rho_latcon=rho_latcon, noise_level=noise_level,verb=verb,niter_sparse = niter_sparse,sparse_code = sparse_code,simplex = simplex,device=device,GaussNoise=GaussNoise,PositiveWeights=PositiveWeights)

        # Initialize the data loader

        data_loader = DataLoader(XTrain, batch_size=batch_size, shuffle=True, **kwargs)

        if Xvalidation is not None:
            Xvalidation = torch.as_tensor(Xvalidation.astype('float32'))
            Xvalidation = Xvalidation.to('cuda')
            validation_loader = DataLoader(Xvalidation, batch_size=batch_size, shuffle=True, **kwargs)
        else:
            validation_loader = None

    else: # if CPU

        if restart:
            iae = torch.load(fname+'.pth')
            iae.anchorpoints = AnchorPoints
            fname += '_restart'
        else:
            iae = IAE(AnchorPoints=torch.as_tensor(AnchorPoints.astype('float32')).cpu(), NSize=NSize,reg_parameter=reg_parameter,rho_latcon=rho_latcon, noise_level=noise_level,verb=verb,niter_sparse = niter_sparse,sparse_code = sparse_code,simplex = simplex,GaussNoise=GaussNoise,PositiveWeights=PositiveWeights)

        data_loader = DataLoader(torch.as_tensor(XTrain.astype('float32')).cpu(), batch_size=batch_size, shuffle=True)

        if Xvalidation is not None:
            validation_loader = DataLoader(torch.as_tensor(Xvalidation.astype('float32')).cpu(), batch_size=batch_size, shuffle=True, **kwargs)
        else:
            validation_loader = None

    iae = iae.to(device)

    # optimization stuff

    optimizer = _get_optimizer(Optimizer,iae.parameters(),learning_rate=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(data_loader), epochs=epochs)
    #####

    ste_time = time.time()
    cost_train = []
    cost_samp = []
    cost_rec = []
    cost_val = []
    UpdateOpt = True

    for e in range(epochs):

        agg_cost = 0.
        agg_cost1 = 0.
        agg_cost2 = 0.
        num_batches = 0.

        for k,X in enumerate(data_loader):

            start = time.time()

            optimizer.zero_grad()

            Z,Ze = iae.encode(X)
            B,Lambda = iae.interpolator(Z,Ze)
            Xrec = iae.decode(B)

            loss = torch.sum(torch.square(Xrec-X), 1)
            cost1 = torch.mean(loss)
            cost2 = 0

            for r in range(iae.NLayers):
                loss = torch.sum(torch.square(Z[iae.NLayers-r-1]-B[r]), 1)
                cost2 += iae.rho_latcon[r]*torch.mean(loss)

            if logloss:
                cost = (1+reg_parameter)*(torch.log(cost1) + reg_parameter*torch.log(cost2))
            else:
                cost = (1+reg_parameter)*(cost1 + reg_parameter*cost2)

            cost.backward(retain_graph=True) #backward autograd

            #torch.nn.utils.clip_grad_norm_(iae.parameters(), max_norm=10.0, norm_type=2)
            optimizer.step()

            agg_cost += cost
            agg_cost1 += cost1
            agg_cost2 += cost2
            num_batches += 1

        agg_cost /= num_batches
        agg_cost1 /= num_batches
        agg_cost2 /= num_batches
        cost_train.append(agg_cost.cpu().data)
        cost_samp.append(agg_cost2.cpu().data)
        cost_rec.append(agg_cost1.cpu().data)

        if validation_loader is not None:
            val_cost=0.
            num_batches = 0.
            with torch.no_grad():
                for k,X in enumerate(validation_loader):

                    Z,Ze = iae.encode(X)
                    B,Lambda = iae.interpolator(Z,Ze)
                    Xrec = iae.decode(B)

                    if Weight is None:
                        loss = torch.sum(torch.square(Xrec-X), 1)
                    else:
                        loss = torch.sum(torch.square((Xrec-X)*Weight), 1)
                    cost1 = torch.mean(loss)
                    cost2 = 0

                    for r in range(iae.NLayers):
                        loss = torch.sum(torch.square(Z[iae.NLayers-r-1]-B[r]), 1)
                        cost2 += iae.rho_latcon[r]*torch.mean(loss)
                    if logloss:
                        cost = (1+reg_parameter)*(torch.log(cost1) + reg_parameter*torch.log(cost2))
                    else:
                        cost = (1+reg_parameter)*(cost1 + reg_parameter*cost2)

                    val_cost += cost
                    num_batches += 1

            val_cost /= num_batches
            cost_val.append(val_cost.cpu().data)
        else:
            val_cost = agg_cost.clone()
            cost_val.append(0)

        if tune:
            tune.report(mean_accuracy=val_cost.cpu().data)

        #if i % 10 == 0:
        #    # This saves the model to the trial directory
        #    torch.save(iae.state_dict(), DIRmodel+"/model.pth")

        if iae.verb and (np.mod(e,100)==0):
            epoch_time = 0.01*(time.time()-ste_time) # Averaged over 100 it.
            ste_time = time.time()
            if sparse_code:
                iae.display(e,epoch_time,agg_cost.data,cost_sparse.data,niter=epochs) # To be updated
            else:
                iae.display(e,epoch_time,agg_cost.data,val_cost.data,niter=epochs) # To be updated

    #####

    torch.save(iae.to('cpu'), fname+'_fullmodel.pth')
    save_model(iae,fname=fname)

    result = {"cost_train": cost_train, "cost_rec": cost_rec, "cost_samp": cost_samp,'cost_val':cost_val}

    np.save(fname+'_training_results.npy',result,allow_pickle=True)

    if display:
        plt.figure(figsize=(15,10))
        plt.loglog(result['cost_train'],label='Training loss',lw=4)
        plt.loglog(result['cost_rec'],label='Reconstruction loss',lw=4)
        plt.loglog(result['cost_samp'],label='Latent loss',lw=4)
        if Xvalidation is not None:
            plt.loglog(result['cost_val'],label='Validation loss',lw=4)
        plt.xlabel("Epoch")
        plt.legend()

    return result

###############################################################################################################################################

def fast_interpolation(XTrain,Amplitude=None,fname='test',display=False,verb=True,norm='1'):

    if torch.cuda.is_available():
        device = 'cuda'
        #kwargs = {'num_workers': 1, 'pin_memory': True}
        kwargs = {}
    else:
        device = 'cpu'
        kwargs = {}

    print("device USED: ",device)

    #model = torch.load(fname+".pth").to(device)
    model = load_model(fname,device=device)

    XTrain = torch.as_tensor(XTrain.astype('float32')).to(device)

    res = model.fast_interpolation(XTrain,norm = norm) # Here we should update the amplitude
    alpha = torch.sum(res["XRec"]*XTrain,1)/torch.sum(res["XRec"]*res["XRec"],1)
    res["XRec"] = torch.einsum('ijk,i->ijk',res["XRec"],alpha.squeeze())

    if norm == '1':
        Amplitude = torch.sum(torch.abs(res["XRec"]),1).squeeze()
    elif norm =='2':
        Amplitude = torch.sqrt(torch.sum(torch.square(res["XRec"] ),1).squeeze())
    elif norm=='inf':
        Amplitude = 2*torch.max(torch.abs(res["XRec"]),1)[0].squeeze()

    Lambda = [r.cpu() for r in res["Lambda"]]

    return {"PhiX": [r.cpu().data for r in res["PhiX"]], "PhiE": [r.cpu().data for r in res["PhiE"]], "Barycenter": [r.cpu().data for r in res["Barycenter"]], "Lambda": [r.cpu().data for r in res["Lambda"]], "XRec": res["XRec"].cpu().data}

    #torch.einsum('ijk,i->ijk',res["XRec"],Amplitude)


###############################################################################################################################################

def bsp(XTrain,Lambda=None,Amplitude=None,learning_rate=1e-3,epochs=100,fname='test',display=False,verb=True,norm='1',Optimizer=0,Irange=None):

    if torch.cuda.is_available():
        device = 'cuda'
        #kwargs = {'num_workers': 1, 'pin_memory': True}
        kwargs = {}
    else:
        device = 'cpu'
        kwargs = {}

    X = torch.as_tensor(XTrain.astype('float32')).to(device)

    # LOAD THE MODEL
    model = load_model(fname,device=device)

    # PERFORM A FAST INTERPOLATION
    Output = model.fast_interpolation(X)

    # PERFORM THE BSP
    iae = BSP(size=X.shape[0],fname=fname,verb=verb,Lambda=Output["Lambda"],Amplitude=Output["Amplitude"],device=device)
    iae = iae.to(device)

    optimizer = _get_optimizer(Optimizer,iae.parameters(),learning_rate=learning_rate)
    result=iae.solve(X, optimizer, epochs)

    if display:
        plt.figure(figsize=(15,10))
        plt.loglog(result['cost_train'],label='Training loss',lw=4)
        plt.xlabel("Epoch")
        plt.legend()

    return {"Barycenter": [r.cpu().data for r in iae.Barycenter], "Lambda": [getattr(iae,'Lambda'+str(r+1)).cpu().data for r in range(iae.NLayers)], "XRec": iae.Xrec.cpu().data,"Amplitude": iae.Amplitude.cpu().data}

###############################################################################################################################################
