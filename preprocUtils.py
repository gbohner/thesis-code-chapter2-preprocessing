import torch
import numpy as np
import math
from collections import OrderedDict
import warnings
import preprocRandomVariables

import copy


def apply(func, M, dim=0):
    """
    Applies func to dim's dim of M tensor and collects results
    """
    tList = [func(m) for m in torch.unbind(M, dim=dim) ]
    res = torch.stack(tList, dim=dim)

    return res 


def toTorchParam(inp, device=None, to_log=False, ndims=3, paramShape=None):
    """
    Converts an object (number, array, list) to torch.nn.Parameter() class
    If paramShape is given, it will put the inp into that shape, else it will create
    an N x 1 x 1 x ... 1 such that the total dimension is ndims
    
    if to_log is given, it will also take the logarithm of the input
    """
    
    if device is None:
        try:
            device=inp.device
        except:
            device='cpu'
    
    if paramShape is None:
        paramShape = [-1] + [1]*(ndims-1)
    
    if to_log:
        return torch.nn.Parameter(
            torch.tensor(inp, device=device).float().view(*paramShape).log()
            )
    else:
        return torch.nn.Parameter(
            torch.tensor(inp, device=device).float().view(*paramShape)
            )
    

def getDataStatistics(train_x, train_y=None, expected_latent_photon_count_guess = 2.0):
    """
    Given training data, and expected true photon counts before gain modulation,
    return some data statistics that may be used to initialise
    usual parameters such as
        - lengthscales
        - output scales
        - gains
        - expected photon counts
    """
    input_device = train_x.device
    
    out = OrderedDict()
    
    out['expected_latent_photon_count'] = expected_latent_photon_count_guess
    
    x_minmax = torch.stack([train_x.min(0)[0], train_x.max(0)[0]])
    
    out['x_minmax'] = x_minmax
    out['x_span'] = x_minmax.unbind(1) # list of dimension-wise bounds tensors of train_x
    out['x_width'] = x_minmax[1,:] - x_minmax[0,:]
    out['x_std'] = train_x.std(0)
    out['n_data'] = torch.tensor(train_x.size(0), device=input_device).float()
    out['n_data_per_dim'] = out['n_data'].pow(1./train_x.size(1))
    
    # For lengthscales combine x_width with x_std or n_data_per_dim
    out['lengthscale_guess'] = out['x_width']/out['n_data_per_dim']*2.0
    
    
    if train_y is not None:
        # Estimate of photomultiplier linear gain
        # Do affine linear regression from train_y.mean(1) to train_y.var(1)
        predictor = nanmean(train_y,1)
        target = nanvar(train_y,1)
        
        # Get rid of pixels with only NaN observations
        predictor = predictor[torch.isnan(predictor) == False].unsqueeze(1)
        target = target[torch.isnan(target) == False].unsqueeze(1)
        
#         predictor = apply(
#             lambda x: x[torch.isnan(x)==0].mean(), train_y, dim=0).unsqueeze(-1)
#         target = apply(
#             lambda x: x[torch.isnan(x)==0].var(), train_y, dim=0).unsqueeze(-1)
        out['y_gain_linear'], tmp, out['y_gain_offset'] = (
            torchLinReg(predictor, target, exact=True))

        # Make a guess about the expected latent output scale by taking the spatial std after gain-correction
        out['latent_output_scale'] = predictor.squeeze().div(out['y_gain_linear']).std()

        ##############################################################################################
        # Guess the location and scale of the pedestal
        tmp = train_y.contiguous().view(-1).cpu()
        tmp = tmp[torch.isnan(tmp)==0]
        hist, bins = np.histogram(tmp[tmp<np.quantile(tmp, 0.5)], bins=200)

        hist = torch.tensor(hist).to(input_device)
        bins = torch.tensor(bins).to(input_device)

        bin_size = bins[1] - bins[0]
        filter_size = torch.tensor(int(17)).to(input_device) #(5./bin_size).round().int()

        filter_weights = preprocRandomVariables.Normal(loc=filter_size.float()/2., scale=filter_size.float()/4.
                                                   ).log_prob(torch.arange(filter_size).float().to(input_device)
                                                             ).exp().to(input_device)
        filter_weights = filter_weights.div(filter_weights.sum()).to(input_device)

        # Gaussian smoothing of histograms
        hist_smooth = torch.nn.functional.conv1d(
            hist.view(1,1,-1).float(), 
            filter_weights.view(1,1,-1), 
            padding=int((filter_size-1)/2))

        # Correction for edge effect
        corr_smooth = torch.nn.functional.conv1d(
            torch.ones_like(hist.view(1,1,-1).float()), 
            filter_weights.view(1,1,-1), 
            padding=int((filter_size-1)/2))

        hist_smooth = hist_smooth.div(corr_smooth).squeeze()

        # Find the peak and the width (to the right) of the peak
        out['y_pedestal_loc'] = bins[hist_smooth.argmax()].to(input_device)
        out['y_pedestal_scale'] = (bins[
            (hist_smooth[hist_smooth.argmax():]-(hist_smooth.max()-hist_smooth.min())/2).abs().argmin() 
            + hist_smooth.argmax()]
                 - out['y_pedestal_loc']).to(input_device)
        
        out['y_pedestal_scale'] = out['y_pedestal_scale']/math.sqrt(2.*math.log(2.)) # Correct from half width at half maximum to scale
    
    return out

def updateModelParams(model, update_dict):
    """
    Updates the model's state dictionary with parameters in the update_dict
    """
    
    cur_state_dict = model.state_dict()
    
    for name, new_param in update_dict.items():
        if name in cur_state_dict.keys():
            assert(new_param.shape==cur_state_dict[name].shape)
            cur_state_dict[name] = new_param
        else:
            warnings.warn('%s is not in the state dictionary of %s' % (name, model.__class__) )
    
    model.load_state_dict(cur_state_dict)
    
def initialiseModelParams(model, dataStats):
    update_dict = {}
    state_dict = model.state_dict()
    for name, param in model.named_parameters():
        if "log_lengthscale" in name:
            # Original lengthscale guess is always O(1).
            # We multiply the original guessed value by the value estiamted given the dataset, by
            # adding the log values together
            update_dict[name] = toTorchParam(param+dataStats['lengthscale_guess'].log().mean(), ndims=param.ndimension())
            
            # Also update the prior parameters (in case the exist, otherwise it returns a warning):
            update_dict[name + '_prior.a'] = state_dict[name + '_prior.a'] + torch.ones_like(state_dict[name + '_prior.a'])*(dataStats['lengthscale_guess'].log().mean()).squeeze()
            update_dict[name + '_prior.b'] = state_dict[name + '_prior.b'] + torch.ones_like(state_dict[name + '_prior.b'])*(dataStats['lengthscale_guess'].log().mean()).squeeze()
            
        
    for name, param in state_dict.items():
        # Center was expected to be given in the [0,1]*data_dims range
        # Modify it to lie at the same point but within x_span:    
        if "center_x_mins" in name:        
            update_dict[name] = torch.ones_like(state_dict[name])*dataStats['x_minmax'][0,:]
            
        if "center_x_widths" in name:
            update_dict[name] = torch.ones_like(state_dict[name])*dataStats['x_width'].squeeze()
            
        #if "log_outputscale" in name: # Deprecated due to ScaleKernel()
        #    update_dict[name] = toTorchParam(param+dataStats['latent_output_scale'].log(), ndims=param.ndimension())
    
    updateModelParams(model, update_dict)

    
def torchLinReg(X, Y, maxiter=1000, loss_rel_change=1e-6, exact=False, ret_model = False):
    """
    Linear regression from X (N x dX) to Y (N x dY)
    Returns weights, y-intercept and x-intercept by default, or trained model if ret_model = True
    """
    
    if exact: # Solve via inverse
        X_bias = torch.cat([X, torch.ones((X.shape[0],1)).to(X.device)], dim=1)
        sol = (X_bias.t().matmul(X_bias)).inverse().matmul(X_bias.t().matmul(Y))
        out = [sol[0:-1], sol[-1], -sol[-1]/sol[0:-1]]
        
    else: # Solve via optimiser
        model = torch.nn.Linear(X.size(1), Y.size(1), bias=True).to(X.device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
        prev_loss = float('inf')
        for t in range(maxiter):
            optimizer.zero_grad()
            loss = loss_fn(model(X), Y)
            if ((prev_loss-loss)/loss) > loss_rel_change:
                prev_loss = loss
            else:
                #print(t)
                break
            loss.backward()
            optimizer.step(lambda : loss)
        
        out = [model.weight.data, model.bias.data, -model.bias.data/model.weight.data]
        
        if ret_model:
            out = model
    
    return out
            
    
        
    

def swap_indices(x, dim=1, index=torch.tensor([1,0])):
    """ with default settings it swaps columns 0 and 1 in x to essentially go from C order to F order """
    return x.index_copy(dim=dim, index=index.to(x.device), tensor=x)
    
import itertools
def create_test_grid(n_test_grid=32, ndims=2, device='cpu', a=0.0, b=1.0):
    if torch.tensor(a).numel()==1 and torch.tensor(n_test_grid).numel()==1:
        gridLinSpace = torch.linspace(a,b, int(n_test_grid))
        all_gridLinSpaces = [gridLinSpace]*ndims
    else:
        all_gridLinSpaces = [torch.linspace(a[i],b[i], int(n_test_grid[i])) for i in range(n_test_grid.numel())]
        
    test_x = torch.tensor(list(itertools.product(*all_gridLinSpaces))).to(device)
    return test_x
    
    
    
import subprocess
import torch.cuda
def get_gpu_memory_stats(min_mem_needed = 4000.):
    """Get the current gpu usage (in MBs) and select the best one or returns None
    
    Returns
    -------
    best_cuda_device : string
    max_mem_avail : number
    gpu_max_mem : dict ( gpu_index : mem )
    gpu_avail_mem : dict ( gpu_index : mem )
    """
    
    gpu_max_mem = {}
    gpu_avail_mem = {}
    best_cuda_device = None
    max_mem_avail = 0.
    best_index = -1
    
    if torch.cuda.is_available():
    
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ])
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_used_mem = dict(zip(range(len(gpu_memory)), gpu_memory))
    

        # Calculate available memory
        for i in range(torch.cuda.device_count()):
            gpu_max_mem[i] = torch.cuda.get_device_properties(i).total_memory/(2**20) # returns in bytes, turn into MB
            gpu_avail_mem[i] = gpu_max_mem[i] - gpu_used_mem[i]
            if gpu_avail_mem[i] > max_mem_avail:
                best_index = i 
                max_mem_avail = gpu_avail_mem[i]

    if max_mem_avail > min_mem_needed:
        best_cuda_device = 'cuda:%d' % best_index
    
    return best_cuda_device, max_mem_avail, gpu_max_mem, gpu_avail_mem


def nansum(A, dim=-1):
    input_device = A.device
    N = (torch.isnan(A)==0.).sum(dim=dim).float()
    A = copy.deepcopy(A).to(input_device)
    A[torch.isnan(A)] = 0
    Asum = A.sum(dim)
    Asum[N==0] = float('nan')
    
    return Asum

def nanmean(A, dim=-1):
    input_device = A.device
    A = copy.deepcopy(A).to(input_device)
    N = (torch.isnan(A)==0.).sum(dim=dim).float()
    A[torch.isnan(A)] = 0
    return A.sum(dim=dim).div(N)

def nanvar(A, dim=-1, bessel_correction = True):
    Amean = nanmean(A, dim)
    Ares2 = nansum((A-Amean.unsqueeze(dim)).pow(2), dim=dim)   
    N = (torch.isnan(A)==0.).sum(dim=dim).float()
    
    if bessel_correction:
        N = N-1.
    
    return Ares2.div(N)

def nanstd(A, dim=-1, bessel_correction = True):
    return nanvar(A, dim=dim, bessel_correction=bessel_correction).sqrt()

def nanmax(A, dim=None):
    input_device = A.device
    A = copy.deepcopy(A).to(input_device)
    A[torch.isnan(A)] = -float('inf')
    if dim is None:
        return A.max()
    else:
        return A.max(dim=dim)
    
def nanmin(A, dim=None):
    input_device = A.device
    A = copy.deepcopy(A).to(input_device)
    A[torch.isnan(A)] = float('inf')
    if dim is None:
        return A.min()
    else:
        return A.min(dim=dim)
    
