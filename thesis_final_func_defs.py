# Load a trained model
import torch
import math
import gpytorch
import numpy as np
import copy

from torch.utils.data import TensorDataset, DataLoader


import preprocUtils
import preprocRandomVariables
import preprocLikelihoods
import preprocModels
import preprocKernels

from collections import OrderedDict
import itertools

from warnings import warn

# File system management
import os
import errno
import zipfile
from glob import glob


# --------------------------------------------------------------
# Loading imputed and Nan datasets
# --------------------------------------------------------------


def loadImputedData(
    dataset_name = 'neurofinder.00.00',
    data_dir='/nfs/data/gergo/Neurofinder_update/',
    device = 'cpu',
    # We can supply a model that corrects for the photomultipler gain
    mll = None
):

    savedImgsImputed = sorted(glob(data_dir+dataset_name+'/preproc2P/imgsImputed*.npy'))
    imgsImputed = torch.tensor(np.load(savedImgsImputed[-1])).to(device)

    # Correct for pmGain, if mll has it
    if mll:
        if hasattr(mll, 'pmGain_y'):
            imgsImputed.div_(mll.pmGain_y.reshape(*imgsImputed.shape[:2]).unsqueeze(-1))
            
    return imgsImputed


def loadNanData(
    dataset_name = 'neurofinder.00.00',
    data_dir='/nfs/data/gergo/Neurofinder_update/',
    device = 'cpu',
    # We can supply a model that corrects for the photomultipler gain
    mll = None
):

    savedImgsNan = sorted(glob(data_dir+dataset_name+'/preproc2P/imgsNan*.npy'))
    imgsNan = torch.tensor(np.load(savedImgsNan[-1])).to(device)

    # Correct for pmGain, if mll has it
    if mll:
        if hasattr(mll, 'pmGain_y'):
            imgsNan.div_(mll.pmGain_y.reshape(*imgsNan.shape[:2]).unsqueeze(-1))
            
    return imgsNan

# --------------------------------------------------------------
# Loading training data 
# --------------------------------------------------------------

def loadTrainingData(dataset_name = 'neurofinder.00.00',
                     stamp = '',
                     data_dir='/nfs/data/gergo/Neurofinder_update/', device = 'cpu'):
        
    savedTrainingData = sorted(glob(data_dir+dataset_name+'/preproc2P/trainingData' + stamp + '.npz'))
        
    if savedTrainingData:
        fHandle = np.load(savedTrainingData[-1])
        trainingData = dict(fHandle)
        fHandle.close()
        for name, arr in trainingData.items():
            trainingData[name] = torch.tensor(arr).to(device)
    
    
    return trainingData

def getTrainingCoverageMap(trainingData, filter_width = 35):
    # Get percentage of pixels belonging to training data in a each given window size (filter_width)
    device = trainingData['mean_im'].device
    
    mean_im_norm = trainingData['mean_im']/trainingData['mean_im'].max()
    training_loc = mean_im_norm.new_zeros(mean_im_norm.size())
    training_loc[trainingData['train_x'].long().unbind(1)] = 1
    
    cur_filter = torch.ones(1,1,filter_width,filter_width).to(device).div(filter_width**2)
    
    trainingCoverageMap = torch.nn.functional.conv2d(
        training_loc.view(1,1,*training_loc.shape), 
        cur_filter, 
        padding=tuple((torch.tensor(cur_filter.shape)[2:]-1)/2))[0][0]
    
    return trainingCoverageMap

def downsampleTrainingData(trainingData, filter_width = 35, targetCoverage = 0.05):
    # Downsample (with rejection) overpopulated areas in training data to ensure spatial uniformity
    device = trainingData['mean_im'].device
    
    trainingCoverageMap = getTrainingCoverageMap(trainingData, filter_width = filter_width)

    # Get fraction covered in training locations
    fracCoveredAtSample = trainingCoverageMap[trainingData['train_x'].long().unbind(1)]

    # Drop samples with probability of exceeding target coverage level
    fracDropProbability = 1 - targetCoverage / fracCoveredAtSample
    keepSample = torch.rand(fracDropProbability.shape[0]).to(device) >= fracDropProbability

    trainingDataUniform = copy.deepcopy(trainingData)
    trainingDataUniform['train_x'] = trainingDataUniform['train_x'][keepSample,:]
    trainingDataUniform['train_y'] = trainingDataUniform['train_y'][keepSample]
    
    return trainingDataUniform



# --------------------------------------------------------------
# Loading fitted models
# --------------------------------------------------------------

def loadFittedModel(
    dataset_name = 'neurofinder.00.00',
    data_dir='/nfs/data/gergo/Neurofinder_update/',
    prior='noPrior', 
    lik='linLik', 
    stamp = '',
    device = 'cpu'
):

    mll = torch.load(data_dir + dataset_name +'/preproc2P/savedModels/mll_' +prior+'_' + lik + stamp, map_location=device)

    model = mll.model
    likelihood = mll.likelihood
    mean_im = mll.mean_im

    train_x = mll.train_x
    train_y = mll.train_y

    dataStats = preprocUtils.getDataStatistics(train_x, train_y)
    
    #print(dataStats)
    #print(OrderedDict(likelihood.named_parameters()))
    
    
    # Set the model and likelihood in evaluation mode
    model.eval()
    likelihood.eval()
    
    # Create test grids over which we predict for easy visualisations
    n_test_grid = torch.tensor(mean_im.shape)
    n_test_grid_small = 32
    test_x = preprocUtils.create_test_grid(n_test_grid, ndims=2, device=device, a=dataStats['x_minmax'][0,:], b=dataStats['x_minmax'][1,:])
    test_x_small = preprocUtils.create_test_grid(n_test_grid_small, ndims=2, device=device, a=dataStats['x_span'][0][0], b=dataStats['x_span'][0][1])
    
    # Get log_photon counts:
    pred_log_photon = model(test_x)
    if isinstance(model.mean_module, gpytorch.means.ConstantMean):
        pred_gain_func = (pred_log_photon.mean()-model.mean_module.constant.data).exp()
    else:
        pred_gain_func = pred_log_photon.mean().exp()
    
    divBy = pred_gain_func.reshape(*mean_im.shape)
    gainRange = [0.1, 10.]

    corr_mean_im = (mean_im).div(torch.clamp(divBy, min=1./gainRange[1], max=1./gainRange[0]))
    #corr_mean_im = (mean_im-likelihood.offset).div(torch.clamp(divBy, min=1./gainRange[1], max=1./gainRange[0]))+likelihood.offset
    
    return mll, model, likelihood, train_x, train_y, dataStats, mean_im, pred_gain_func.reshape(*n_test_grid), corr_mean_im


# --------------------------------------------------------------
# Estimating the gray level -> latent signal transition
# --------------------------------------------------------------

def getLambdaLogProb(log_w, log_lam, dim=-1, requires_grad=True):
    if dim == -1:
        dim=log_w.ndimension()-1
    # Get the log probabilities (numerically stable), then logsumexp()
    x = torch.arange(log_w.size(dim), device=log_w.device).float().view(*([1]*max(dim,0))+[-1]+[1]*max(log_w.ndimension()-dim-1,0))

    lprobs = (log_w
              - (x+1.).lgamma()
              + x*log_lam
              -log_lam.exp())

    del x
    torch.cuda.empty_cache()

    if requires_grad:
        return lprobs.logsumexp(dim=dim)
    else:
        return lprobs.logsumexp(dim=dim).data
    
# Define an optimiser that starts from MAP estimate and uses lambda log prob as objective
class LambdaOptimiser(torch.nn.Module):
    def __init__(self, log_w, lambda_guess=torch.tensor(1.)):
        super(LambdaOptimiser, self).__init__()
        self.register_buffer("log_w", log_w)
        
        # Put log lambda into the appropriate shape
        log_lam = lambda_guess.float().clamp(min=1e-1).log()
        log_lam = log_lam.view(list(log_lam.size())+[1]*max(0, log_w.ndimension()-log_lam.ndimension()))
        self.register_parameter("log_lam", preprocUtils.toTorchParam(log_lam, 
                                                                     paramShape=log_lam.size(), 
                                                                     device=log_lam.device))
        
    def forward(self):
        return -getLambdaLogProb(self.log_w, self.log_lam).sum()
        

def getOptLambda(log_w, lambda_guess=None): # Opt lambda given discrete estimate of photon distribution
#     if log_w[0]==0: # Special case of certainty
#         return torch.tensor([0.])
    lambda_guess = lambda_guess if lambda_guess is not None else torch.tensor(1., device=log_w.device)
    lamModule = LambdaOptimiser(log_w.detach(), lambda_guess=lambda_guess)
    optim = torch.optim.LBFGS(lamModule.parameters()) 
    # LBFGS does optimisation internally via "closure", no need to iterate outside
    def closure():
        optim.zero_grad()
        loss = lamModule()
        #print(loss, lamModule.log_lam.exp().data)
        loss.backward()
        return loss  

    optim.step(closure)
    
    out = lamModule.log_lam.exp().data
    
    del lamModule
    
    return out
    

def im2logPhotonProb(im, photon_log_probs, gray_levels, interpolate=False):
    if not interpolate:
        # Nearest neighbor version
        return photon_log_probs[
            (im.unsqueeze(-1) - gray_levels.view(*([1]*im.ndimension()+[-1]))).abs().min(-1)[1],:]
    else:

        # Interpolation version
        dists = (im.unsqueeze(-1) - gray_levels.view(*([1]*im.ndimension()+[-1]))) # Distances along last dimension

        # Find the last element in gray_levels that im is larger than (so dists>=0), then interpolate or extrapolate appropriately
        last_pos_ind = (((dists>=0).sum(-1))-1).clamp(0, dists.size(-1)-2).unsqueeze(-1)
        interp_inds = torch.cat([last_pos_ind, last_pos_ind+1], dim=-1)

        static_indices = np.indices(interp_inds.shape)
        static_indices[-1] = interp_inds
        dists_sorted = dists[static_indices]


        # Get the two closest distances, and values to linearly inter-/extra-polate
        dist0 = dists_sorted[...,0]
        dist1 = dists_sorted[...,1]

        val0 = photon_log_probs[interp_inds[...,0],:]
        val1 = photon_log_probs[interp_inds[...,1],:]

        dist_sign_same = ((dist0*dist1)>=0.).float()

        denom = (
            # If different sign (interpolate)
            (1.-dist_sign_same)*(dist1.abs()+dist0.abs()) 
            # If same sign (extrapolate)
            + dist_sign_same *((dist0-dist1).abs())
        )


        rx = dist0.unsqueeze(-1)

        m = (val1-val0)/denom.unsqueeze(-1)


        fx = m*rx + val0


        return fx



def im2photon(im, inverse_poiss_MAP, gray_levels, keep_zeros=True):
    # Linear inter/extra-polation version
    
    dists = (im.unsqueeze(-1) - gray_levels.view(*([1]*im.ndimension()+[-1]))) # Distances along last dimension

    # Find the last element in gray_levels that im is larger than (so dists>=0), then interpolate or extrapolate appropriately
    last_pos_ind = (((dists>=0).sum(-1))-1).clamp(0, dists.size(-1)-2).unsqueeze(-1)
    interp_inds = torch.cat([last_pos_ind, last_pos_ind+1], dim=-1)

    static_indices = np.indices(interp_inds.shape)
    static_indices[-1] = interp_inds
    dists_sorted = dists[static_indices]


    # Get the two closest distances, and values to linearly inter-/extra-polate
    dist0 = dists_sorted[...,0]
    dist1 = dists_sorted[...,1]

    val0 = inverse_poiss_MAP[interp_inds[...,0]]
    val1 = inverse_poiss_MAP[interp_inds[...,1]]

    dist_sign_same = ((dist0*dist1)>=0.).float()

    denom = (
        # If different sign (interpolate)
        (1.-dist_sign_same)*(dist1.abs()+dist0.abs()) 
        # If same sign (extrapolate)
        + dist_sign_same *((dist0-dist1).abs())
    )


    rx = dist0

    m = (val1-val0)/denom


    fx = m*rx + val0

    if keep_zeros:
        fx = fx * (im!=0).type(fx.type())
    
    return fx



def getInverseMapEstimate(
    likelihood,
    max_gray_level = torch.tensor(float(5000)), # Should be set to imgsImputed.max() so it changes based on dataset
    max_photon = float(70),
    max_light_level = 1.,
    light_level_skip = 1e-2
):
    # Calculate the mean photon probability for each grey level to use as image conversion
    device = likelihood.offset.data.device
    
     # Get the gray levels present in the data
    gray_levels = torch.cat([torch.tensor([-0.02, 0., 1e-10]), # Important to add non-zero values close to zero
                     torch.logspace(-3, max_gray_level.log10(),100)]).to(device) # Calculate piecewise linear approx in log space to avoid wasteful computations
    
    #try:
    if type(likelihood) == preprocLikelihoods.PoissonInputPhotomultiplierLikelihood or type(likelihood) == preprocLikelihoods.PoissonInputUnderamplifiedPhotomultiplierLikelihood:
        # Assume we are using a photon-count based likelihood model

        # Get "fake model predictions" (representing light level) to use as input in getting photon log probabilities
        light_levels = torch.arange(1e-4, max_light_level,light_level_skip).to(device) # In photon
        model_out = gpytorch.random_variables.GaussianRandomVariable(light_levels.log(), gpytorch.lazy.DiagLazyVariable(1e-7*torch.ones_like(light_levels)))

        # Get the photon log probability distribution (over 0 - max_photon photons) for each gray levels
        photon_counts, photon_log_probs = (
            likelihood.getPhotonLogProbs(model_out.mean().view(-1,1).exp(), max_photon=max_photon, reNormalise = False))
        p_PM = likelihood.createResponseDistributions(photon_counts)

        photon_log_probs = likelihood.getLogProbSumOverTargetSamples(p_PM, gray_levels.view(-1))

        # Normalise the unnormalised distribution over 0-max_photons (it would be normalised if max_photon=Inf)
        log_photon_prob_marginals = photon_log_probs - photon_log_probs.logsumexp(1).unsqueeze(1)

        # Given the photon log probability marginals, we can find the 
        # Maximum a posteriory estimate of the mean photon count for that gray level.
        inverse_poiss_MAP = torch.cat([getOptLambda(log_photon_prob_marginals[i,:], 
                                                lambda_guess=log_photon_prob_marginals[i,:].max(0)[1]) 
                                   for i in range(log_photon_prob_marginals.size(0))])
    else:
        # If we're using a non-photon based likelihood (ie Linear-Gaussian likelihood)
        # the above will fail, so in this branch we compute the observation->latent conversion 
        # based on the 'LinearGainLikelihood'
        # 
        warn('getInverseMapEstimate() is using threshold-linear mapping')
        inverse_poiss_MAP = (
                (gray_levels-likelihood.offset.data).clamp(0,float('inf')).log()
                -likelihood.log_gain.data
            ).exp()
        
        
    
    # Return the gray level -> MAP estimate of mean of generating Poisson distribution conversion
    return gray_levels, inverse_poiss_MAP
    

    
# Median filtering to reduce photon shot noise
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

   


    
    
# --------------------------------------------------------------
# Utility
# --------------------------------------------------------------
    
def progress_bar(func, inp, index, report=True, report_freq=10):
    if report:
        if index % report_freq == 0:
            print(index)
    return func(inp)


#import os, errno
def mkdirs(newdir, mode=0o777):
    try: os.makedirs(newdir, mode)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory 
        if err.errno != errno.EEXIST or not os.path.isdir(newdir): 
            raise
            
            
            
            
# --------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------
               
def logistic(x,  x0=0., k=1., L=1.):
    return x.add(-x0).mul(-k).exp().add(1).reciprocal().mul(L)            
            
            
def fast_digitise(A, r0, r1, nbins=None, binsize=None):
    """Inspired by https://stackoverflow.com/questions/26783719/efficiently-get-indices-of-histogram-bins-in-python
    
    Treat <r0+jitter and r1<= as two seperate bins (so add 1., then clamp at 0)
    
    # This approach works because pytorch seem to remember sign very well even with little jitter 
    (so if A==r0, A-(r0+jitter) is negative), but (r1-(r0+jitter) / (r1-r0)) = 1, which is weird but useful
    """
    
    if binsize is None:
        binsize=1.
    
    if nbins is None:
        nbins = int(torch.tensor((r1-r0)/binsize).floor().add(2.))
    
    jitter = 1e-12
    bin_center_correction = (r1-r0)/(2.*float(nbins-2.)) # So that r0 and r1 are bin centers rather than edges
#     r0 -= bin_edge_correction
#     r1 += bin_edge_correction
    
    
    out_hists = ((A-(r0+jitter)) * (float(nbins-2)/(r1-r0))).floor().long().add(1.).clamp(0, nbins-1)
    bin_centers = torch.cat([torch.tensor(r0).view(-1), 
                                torch.linspace(r0+bin_center_correction, r1-bin_center_correction, nbins-2) , 
                                torch.tensor(r1).view(-1)]).to(A.device)

    return out_hists, bin_centers


def fast_histograms(A, r0, r1, nbins, dim = -1, output_device = None):
    dim = dim if dim>0 else (A.ndimension()+dim)
    output_device = output_device if output_device is not None else A.device

    Adigitised, bin_centers = fast_digitise(A, r0, r1, nbins)
    Adims = list(A.size())
    Adims[dim]=int(bin_centers.numel())
    Ahists = torch.zeros(*Adims, device=A.device)
    
    if torch.isnan(A.view(-1)).sum()>0: # if nans are present, deal with them
        nans_present = True
        Adigitised = Adigitised.float()
        Adigitised[torch.isnan(A)] = float('nan')
    else:
        nans_present = False
    
    
    if nans_present: # Do nansum in float comparisons
        for i in range(Ahists.size(dim)):
            ind_range = [slice(None)]*max(dim,0)+[i]+[slice(None)]*max(A.ndimension()-dim-1,0)
            Ahists[ind_range].add_(preprocUtils.nansum((Adigitised==float(i)).float(),dim=dim).float())
    else: # Faster to do in the long() format if no nans present
        for i in range(Ahists.size(dim)):
            ind_range = [slice(None)]*max(dim,0)+[i]+[slice(None)]*max(A.ndimension()-dim-1,0)
            Ahists[ind_range].add_((Adigitised==i).sum(dim).float())
        
    return Ahists.to(output_device), bin_centers.to(output_device)