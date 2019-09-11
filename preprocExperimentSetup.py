import torch
import math
import json
import numpy as np
from numpy import array, zeros
from scipy.misc import imread
from glob import glob

# File system management
import os
import errno

from IPython.core.debugger import set_trace
import warnings

from preprocUtils import toTorchParam
import copy


# For trainModel
import gpytorch

import preprocUtils
import preprocRandomVariables
import preprocLikelihoods
import preprocModels
import preprocKernels

import itertools
import time, datetime





def imputeDataset(dataset_name, max_T=500, data_dir='/nfs/data3/gergo/Neurofinder_update/', 
                        stamp='', force_redo=False, device='cpu',
                        returnNans = False):
    
    # Check if data has already been preprocessed
    if not os.path.exists(data_dir+dataset_name+'/preproc2P/'):
        try:
            os.makedirs(data_dir+dataset_name+'/preproc2P/')
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    imgsImputedLoaded = False
    if force_redo is False:
        savedImgsImputed = sorted(glob(data_dir+dataset_name+'/preproc2P/imgsImputed*.npy'))
        
        if savedImgsImputed:
            imgsImputed = torch.tensor(np.load(savedImgsImputed[-1]))
            imgsImputedLoaded = True
            
            if imgsImputed.size(2) < max_T:
                warnings.warn("""In the saved imgsImputed data there is only {} frames,
                              less than the requested {}, using only available number"""
                              .format(imgsImputed.size(2), max_T))
            else:
                imgsImputed = imgsImputed[:,:,:max_T]
                
        if returnNans:
            savedImgsNans = sorted(glob(data_dir+dataset_name+'/preproc2P/imgsNans*.npy'))
            
            if savedImgsNans:
                imgsNans = torch.tensor(np.load(savedImgsNans[-1]))
                
                if imgsNans.size(2) < max_T:
                    warnings.warn("""In the saved imgsNans data there is only {} frames,
                                  less than the requested {}, using only available number"""
                                  .format(imgsNans.size(2), max_T))
                else:
                    imgsNans = imgsNans[:,:,:max_T]
                
            else:
                warnings.warn("""imgsNans could not be loaded""")
                
    if not imgsImputedLoaded:
        ########################################################################
        # Load the data

        files = sorted(glob(data_dir+dataset_name+'/images/*.tiff'))
        imgs = np.array([imread(f) for f in files[:min(max_T, len(files))]])

        imgs = torch.tensor(imgs.astype(np.float32)).permute(1,2,0).to(device)


        ########################################################################
        # Find co-aligned zeros as likely missing values and impute them

        # Create filters that return zero only if there are at least m colinear zeros in either x or y direction at that location
        m = 4
        filter_size = 2*m-1
        all_filters = list()
        for i in range(m):
            single_filter =  torch.zeros(1, 1, filter_size, filter_size, 1 ) #n_filters, n_chnnel, height, width, time
            single_filter[:,:, m-1, i:i+m, :] = 1.
            all_filters.append(single_filter) 

            single_filter =  torch.zeros(1, 1, filter_size, filter_size, 1 ) #n_filters, n_chnnel, height, width, time
            single_filter[:,:, i:i+m, m-1, :] = 1.
            all_filters.append(single_filter) 

        all_filters = torch.cat(all_filters, dim=0).to(device)

        # Create convolution with multiple filters, check if any of them returns zeros (meaning it found co-linear zeros)
        missingnessFilter = torch.nn.Conv3d(1,all_filters.size(0),kernel_size=(7,7,1),stride=1,padding=(m-1, m-1, 0)).to(device)
        missingnessFilter.bias.requires_grad = False
        missingnessFilter.bias*= 0. # Zero out the bias
        missingnessFilter.weight = toTorchParam(all_filters, paramShape = all_filters.size()) # Set the weight kernels to my own

        # Do minibatches in case we're on GPU
        #missingPixelsMask = missingnessFilter(imgs.view(1,1,*imgs.shape)).min(1)[0].squeeze() # Find non-zeros over the various filters
        
        missingPixelsMask = []
        for i in range(imgs.size(2)):
            missingPixelsMask.append(
                missingnessFilter(imgs[:,:,i].view(1,1,*(imgs[:,:,i].unsqueeze(-1).shape)).detach().data).min(1)[0].squeeze().detach().data
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        missingPixelsMask = torch.stack(missingPixelsMask, dim=2)

        # Save the array with 'nan' values included
        imgsNans = copy.deepcopy(imgs.cpu())
        imgsNans[missingPixelsMask.cpu() == 0] = float('nan')
        np.save(file=data_dir+dataset_name+'/preproc2P/imgsNans' + stamp, arr=imgsNans.cpu().numpy())
        
        if not returnNans:
            del imgsNans
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        
        #Due to memory issues:
        
        
        # Store the missing pixel locations as 0s
        allOnesExceptNanIsZero = torch.ones(1, 1, *imgs.shape, device='cpu')
        allOnesExceptNanIsZero[:, :, missingPixelsMask.cpu() == 0.] = 0.

        del missingPixelsMask
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        imgsImputed = imgs.cpu().view(1,1,*imgs.shape).cpu()
        imgsImputed[allOnesExceptNanIsZero==0] = 0.

        del imgs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        while (allOnesExceptNanIsZero==0).sum() > 0:
            imputation_filter = torch.ones(1,1,3,3,3).cpu() # Just use a 3x3x3 mean filter to successively impute values
            min_nonNanNeighbors = 7.

            print('Need to impute %d values...' % ((allOnesExceptNanIsZero==0).sum()))

            divBy = torch.nn.functional.conv3d(
                allOnesExceptNanIsZero, 
                imputation_filter, 
                padding=tuple((torch.tensor(imputation_filter.shape)[2:]-1)/2))


            replaceWith = torch.nn.functional.conv3d(
                imgsImputed, 
                imputation_filter, 
                padding=tuple((torch.tensor(imputation_filter.shape)[2:]-1)/2))


            locs = ((divBy >= min_nonNanNeighbors)*(allOnesExceptNanIsZero==0.))

            imgsImputed[locs] = replaceWith[locs].div(divBy[locs])
            allOnesExceptNanIsZero[locs] = 1.

        imgsImputed = imgsImputed.squeeze().to(device)
        print('Imputed all missing values with local mean in array imgsImputed')

        # Save the imputed array
        np.save(file=data_dir+dataset_name+'/preproc2P/imgsImputed' + stamp, arr=imgsImputed.cpu().numpy())
    
    if returnNans:
        return imgsImputed, imgsNans
    else:
        return imgsImputed




###############################################################################
##################### Cross correlation #######################################
###############################################################################



def getLowLocalCrossCorr(imgsImputed, validPixel, device='cpu', 
                         crossCorrMax=0.1, crossCorrQuantile=0.5, crossCorrRestrictive=False):
    # Compute semi-local cross-correlation with non-immediate neighbours
    filter_corrOuter = torch.ones(5,5).to(device)
    filter_corrOuter[1:4,1:4] = 0. # Disregard immediate neighbors due to light contamination
    filter_corrOuter = filter_corrOuter.div(filter_corrOuter.sum()).view(1,1,5,5,1)

    meanNearby = torch.nn.functional.conv3d(
            imgsImputed.view(1,1,*imgsImputed.shape), 
            filter_corrOuter, 
            padding=tuple((torch.tensor(filter_corrOuter.shape)[2:]-1)/2)).squeeze()

    crossCorr = (((meanNearby - meanNearby.mean(2).unsqueeze(-1))*(imgsImputed - imgsImputed.mean(2).unsqueeze(-1)))
                 .mean(2)
                 .div(meanNearby.std(2)*imgsImputed.std(2))
                 )
    
    
    # Select at least N or until corrVal > thresh
    crossCorr[validPixel==0] = float('inf')
    if not crossCorrRestrictive:
        crossCorrThresh = max(crossCorrMax, np.quantile(crossCorr.abs().numpy(), q=crossCorrQuantile)) #q= 30000./crossCorr.numel()
    else:
        crossCorrThresh = min(crossCorrMax, np.quantile(crossCorr.abs().numpy(), q=crossCorrQuantile)) #q= 30000./crossCorr.numel()
        
    lowCrossCorrPixel = (crossCorr.abs() <= crossCorrThresh)
    
    return crossCorr, lowCrossCorrPixel


###############################################################################
##################### Fano factor  ############################################
###############################################################################



# Set these to whatever you want for your gaussian filter
def createGaussianFilter2D(filter_size = 71, sigma = None):

    sigma = sigma if sigma is not None else float(filter_size)/4.

    # Create a x, y coordinate grid of shape (filter_size, filter_size, 2)
    x_cord = torch.arange(filter_size)
    x_grid = x_cord.repeat(filter_size).view(filter_size, filter_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (filter_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_filter = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_filter = gaussian_filter / torch.sum(gaussian_filter)

    # Reshape to 2d depthwise convolutional weight
    gaussian_filter = gaussian_filter.view(1, 1, filter_size, filter_size)
    
    return gaussian_filter



# These seem to be unnecessary for now 
import numpy as np
def getIQR(Ainp, width = 0.2, mid=0.5, axis=2):
    A = Ainp.detach().numpy()
    qlow = Ainp.new_tensor(np.quantile(A, q=mid-width/2., axis=axis))
    #qmid = Ainp.new_tensor(np.quantile(A, q=mid, axis=axis))
    qhigh = Ainp.new_tensor(np.quantile(A, q=mid+width/2., axis=axis))
    
    return qlow,qhigh

def getRobustMeanAndSigma(Ainp, width = 0.2, **kwargs):
    qlow, qhigh = getIQR(Ainp, width=width, **kwargs)
    
    robustMean = (qhigh+qlow)/2.
    robustSigma = (qhigh-qlow)/width*0.5/1.349
    
    return robustMean, robustSigma

def getQuartileCoeffOfDispersion(Ainp, **kwargs):
    qlow, qhigh = getIQR(Ainp, width=0.5, **kwargs)
    return (qhigh-qlow).div(qhigh+qlow)

def getRobustFanofactor(Ainp, **kwargs):
    robustMean, robustSigma = getRobustMeanAndSigma(Ainp, **kwargs)
    return robustSigma.pow(2.).div(robustMean)



def getLowLocalFanoFactor(imgsImputed, device='cpu'):
    cur_mean_im = imgsImputed.mean(2)
    cur_var_im = imgsImputed.var(2)
    
    
    cur_filter = createGaussianFilter2D(filter_size = 61, sigma = 21)
    # Add hole to the filter
    hole_half = 2
    filter_middle = slice(int((cur_filter.size(2)-1)/2-hole_half), int((cur_filter.size(2)-1)/2+hole_half+1))
    cur_filter[0,0,filter_middle, filter_middle]=0.
    cur_filter = cur_filter.div(cur_filter.sum())


    logVars = torch.nn.functional.conv2d(
        cur_var_im.log().view(1,1,*cur_mean_im.shape[:2]),
        cur_filter,
        padding=tuple((torch.tensor(cur_filter.shape)[2:]-1)/2)
    )

    logMeans = torch.nn.functional.conv2d(
        cur_mean_im.log().view(1,1,*cur_mean_im.shape[:2]),
        cur_filter,
        padding=tuple((torch.tensor(cur_filter.shape)[2:]-1)/2)
    )

    numDivisor = torch.nn.functional.conv2d(
        torch.ones_like(cur_mean_im).view(1,1,*cur_mean_im.shape[:2]),
        cur_filter,
        padding=tuple((torch.tensor(cur_filter.shape)[2:]-1)/2)
    )

    localAvgFano = (logVars - logMeans).div(numDivisor).exp().squeeze()
    pixelFano = cur_var_im/cur_mean_im
    
    lowFanoPixel = (pixelFano/localAvgFano)<1.
    
    # Remove too low fano factor pixels as well (tend to mess with gain computation)
    tooLowFanoThreshold = np.quantile(pixelFano.detach().numpy(),q=0.05)
    
    lowFanoPixel = (lowFanoPixel==1) * (pixelFano>tooLowFanoThreshold)
    
    return localAvgFano, pixelFano, lowFanoPixel


def correctFanoFactor(imgsImputed, noisyPixel, max_relative_gain = 3., device='cpu'):
    cur_mean_im = imgsImputed.mean(2)
    cur_var_im = imgsImputed.var(2)
    
    countsValid = torch.zeros_like(cur_mean_im)
    countsValid[noisyPixel] = 1.


    logmean_im_toFilter = copy.deepcopy(cur_mean_im.log().clamp(-15,50))
    logmean_im_toFilter[noisyPixel==0]=0.
    logvar_im_toFilter = copy.deepcopy(cur_var_im.log().clamp(-15,50))
    logvar_im_toFilter[noisyPixel==0]=0.

    cur_filter = createGaussianFilter2D(filter_size = 101, sigma = 15)

    logVars = torch.nn.functional.conv2d(
        logvar_im_toFilter.view(1,1,*logvar_im_toFilter.shape[:2]),
        cur_filter,
        padding=tuple((torch.tensor(cur_filter.shape)[2:]-1)/2)
    )

    logMeans = torch.nn.functional.conv2d(
        logmean_im_toFilter.view(1,1,*logvar_im_toFilter.shape[:2]),
        cur_filter,
        padding=tuple((torch.tensor(cur_filter.shape)[2:]-1)/2)
    )

    numDivisor = torch.nn.functional.conv2d(
        countsValid.view(1,1,*logvar_im_toFilter.shape[:2]),
        cur_filter,
        padding=tuple((torch.tensor(cur_filter.shape)[2:]-1)/2)
    )


    pmGain = (logVars-logMeans).div(numDivisor).exp().squeeze()
    pmGain[torch.isnan(pmGain)]=pmGain[torch.isnan(pmGain)==False].min() # Set missing pixels to minimum gain
    pmGain = pmGain.clamp(min=pmGain.max()/max_relative_gain) # Some clamping
    #imagesc(pmGain)
    
    imgsImputed = imgsImputed/pmGain.unsqueeze(-1)
    
    return imgsImputed, pmGain




###############################################################################
##################### Create training data ####################################
###############################################################################



def extractTrainingData(dataset_name, max_T=500, data_dir='/nfs/data3/gergo/Neurofinder_update/', 
                        remove_PCs = None, 
                        normalize_Fano = True,
                        use_imputed_data = False,
                        stamp='', force_redo='trainingData_only', device='cpu'):
    """
    This function loads the appropriate dataset (in neurofinder format) does the necessary data imputation and
    extracts a set of background-pixels that are assumed to be representative of the gain function
    
    Saves output and intermediate results into the "preproc2P" subfolder, 
    and loads the latest one of these files depending on the value of
    "force_redo":
        - False: Load the training data if exists
        - 'trainingData_only': (default) Load the imputed images, but re-extract the training x and y
        - True: redo the whole procedure including imputation (usually unnecessary)
        
    Optionally one can remove the first "remove_PCs" temporal principal components, 
        as they often represent fluctuation due to the behavioral experiment, rather than just being noise.
        Not that removing extra PCs that are truly just responsible for noise 
        should not hurt the ultimate performance of the method (TODO: explain more)
        
    Furthermore, the photomultiplier might induce a spatially varying variance/mean ratio (Fano factor), 
        that contradicts the "spatial independence" assumption inherent in likelihoods models.
        This may be corrected by computing a robust and smooth estimate of the local gain 
        (by computing said var/mean ratio for each pixel, than filtering by taking their geometric mean),
        and dividing the training data by that filtered gain estimate.
    """
    
    # Check if data has already been preprocessed
    if not os.path.exists(data_dir+dataset_name+'/preproc2P/'):
        try:
            os.makedirs(data_dir+dataset_name+'/preproc2P/')
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
    if not force_redo:
        savedTrainingData = sorted(glob(data_dir+dataset_name+'/preproc2P/trainingData*.npz'))
        
        # If there are no saved files, just run the rest of this script
        if savedTrainingData: 
            fHandle = np.load(savedTrainingData[-1])
            trainingData = dict(fHandle)
            fHandle.close()
            for name, arr in trainingData.items():
                trainingData[name] = torch.tensor(arr).to(device)
                
            
            if trainingData['train_y'].size(1) < max_T:
                warnings.warn("""In the saved training data there is only {} frames,
                              less than the requested {}, using only available number"""
                              .format(trainingData['train_y'].size(1), max_T))
            else:
                trainingData['train_y'] = trainingData['train_y'][:,:max_T]
            
            
            return trainingData
        
    
    # Load the dataset, only re-impute if force_redo is true, if it is 'training_data '
    if use_imputed_data:
        imgsImputed = imputeDataset(dataset_name, max_T, data_dir, 
                            stamp, force_redo==True, device, returnNans=False)
    else:
        imgsImputed, imgsNans = imputeDataset(dataset_name, max_T, data_dir, 
                            stamp, force_redo==True, device, returnNans=True)
    
    
    ########################################################################
    # Find non-signal pixels and generate the training data
    
    mean_im = imgsImputed.mean(2)
    
    dataIsSaturated = ((imgsImputed==imgsImputed.max()).sum() > 1e-5*imgsImputed.numel())
  
    if dataIsSaturated:
        # Remove high quantile pixels
        highMean = np.quantile(mean_im.numpy(), 0.85)
        validPixel = (mean_im <= highMean)
    else:
        validPixel = torch.ones_like(mean_im)
        
        
    # Optionally remove the first X PCs from train_y, 
    # as there might be global modulation due to experimental conditions,
    # which may corrupt our Poissonity assumption. Arguably this should be done before crossCorr computation?
    if remove_PCs is not None:
        # # Get the PCs from full dataset
        U, S, V = torch.svd((imgsImputed-mean_im.unsqueeze(-1)).view(-1, imgsImputed.size(2)))
        
        # Get the PCs from just training data (this seems to be a bad option for locating pixels based on crossCorr)
        # U, S, V = torch.svd((train_y-train_y.mean(1).unsqueeze(-1)))
        
        to_remove = U[:, :remove_PCs].matmul(S[:remove_PCs].diag()).matmul(V[:,:remove_PCs].t())
        
        imgsImputed -= to_remove.view(*imgsImputed.size())
        
        if not use_imputed_data:
            imgsNans -= to_remove.view(*imgsImputed.size())
    
    # Compute semi-local cross-correlation with non-immediate neighbours
    crossCorr, lowCrossCorrPixel = getLowLocalCrossCorr(imgsImputed, validPixel, device=device)
    
    # Compute local fano factor
    localAvgFano, pixelFano, lowFanoPixel = getLowLocalFanoFactor(imgsImputed, device=device)

    # Get the identity of noisy pixels
    noisyPixelForFano = (((validPixel==1)*lowCrossCorrPixel*lowFanoPixel)==1)

    # Correct the photomultiplier gain
    if normalize_Fano:
        imgsImputed, pmGain = correctFanoFactor(imgsImputed, noisyPixelForFano, max_relative_gain = 50.)
        
        if not use_imputed_data:
            imgsNans.div_(pmGain.unsqueeze(-1))
    
    # Relax the definition of "low fano factor" (allow a bit higher and also too low)
    lowFanoPixel = (pixelFano/localAvgFano)<1.5
    
    # Tighten the definition of "low cross correlation"
    crossCorrThresh = min(0.1, np.quantile(crossCorr.abs().numpy(), q=0.15))
    lowCrossCorrPixel = (crossCorr.abs() <= crossCorrThresh)
    
    # Redefine noisyPixel with our changed definitions
    #noisyPixel = (((validPixel==1)*lowCrossCorrPixel*lowFanoPixel)==1)
    
    # Use only cross correlation and validity criterion, as the fano factor should have been corrected for
    noisyPixel = (((validPixel==1)*lowCrossCorrPixel)==1)
    
    
    
    # Create the training grid
    xs, ys = torch.meshgrid([torch.arange(mean_im.shape[0]).to(device), torch.arange(mean_im.shape[1]).to(device)])
    train_x = torch.cat([xs[noisyPixel].contiguous().view(-1,1), ys[noisyPixel].contiguous().view(-1,1)], dim=1).contiguous().float()
    
    if use_imputed_data:        
        train_y = imgsImputed[noisyPixel,:].view(noisyPixel.sum(), -1).contiguous().float()
    else:
        train_y = imgsNans[noisyPixel,:].view(noisyPixel.sum(), -1).contiguous().float()
        
        
    trainingData = {
                'train_x' : train_x.cpu().numpy(),
                'train_y' : train_y.cpu().numpy(),
                'mean_im' : imgsImputed.mean(2).cpu().numpy()
            }
    
    if normalize_Fano:
        trainingData['pmGain_y'] = pmGain.view(-1).cpu().numpy()
        trainingData['pmGain_x'] = torch.cat([xs.contiguous().view(-1,1), ys.contiguous().view(-1,1)], dim=1).contiguous().float().cpu().numpy()
                 

    
    # Save the training data
    np.savez(file=data_dir+dataset_name+'/preproc2P/trainingData' + stamp, 
             **trainingData)
    
    for name, arr in trainingData.items():
        trainingData[name] = torch.tensor(arr).to(device)
    
    return trainingData





###############################################################################
##################### Train the model #########################################
###############################################################################



def trainModel(dataset_name, trainingData, prior_model, likelihood_model, device='cpu',
               data_dir='/nfs/data3/gergo/Neurofinder_update/',
               stamp = '_'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%dT%H%M%S'),
               n_iter = 40, x_batchsize=2**13, y_batchsize = 200, manual_seed=2713,
               verbose = 2,
               model_grid_size = 25,
               model_interp_point_number = int(5),
               init_mll = None
              ):
    """
    Sets up and trains a model given:
        * trainingData
        * prior_model (dictionary, 'mean' is instance of gpytorch.means.Mean, 
                                   'kernel' is instance of gpytorch.kernels.Kernel)
        * likelihood_model (instance of gpytorch.likelihoods.likelihood)
    
    Saves and returns a trained "mll" (instance of gpytorch.mlls.MarginalLogLikelihood) 
        which contains both the trained model and the likelihood
    """
    
    if "cuda" in device:
        torch.cuda.set_device(torch.device(device).index)
    
    # Check if results folder already exists
    if not os.path.exists(data_dir+dataset_name+'/preproc2P/savedModels/'):
        try:
            os.makedirs(data_dir+dataset_name+'/preproc2P/savedModels/')
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    
    train_x = trainingData['train_x'].to(device)
    train_y = trainingData['train_y'].to(device)
    mean_im = trainingData['mean_im'].to(device)
    pmGainLoaded = False
    if 'pmGain_x' in trainingData:
        pmGainLoaded = True
        pmGain_x = trainingData['pmGain_x'].to(device)
        pmGain_y = trainingData['pmGain_y'].to(device)
    
    del trainingData
    
    # Register the training parameters and set seed
    trainingParams = torch.nn.Module()
    trainingParams.register_buffer("n_iter", torch.tensor(n_iter))
    if x_batchsize is not None:
        trainingParams.register_buffer("x_batchsize", torch.tensor(x_batchsize))
    if y_batchsize is not None:
        trainingParams.register_buffer("y_batchsize", torch.tensor(y_batchsize))
    trainingParams.register_buffer("manual_seed", torch.tensor(manual_seed))
    trainingParams = trainingParams.to(device)
    torch.manual_seed(manual_seed)
    
    # Get data statistics to initialise model
    dataStats = preprocUtils.getDataStatistics(train_x, train_y)
    
    #set_trace()
    
    # Set up the Gaussian Process Model    
    model = preprocModels.FlexibleVariationalGridInterpolationModel(
        train_x, train_y,
        mean_module = prior_model['mean'],
        covar_module = prior_model['kernel'],
        interp_point_number = int(model_interp_point_number),
        grid_size=model_grid_size, 
        grid_bounds=None, #grid_bounds = dataStats['x_span'],#grid_bounds=None, # Calculate bounds automatically!
        force_prior_covar = False
    ).to(device)
    
    if isinstance(model.mean_module, gpytorch.means.ConstantMean):
        preprocUtils.updateModelParams(model, 
                 {'mean_module.constant' : 
                      preprocUtils.toTorchParam(dataStats['expected_latent_photon_count'], to_log=True, ndims=2)})
        
    # Initialise the model parameters given dataStats
    preprocUtils.initialiseModelParams(model, dataStats)
    
    
    # Initialise the likelihood model given data statistics
    likelihood = likelihood_model.to(device)
    preprocUtils.updateModelParams(likelihood,
      {'log_gain' : preprocUtils.toTorchParam(dataStats['y_gain_linear'], ndims=1, to_log=True),
       'log_noise' : preprocUtils.toTorchParam((0.6*dataStats['y_gain_linear'])**2, ndims=1, to_log=True),
       'offset' : preprocUtils.toTorchParam(dataStats['y_pedestal_loc'], ndims=1),
       'log_noise_pedestal' : preprocUtils.toTorchParam(dataStats['y_pedestal_scale']**2, ndims=1, to_log=True),
       'logit_underamplified_probability': preprocUtils.toTorchParam(-2., ndims=1),
       'log_underamplified_amplitude': preprocUtils.toTorchParam(dataStats['y_gain_linear']*0.7, ndims=1, to_log=True),
      })
    

    # If a previous (usually linear) model has been fit, one can use its parameters as initialisation
    if init_mll is not None:
        fitted_model_state_dict = init_mll.model.state_dict()
        preprocUtils.updateModelParams(model, fitted_model_state_dict)
        
        fitted_likelihood_state_dict = init_mll.likelihood.state_dict()
        preprocUtils.updateModelParams(likelihood, fitted_likelihood_state_dict)
        
        # Redo the likelihood update from raw data statistics rather than the wrong linear model
        preprocUtils.updateModelParams(likelihood,
          {'log_gain' : preprocUtils.toTorchParam(dataStats['y_gain_linear'], ndims=1, to_log=True),
           'log_noise' : preprocUtils.toTorchParam((0.6*dataStats['y_gain_linear'])**2, ndims=1, to_log=True),
           'offset' : preprocUtils.toTorchParam(dataStats['y_pedestal_loc'], ndims=1),
           'log_noise_pedestal' : preprocUtils.toTorchParam(dataStats['y_pedestal_scale']**2, ndims=1, to_log=True),
           'logit_underamplified_probability': preprocUtils.toTorchParam(-2., ndims=1),
           'log_underamplified_amplitude': preprocUtils.toTorchParam(dataStats['y_gain_linear']*0.7, ndims=1, to_log=True),
          })
    
    # Define the loss function
    mll = gpytorch.mlls.VariationalMarginalLogLikelihood(likelihood, model, n_data=dataStats['n_data']).to(device)

    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ###########################################################################################

    # Model save parameters
    
    # Saved file naming convention
    priorNameDict = {
        'WhiteNoiseKernelBugfix()' : 'noPrior',
        'ScaleKernel(\n  (base_kernel): AdditiveKernel(\n    (kernels): ModuleList(\n      (0): ScaleKernel(\n        (base_kernel): AdditiveKernel(\n          (kernels): ModuleList(\n            (0): ScaleKernel(\n              (base_kernel): SymmetriseKernelLinearly(\n                (base_kernel_module): AdditiveKernel(\n                  (kernels): ModuleList(\n                    (0): RBFKernel(\n                      (log_lengthscale_prior): SmoothedBoxPrior()\n                    )\n                    (1): RBFKernel(\n                      (log_lengthscale_prior): SmoothedBoxPrior()\n                    )\n                  )\n                )\n                (center_prior): SmoothedBoxPrior()\n              )\n              (log_outputscale_prior): SmoothedBoxPrior()\n            )\n            (1): ScaleKernel(\n              (base_kernel): SymmetriseKernelRadially(\n                (base_kernel_module): MexicanHatKernel(\n                  (log_lengthscale_prior): SmoothedBoxPrior()\n                )\n                (center_prior): SmoothedBoxPrior()\n              )\n              (log_outputscale_prior): SmoothedBoxPrior()\n            )\n          )\n        )\n        (log_outputscale_prior): SmoothedBoxPrior()\n      )\n      (1): ScaleKernel(\n        (base_kernel): RBFKernel(\n          (log_lengthscale_prior): SmoothedBoxPrior()\n        )\n        (log_outputscale_prior): SmoothedBoxPrior()\n      )\n    )\n  )\n  (log_outputscale_prior): SmoothedBoxPrior()\n)': 'expertPrior',
        'ScaleKernel(\n  (base_kernel): SymmetriseKernelRadially(\n    (base_kernel_module): MexicanHatKernel(\n      (log_lengthscale_prior): SmoothedBoxPrior()\n    )\n    (center_prior): SmoothedBoxPrior()\n  )\n  (log_outputscale_prior): SmoothedBoxPrior()\n)' : 'mexRadPrior',
        'ScaleKernel(\n  (base_kernel): AdditiveKernel(\n    (kernels): ModuleList(\n      (0): ScaleKernel(\n        (base_kernel): SymmetriseKernelLinearly(\n          (base_kernel_module): MexicanHatKernel(\n            (log_lengthscale_prior): SmoothedBoxPrior()\n          )\n          (center_prior): SmoothedBoxPrior()\n        )\n        (log_outputscale_prior): SmoothedBoxPrior()\n      )\n      (1): ScaleKernel(\n        (base_kernel): RBFKernel(\n          (log_lengthscale_prior): SmoothedBoxPrior()\n        )\n        (log_outputscale_prior): SmoothedBoxPrior()\n      )\n    )\n  )\n  (log_outputscale_prior): SmoothedBoxPrior()\n)' : 'additiveMexLinPrior',
        'ScaleKernel(\n  (base_kernel): AdditiveKernel(\n    (kernels): ModuleList(\n      (0): ScaleKernel(\n        (base_kernel): SymmetriseKernelLinearly(\n          (base_kernel_module): RBFKernel(\n            (log_lengthscale_prior): SmoothedBoxPrior()\n          )\n          (center_prior): SmoothedBoxPrior()\n        )\n        (log_outputscale_prior): SmoothedBoxPrior()\n      )\n      (1): ScaleKernel(\n        (base_kernel): RBFKernel(\n          (log_lengthscale_prior): SmoothedBoxPrior()\n        )\n        (log_outputscale_prior): SmoothedBoxPrior()\n      )\n    )\n  )\n  (log_outputscale_prior): SmoothedBoxPrior()\n)': 'additiveRbfLinPrior'
        #TODO other priors
    }
    likelihoodNameDict = {
        preprocLikelihoods.LinearGainLikelihood : 'linLik',
        preprocLikelihoods.PoissonInputPhotomultiplierLikelihood : 'poissLik',
        preprocLikelihoods.PoissonInputUnderamplifiedPhotomultiplierLikelihood : 'unampLik',
    }
    
    save_fname = (data_dir+dataset_name+'/preproc2P/savedModels/mll' +
                  '_' + priorNameDict.get(str(mll.model.covar_module), 'unknownPrior') + 
                  '_' + likelihoodNameDict.get(mll.likelihood.__class__, 'unknownLik') +
                  stamp)
    
    
    
    
    
    ###########################################################################################
    
    # Train the model
    
    # For mini-batch training
    if x_batchsize is not None:
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=x_batchsize, shuffle=True, drop_last=True)

    model.train()
    likelihood.train()

    # Number of iteration
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.075)

    # We use a Learning rate scheduler from PyTorch to lower the learning rate during optimization
    # We're going to drop the learning rate by 1/10 after 3/4 of training
    # This helps the model converge to a minimum
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.75 * n_iter], gamma=0.1)
    
    # Define the train() loop
    def train():
        for i in range(n_iter):
            scheduler.step()

            if x_batchsize is not None:

                if y_batchsize is not None:
                    # Shuffle ys, then put them into index-batches
                    y_inds = torch.randn(train_y.size(1)).sort()[1]
                    tmp = 0
                    y_minibatch_inds = list()
                    while tmp < y_inds.size(0):
                        y_minibatch_inds.append(y_inds[tmp:min(tmp+y_batchsize, y_inds.size(0))])
                        tmp = tmp+y_batchsize

                    if len(y_minibatch_inds) >= 2:
                        # Drop the last (likely incomplete) batch
                        y_minibatch_inds = y_minibatch_inds[:-1]

                else:
                    y_minibatch_inds = [slice(0,train_y.size(1),1)]


                total_loss = 0.
                
                # Within each iteration, we will go over each minibatch of data
                for x_batch, y_batch in train_loader:
                    x_batch = torch.autograd.Variable(x_batch.float())
                    y_batch = torch.autograd.Variable(y_batch.float())

                    # Do minibatches in the number of time samples in y
                    ally_loss = 0.
                    for y_inds in y_minibatch_inds:

                        optimizer.zero_grad()

                        with gpytorch.settings.use_toeplitz(True):#, gpytorch.beta_features.diagonal_correction():
                            output = model(x_batch)
                            #set_trace()
                            loss = -mll(output, y_batch[:,y_inds])

                        # The actual optimization step
                        loss.backward()
                        
                        # Check for nan gradients so one can find out what caused them, and zero them so training can continue
                        modelSaved = False
                        for param in itertools.chain(model.parameters(), likelihood.parameters()):
                            if param.requires_grad:
                                if torch.isnan(param.grad).sum() > 0:
#                                     if not modelSaved:
#                                         # Register the training data
#                                         mll.register_buffer('x_batch', x_batch)
#                                         mll.register_buffer('y_batch', y_batch)
#                                         mll.register_buffer('y_inds',  y_inds)
#                                         mll.register_buffer('train_x', train_x)
#                                         mll.register_buffer('train_y', train_y)
#                                         mll.register_buffer('mean_im', mean_im)

#                                         # Register training parameters
#                                         mll.trainingParams = trainingParams
#                                         cur_fname = (data_dir+dataset_name+
#                                                    '/preproc2P/savedModels/mll_bug_'+
#                                                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%dT%H%M%S'))
#                                         torch.save(mll, cur_fname)
#                                         modelSaved = True
                                        
#                                         warnings.warn('Nan gradients present, saving bug report at: {}'.format(cur_fname))
                                    
                                    param.grad.data = torch.zeros_like(param.grad)
                                
                        
                        optimizer.step()
                        
                        ally_loss += loss.detach().data[0]
                        total_loss += loss.detach().data[0]
                        
                        if verbose > 2:
                                print('Iter %d/%d - Loss (partial y): %.3f (%.3f)' % (i + 1, n_iter, loss.data[0], optimizer.param_groups[0]['lr']))
                    
                    if verbose > 1:
                                print('Iter %d/%d - Loss (all y): %.3f (%.3f)' % (i + 1, n_iter, float(ally_loss), optimizer.param_groups[0]['lr']))


                if verbose > 0:
                    print('Iter %d/%d - Total Loss: %.3f (%.3f)' % (i + 1, n_iter, float(total_loss), optimizer.param_groups[0]['lr']))
                    
            else:
                optimizer.zero_grad()
                # We're going to use two context managers here

                # The use_toeplitz flag makes learning faster on the GPU
                # See the DKL-MNIST notebook for an explanation

                # The diagonal_correction flag improves the approximations we're making for variational inference
                # It makes running time a bit slower, but improves the optimization and predictions
                with gpytorch.settings.use_toeplitz(True):#, gpytorch.beta_features.diagonal_correction():
                    output = model(train_x)
                    loss = -mll(output, train_y)
                    
                    if verbose > 0:
                        print('Iter %d/%d - Loss: %.3f (%.3f)' % (i + 1, n_iter, loss.data[0], optimizer.param_groups[0]['lr']))

                # The actual optimization step
                loss.backward()
                optimizer.step()
                
            #################################################################    
            # Save the model after every iteration
            
            # Register the training data
            mll.register_buffer('train_x', train_x)
            mll.register_buffer('train_y', train_y)
            mll.register_buffer('mean_im', mean_im)

            if pmGainLoaded:
                mll.register_buffer('pmGain_x', pmGain_x)
                mll.register_buffer('pmGain_y', pmGain_y)
                
            # Register training parameters
            mll.trainingParams = trainingParams
            
            torch.save(mll, save_fname)
        
    
    
    train()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Move to cpu to ensure we have enough memory to register buffers
    mll = mll.cpu()
    
    # Register the training data
    mll.register_buffer('train_x', train_x.cpu())
    mll.register_buffer('train_y', train_y.cpu())
    mll.register_buffer('mean_im', mean_im.cpu())
    
    if pmGainLoaded:
        mll.register_buffer('pmGain_x', pmGain_x.cpu())
        mll.register_buffer('pmGain_y', pmGain_y.cpu())
    
    # Register training parameters
    mll.trainingParams = trainingParams.cpu()
    
    torch.save(mll, save_fname)
    
    return mll
