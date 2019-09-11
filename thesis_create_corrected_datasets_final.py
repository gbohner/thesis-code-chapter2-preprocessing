# Given a fitted model and a dataset, create (and save) a corrected set of tiff images

# Get which device the script should run on from command line
import sys
if len(sys.argv) == 1:
    device = 'cpu'
else:
    device = sys.argv[1]
    
# Load a trained model
import torch
import math
import gpytorch
import numpy as np
import scipy.io

if "cuda" in device:
    torch.cuda.set_device(torch.device(device).index)

from torch.utils.data import TensorDataset, DataLoader


import preprocUtils
import preprocRandomVariables
import preprocLikelihoods
import preprocModels
import preprocKernels

from collections import OrderedDict
import itertools

# File system management
import os
import errno
import zipfile

# Saving tiffs
import tifffile

# Get current git hash to ensure reproducible results
import git
git_cur_repo = git.Repo(search_parent_directories=True)
git_cur_sha = git_cur_repo.head.object.hexsha

# Get a bunch of useful utility functions for loading data and results
from thesis_final_func_defs import *

# Define which datasets do we want to work with
data_dir='/nfs/data/gergo/Neurofinder_update/'
all_dataset_names = [
                    'neurofinder.00.00', 
                     #'neurofinder.00.01', 
                     'neurofinder.00.00.test',
                     'neurofinder.00.01.test',
                     'neurofinder.01.00', 
                     'neurofinder.01.00.test',
                     'neurofinder.01.01.test',                     
                     'neurofinder.02.00', 
                     #'neurofinder.02.01',
                     'neurofinder.02.00.test',
                     'neurofinder.02.01.test',
                     'neurofinder.03.00', 
                     'neurofinder.03.00.test',
                     'neurofinder.04.00',
                     #'neurofinder.04.01',
                     'neurofinder.04.00.test',
                     #'neurofinder.04.01.test'
                    ]


stamp_git_load = '_gitsha_' + '2bd0d720de0995be6b0f1795304839f9877cb6c3'
stamp_training_type = '_rPC_1_origPMgain_useNans'
stamp_trainingCoverage = '_targetCoverage_10'
stamp_modelGridType = '_grid_30_7'#'_grid_50_9' #'_grid_50_5'

stamp_git_save = '_gitsha_' + '2bd0d72' + '_evalgit_db4ade8' # Also stamp the updated correction procedure

# Define which model we wish to use to create the corrected data
#prior='noPrior'
prior = 'expertPrior'
#lik='linLik'
lik = 'unampLik'
stamp_load = stamp_git_load + stamp_training_type + stamp_trainingCoverage + stamp_modelGridType
stamp_save = stamp_git_save + stamp_training_type + stamp_trainingCoverage + stamp_modelGridType


for dataset_name in all_dataset_names:
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(dataset_name, prior, lik, stamp_load)
    
    # Load the appropriate fitted model
    mll, model, likelihood, train_x, train_y, \
    dataStats, mean_im, pred_gain_func, corr_mean_im = \
    loadFittedModel(
        dataset_name = dataset_name,
        data_dir=data_dir,
        prior=prior, 
        lik=lik, 
        stamp = stamp_load,
        device = device
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load the appropriate data (with potentially correcting for photomultiple gain included in the mll model object
    imgsImputed = loadImputedData(
        dataset_name = dataset_name,
        data_dir=data_dir,
        device = device,
        # We can supply a model that corrects for the photomultipler gain
        mll = mll
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    # Get the MAP transformation 
    gray_levels, inverse_poiss_MAP = getInverseMapEstimate(
        likelihood,
        max_gray_level = imgsImputed.max(),
        max_photon = float(200)
    )
    gray_levels = gray_levels[2:] # Ignore negative values
    inverse_poiss_MAP = inverse_poiss_MAP[2:] # Ignore negative values

    imgsImputedPhoton = torch.stack([
        progress_bar(
            func = lambda image: im2photon(image, inverse_poiss_MAP, gray_levels, keep_zeros=True).to('cpu').detach(),
            inp = image,
            index = index,
            report = True,
            report_freq = 400
        )
        for index, image in enumerate(imgsImputed.permute(2,0,1).detach().to(device))], 
        dim=2).detach()
    
    
    # Correct the individual images with the gain
    imgsImputedCorr = imgsImputedPhoton.div(pred_gain_func.to('cpu').unsqueeze(2)).detach()
    
    # Move it to the CPU and Save the predicted gain function in MAT format for later use
    pred_gain_func = pred_gain_func.cpu()
    
    del imgsImputed
    del inverse_poiss_MAP
    del gray_levels
    del mll
    del model
    del likelihood
    del train_x
    del train_y
    del dataStats
    del mean_im
    del corr_mean_im

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    
    # Save the images as tiff images
    # Write images in the usual neurofinder format into the preproc2P subfolder,
    # within "images+stamp" / imageXXXXX.tiff

    # Make sure we saturate the uint16 range appropriately before conversion
    # Also save this multiplier number to a csv file so we can convert back easier later
    stretch_factor = float(65535) / imgsImputedCorr.max()
    imgsImputedCorrStretched = imgsImputedCorr * stretch_factor
    
    # Create the appropriate directory
    tiffs_foldername = data_dir+dataset_name+'/preproc2P/images_' + prior + '_' + lik + stamp_save
    mkdirs(tiffs_foldername)
    
    # Save the stretch_factor
    with open(tiffs_foldername + '/stretch_factor.csv','w+') as f:
        f.write("%f" % stretch_factor)
    
    
    # Save the applied spatial gain correction 
    scipy.io.savemat(tiffs_foldername + '/spatial_gain', {'spatial_gain' :pred_gain_func.detach().numpy()})
    
    
    for index, image in enumerate(imgsImputedCorrStretched.permute(2,0,1)):
        tifffile.imsave(
            tiffs_foldername + '/image' + str(index).zfill(5) + '.tif',
            image.detach().numpy().astype('uint16')
        )
        
    