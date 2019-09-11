# Get which device the script should run on from command line
import sys
if len(sys.argv) == 1:
    device = 'cpu'
else:
    device = sys.argv[1]
    
from preprocExperimentSetup import *
# File system management
import os
import errno
import zipfile

import torch
import itertools

# Get current git hash to ensure reproducible results
import git
git_cur_repo = git.Repo(search_parent_directories=True)
git_cur_sha = git_cur_repo.head.object.hexsha

# Define which datasets do we want to work with
data_dir='/nfs/data/gergo/Neurofinder_update/'
all_dataset_names = ['neurofinder.00.00', 
                     'neurofinder.00.01', 
                     'neurofinder.00.00.test',
                     'neurofinder.00.01.test',
                     'neurofinder.01.00', 
                     'neurofinder.01.00.test',
                     'neurofinder.01.01.test',                     
                     'neurofinder.02.00', 
                     'neurofinder.02.01',
                     'neurofinder.02.00.test',
                     'neurofinder.02.01.test',
                     'neurofinder.03.00', 
                     'neurofinder.03.00.test',
                     'neurofinder.04.00',
                     #'neurofinder.04.01',
                     'neurofinder.04.00.test'#,
                     #'neurofinder.04.01.test'
                    ]


#import nbimporter
#from examineTrainingData import *
# Doesn't work with python scripts, just copy those 3 functions here manually for now
from thesis_final_func_defs import *


# Loading the appropriate training data type
stamp_git = '_gitsha_' + '2bd0d720de0995be6b0f1795304839f9877cb6c3'
stamp_training_type = '_rPC_1_origPMgain_useNans'


# --------------------------------------------------------------
# Set up training all models
# --------------------------------------------------------------


# Set up all combinations of priors, likelihoods and datasets
# Priors
all_priors = [
{
    'mean' : gpytorch.means.ZeroMean(),
    'kernel' : preprocKernels.WhiteNoiseKernelBugfix(variances=torch.tensor([10.]))
             },
# {
#     'mean' : gpytorch.means.ConstantMean(),
#     'kernel' : k1
# }
]

# Likelihoods
all_likelihood_classes = [
   #preprocLikelihoods.LinearGainLikelihood,
   #preprocLikelihoods.PoissonInputPhotomultiplierLikelihood,
   preprocLikelihoods.PoissonInputUnderamplifiedPhotomultiplierLikelihood
]



for likelihood_class, prior_model_base, dataset_name in itertools.product(all_likelihood_classes, all_priors, all_dataset_names):
    
    print(dataset_name, likelihood_class, prior_model_base)

    # Load the training data (created with appropriate stamps)
    trainingData = loadTrainingData(
            dataset_name = dataset_name,
            data_dir = data_dir,
            stamp = stamp_git + stamp_training_type
        )
    
    # Downsample the training data before use
    stamp_trainingCoverage = '_targetCoverage_10'
    trainingDataUniform = downsampleTrainingData(trainingData, filter_width= 15, targetCoverage=0.10)
    for name, arr in trainingData.items(): # Move training data to the appropriate device
                trainingData[name] = torch.tensor(arr).to(device)

        
    
    # Set up current prior and likelihood
    prior_model = {
        'mean' : copy.deepcopy(prior_model_base['mean']),
        'kernel' : copy.deepcopy(prior_model_base['kernel'])
    }
    
    likelihood_model = likelihood_class()
    
    # Clean cuda cache if being used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    stamp_modelGridType = '_grid_30_7_largeBatch'
    mll=trainModel(dataset_name, trainingData, prior_model, likelihood_model, device=device,
           data_dir='/nfs/data/gergo/Neurofinder_update/',
           stamp = stamp_git + stamp_training_type + stamp_trainingCoverage + stamp_modelGridType,
           n_iter = 20, 
           x_batchsize=None,#2**14, 
           y_batchsize = None,#200, 
           manual_seed=2713,
           verbose = 1,
           model_grid_size = 30,
           model_interp_point_number = 7
          )

