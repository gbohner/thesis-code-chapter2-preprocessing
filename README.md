# Spatial background equalisation for calcium imaging

Code for Gergo Bohner's thesis work, Chapter 2. Full text: https://github.com/gbohner/thesis-pdf-git

--

This repository is a snapshot taken at thesis submission (14 June 2019) of the original repository https://github.com/gbohner/preproc2P. It furthermore contains a more detailed README, explaining the use of individual code pieces to reproduce the figures in the thesis.


### Technical notes

I wrote all code for this chapter exclusively in Python, and the runtime environment was Ubuntu 16.04 LTS, with a conda environment that could be recreated from the ```environment.yml``` file. The code is optimised to be ran on nodes with CUDA compatible GPUs, and the python scripts accept a parameter to select the CUDA compatible GPU, e.g. ```cuda:0```. Although all code runs on ```cpu``` as well by default, the run times are significantly longer for both image processing operations and the pytorch model learning. The GPUs used had 8 GB of memory available, and therefore most operations had to be carried out on chunks of the large input image series. The chunk sizes used are for approx 512*512 frames, if the frame size is different, or your GPU has less memory, you will want to change the default chunk sizes.

Importantly, my model implementation relies on the excellent gpytorch framework, which however changes quickly. Therefore I provide the the commit hash for the gpytorch version used, and this version should be used when running my code: https://github.com/cornellius-gp/gpytorch/tree/1c8f891981a3d50e57017c7d53f25f7992f000a3


### Data input

The data used in the chapter for exclusively from the Neurofinder challenge, and therefore in a well-defined format. If you wish to use data stored in a different format, you have 2 options. You either convert your data in the same format as used by Neurofinder (see http://neurofinder.codeneuro.org/ and https://github.com/codeneuro/neurofinder-datasets for details), or you adapt the code for the ```imputeDataset()``` function in ```preprocExperimentSetup.py#L36```. This function handles the the data I/O and creates intermediate ```.npy``` data files that the rest of the functions operate on.


### Running the pipeline

The operations -- as outlined in the thesis text -- are carried out sequentially in the ```thesis_finalExperiment.ipynb``` notebook for steps 1-4:

1.) First we identify missing values in the data (generally recorded as "0" value observations), and either mark them as NaN or impute them using information from neighbours. This is discussed in section 2.3.1 and shown in figure 2.5 in the thesis.

2.) Next we identify background pixels and create the training data for our model fits. This is discussed in sections 2.2.4 and 2.3.1 and shown in figure 2.6 in the thesis.

3.) We initialise the model fits by extracting simple statistics from the data. This is discussed in section 2.3.1, and shown table 2.2 and figure 2.7 in the thesis. Visualisation for figures 2.4-2.7 concerning the data statistics are carried out in the ```Dataset_statistics.ipynb``` notebook.

4.) Then the model fits are carried out on the training data for the various prior and likelihood models proposed in the thesis chapter. Although the Ipython notebook showcases examples of fitting the models, the final model fits shown in the thesis were carried out in a noninteractive fashion by the various ```thesis_run_model_fits_*.py``` script files. The resulting fits for ```thesis_run_model_fits_2_expert_linInit.py``` are discussed in section 2.3.2 and parameter fits are shown table 2.3.

5.) The evaluation of models is discussed in section 2.3.2 involves the "corrected" dataset, which can be computed by inverting the model. This is explained in the notebook, but in the final version is carried out by python scripts ```thesis_create_corrected_datasets_final*.py```. The method with comments and intermediate visualisations is shown in the ```createCorrectedDataset.ipynb``` notebook.

6.) The final evaluation that is discussed in section 2.3.2 is carried out in the ```thesis_evaluate_results.ipynb``` notebook. This creates all subfigures that were used in the results figures for each *.00 dataset, figures 2.8-2.12.

7.) The corrected datasets are also saved, and later used in improving neural soma segmentation, discussed in Chapter 3, and https://github.com/gbohner/thesis-code-chapter3-chomp.


 

