## run the MVPD PCA_LR model

import os, sys
sys.path.append("..")
from mvpd import data_loading, model_exec

"""
Step 1 - Analysis Specification
"""
# Model Input Info
inputinfo=data_loading.structtype()
## subject/participant
inputinfo.sub='sub-01'
## functional data
inputinfo.filepath_func=[]
inputinfo.filepath_func+=['./testdata/'+inputinfo.sub+'/'+inputinfo.sub+'_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run1.nii.gz']
inputinfo.filepath_func+=['./testdata/'+inputinfo.sub+'/'+inputinfo.sub+'_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run2.nii.gz']
inputinfo.filepath_func+=['./testdata/'+inputinfo.sub+'/'+inputinfo.sub+'_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run3.nii.gz']
inputinfo.filepath_func+=['./testdata/'+inputinfo.sub+'/'+inputinfo.sub+'_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run4.nii.gz']
inputinfo.filepath_func+=['./testdata/'+inputinfo.sub+'/'+inputinfo.sub+'_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run5.nii.gz']
inputinfo.filepath_func+=['./testdata/'+inputinfo.sub+'/'+inputinfo.sub+'_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run6.nii.gz']
inputinfo.filepath_func+=['./testdata/'+inputinfo.sub+'/'+inputinfo.sub+'_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run7.nii.gz']
inputinfo.filepath_func+=['./testdata/'+inputinfo.sub+'/'+inputinfo.sub+'_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run8.nii.gz']
## predictor ROI mask
inputinfo.filepath_mask1='./testdata/'+inputinfo.sub+'/'+inputinfo.sub+'_FFA_80vox_bin.nii.gz'
## target ROI mask
inputinfo.filepath_mask2='./testdata/GM_thr0.1_bin.nii.gz'
## save settings
inputinfo.roidata_save_dir='./testdata/'+inputinfo.sub+'_roi_data/'
inputinfo.results_save_dir='./testresults/'
inputinfo.save_prediction=False # whether to save the model prediction of the timecourses in the target ROI, False(default)

# MVPD Model Parameters
params=data_loading.structtype()
## general MVPD model class
params.mode_class='LR' # ['LR'(default), 'NN']
## cross validation: leave k run out
params.leave_k=1 # leave one run out(default)

### LR model parameters
#### dimensionality reduction
params.dim_reduction=True # False(default)
params.dim_type='pca' # ['pca'(default), 'ica']
params.num_dim=3 # number of dimensions after dimensionality reduction, default=3
#### regularization 
params.lin_reg=False # False(default)

"""
Step 2 - Data Loading
"""
data_loading.load_data(inputinfo)

"""
Step 3 - Analysis Execution
"""
model_exec.MVPD_exec(inputinfo, params)

