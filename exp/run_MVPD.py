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
inputinfo.results_save_dir='./results/'
inputinfo.save_prediction=False # whether to save the model prediction of the timecourses in the target ROI, False(default)

# MVPD Model Parameters
params=data_loading.structtype()
## cross validation: leave k run out
params.leave_k=1 # leave one run out(default)
## dimensionality reduction
params.dim_reduction=False # False(default)
params.dim_type='pca' # ['pca'(default), 'ica']
params.num_dim=3 # number of dimensions after dimensionality reduction, default=3
## general MVPD model class
params.mode_class='NN' # ['LR'(default), 'NN']

### LR model parameters
#### regularization 
params.lin_reg=False # False(default)
params.reg_type='RidgeCV' # ['Ridge'(default), 'Lasso', 'RidgeCV']
params.reg_strength=0.001 # regularization strength, default=0.001
params.reg_strength_list=[0.1,1.0,10.0] # RidgeCV: array of reg_strength values to try, default=[0.1,1.0,10.0]

### NN model parameters
params.NN_type='NN_standard' # ['NN_standard'(default), 'NN_dense']
params.input_size=80 # size of predictor ROI
params.output_size=53539 # size of target ROI
params.hidden_size=100 # number of units per hidden layer
params.num_hLayer=5 # number of hidden layers, default=1
params.num_epochs=5000 # number of epochs for training, default=5000
params.save_freq=1000 # checkpoint saving frequency, default=num_epochs
params.print_freq=100 # results printing out frequency, default=100
params.batch_size=32 # batch size, default=32
params.learning_rate=1e-3 # SGD optimizer learning rate, default=1e-3
params.momentum_factor=0.9 # SGD optimizer momentum, default=0.9
params.w_decay=0 # SGD optimizer weight decay (L2 penalty)
 
"""
Step 2 - Data Loading
"""
data_loading.load_data(inputinfo)

"""
Step 3 - Analysis Execution
"""
model_exec.MVPD_exec(inputinfo, params)

