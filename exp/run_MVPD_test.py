# Example script to run MVPD analysis using the L2_LR, PCA_LR, NN_1layer, NN_5layer, and NN_5layer_dense model.

import os, sys
sys.path.append("..")
from mvpd import data_loading, model_exec

# MVPD analysis using L2_LR model
print("\nstart running L2_LR model\n")

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
inputinfo.results_save_dir='./testresults/L2_LR/'
inputinfo.save_prediction=False # whether to save the model prediction of the timecourses in the target ROI, False(default)

# MVPD Model Parameters
params=data_loading.structtype()
## general MVPD model class
params.mode_class='LR' # ['LR'(default), 'NN']
## cross validation: leave k run out
params.leave_k=1 # leave one run out(default)

### LR model parameters
#### dimensionality reduction
params.dim_reduction=False # False(default)
#### regularization 
params.lin_reg=True # False(default)
params.reg_type='Ridge' # ['Ridge'(default), 'Lasso', 'RidgeCV']
params.reg_strength=0.001 # regularization strength, default=0.001

"""
Step 2 - Data Loading
"""
data_loading.load_data(inputinfo)

"""
Step 3 - Analysis Execution
"""
model_exec.MVPD_exec(inputinfo, params)


"""
Validity test of toolbox
"""
import numpy as np
import nibabel as nib

def img2var(img_path, mask_path):
    """
    Extract the numpy array of variance explained in the target ROI from the corresponding brain image.
    
    INPUT FORMAT
    img_path - the path to the directory containing the brain image of variance explained
    mask_path - the path to the directory conataining the mask of the target ROI
    
    OUTPUT FORMAT
    vari_data - the extracted numpy array of variance explained in the target ROI
    
    """
    img_data = nib.load(img_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()
    nozero_idx = np.nonzero(mask_data)
    num_nozero_idx = np.shape(nozero_idx)[1]
    
    vari_data = np.zeros(num_nozero_idx)
    
    for i in range(num_nozero_idx):
        x = nozero_idx[0][i]
        y = nozero_idx[1][i]
        z = nozero_idx[2][i]
        vari_data[i] = img_data[x][y][z]
        
    return vari_data
   
mask_path = './testdata/GM_thr0.1_bin.nii.gz' # path to the mask of target ROI
tgt_path = './sub-01_L2_LR_var_expl_map_threshold_avgruns.nii.gz' # path to the pre-implemented L2_LR variExpl map
pred_path = inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_threshold_avgruns.nii.gz' # path to the obtained L2_LR variExpl map from running the MVPD model
   
tgt_vari = img2var(tgt_path, mask_path)
pred_vari = img2var(pred_path, mask_path)

# Pearson product-moment correlation coefficients
corr_L2_LR = np.corrcoef(tgt_vari, pred_vari)[0,1]

print("The Pearson correlation between the variance explained of the pre-implemented L2_LR model and your test model is: ", corr_L2_LR)

# MVPD analysis using PCA_LR model
print("\nstart running PCA_LR model\n")

"""
Step 1 - Analysis Specification
"""
# Model Input Info
## save settings
inputinfo.results_save_dir='./testresults/PCA_LR/'

# MVPD Model Parameters
## dimensionality reduction
params.dim_reduction=True # False(default)
params.dim_type='pca' # ['pca'(default), 'ica']
params.num_dim=3 # number of dimensions after dimensionality reduction, default=3
## regularization 
params.lin_reg=False # False(default)

"""
Step 2 - Data Loading
"""
#data_loading.load_data(inputinfo)

"""
Step 3 - Analysis Execution
"""
model_exec.MVPD_exec(inputinfo, params)

"""
Validity test of toolbox
"""
mask_path = './testdata/GM_thr0.1_bin.nii.gz' # path to the mask of target ROI
tgt_path = './sub-01_PCA_LR_var_expl_map_threshold_avgruns.nii.gz' # path to the pre-implemented PCA_LR variExpl map
pred_path = inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_threshold_avgruns.nii.gz' # path to the obtained PCA_LR variExpl map from running the MVPD model

tgt_vari = img2var(tgt_path, mask_path)
pred_vari = img2var(pred_path, mask_path)

# Pearson product-moment correlation coefficients
corr_PCA_LR = np.corrcoef(tgt_vari, pred_vari)[0,1]

print("The Pearson correlation between the variance explained of the pre-implemented PCA_LR model and your test model is: ", corr_PCA_LR)

# MVPD analysis using NN_1layer model
print("\nstart running NN_1layer model\n")

"""
Step 1 - Analysis Specification
"""
# Model Input Info
## save settings
inputinfo.results_save_dir='./testresults/NN_1layer/'

# MVPD Model Parameters
## general MVPD model class
params.mode_class='NN' # ['LR'(default), 'NN']
## dimensionality reduction
params.dim_reduction=False

### NN model parameters
params.NN_type='NN_standard' # ['NN_standard'(default), 'NN_dense']
params.input_size=80 # size of predictor ROI
params.output_size=53539 # size of target ROI
params.hidden_size=100 # number of units per hidden layer
params.num_hLayer=1 # number of hidden layers, default=1
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

"""
Validity test of toolbox
"""
mask_path = './testdata/GM_thr0.1_bin.nii.gz' # path to the mask of target ROI
tgt_path = './sub-01_NN_1layer_var_expl_map_threshold_avgruns.nii.gz' # path to the pre-implemented NN_1layer variExpl map
pred_path = inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_threshold_avgruns.nii.gz' # path to the obtained NN_1layer variExpl map from running the MVPD model

tgt_vari = img2var(tgt_path, mask_path)
pred_vari = img2var(pred_path, mask_path)

# Pearson product-moment correlation coefficients
corr_NN_1layer = np.corrcoef(tgt_vari, pred_vari)[0,1]

print("The Pearson correlation between the variance explained of the pre-implemented NN_1layer model and your test model is: ", corr_NN_1layer)

# MVPD analysis using NN_5layer model
print("\nstart running NN_5layer model\n")

"""
Step 1 - Analysis Specification
"""
# Model Input Info
## save settings
inputinfo.results_save_dir='./testresults/NN_5layer/'

# MVPD Model Parameters
## general MVPD model class
params.mode_class='NN' # ['LR'(default), 'NN']

### NN model parameters
params.num_hLayer=5 # number of hidden layers, default=1

"""
Step 2 - Data Loading
"""
#data_loading.load_data(inputinfo)

"""
Step 3 - Analysis Execution
"""
model_exec.MVPD_exec(inputinfo, params)

"""
Validity test of toolbox
"""
mask_path = './testdata/GM_thr0.1_bin.nii.gz' # path to the mask of target ROI
tgt_path = './sub-01_NN_5layer_var_expl_map_threshold_avgruns.nii.gz' # path to the pre-implemented NN_5layer variExpl map
pred_path = inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_threshold_avgruns.nii.gz' # path to the obtained NN_5layer variExpl map from running the MVPD model

tgt_vari = img2var(tgt_path, mask_path)
pred_vari = img2var(pred_path, mask_path)

# Pearson product-moment correlation coefficients
corr_NN_5layer = np.corrcoef(tgt_vari, pred_vari)[0,1]

print("The Pearson correlation between the variance explained of the pre-implemented NN_5layer model and your test model is: ", corr_NN_5layer)

# MVPD analysis using NN_5layer_dense model
print("\nstart running NN_5layer_dense model\n")

"""
Step 1 - Analysis Specification
"""
# Model Input Info
## save settings
inputinfo.results_save_dir='./testresults/NN_5layer_dense/'

# MVPD Model Parameters
## general MVPD model class
params.mode_class='NN' # ['LR'(default), 'NN']

### NN model parameters
params.NN_type='NN_dense' # ['NN_standard'(default), 'NN_dense']
params.num_hLayer=5 # number of hidden layers, default=1

"""
Step 2 - Data Loading
"""
data_loading.load_data(inputinfo)

"""
Step 3 - Analysis Execution
"""
model_exec.MVPD_exec(inputinfo, params)

"""
Validity test of toolbox
"""
mask_path = './testdata/GM_thr0.1_bin.nii.gz' # path to the mask of target ROI
tgt_path = './sub-01_NN_5layer_dense_var_expl_map_threshold_avgruns.nii.gz' # path to the pre-implemented NN_5layer_dense variExpl map
pred_path = inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_threshold_avgruns.nii.gz' # path to the obtained NN_5layer_dense variExpl map from running the MVPD model

tgt_vari = img2var(tgt_path, mask_path)
pred_vari = img2var(pred_path, mask_path)

# Pearson product-moment correlation coefficients
corr_NN_5layer_dense = np.corrcoef(tgt_vari, pred_vari)[0,1]

print("The Pearson correlation between the variance explained of the pre-implemented NN_5layer_dense model and your test model is: ", corr_NN_5layer_dense)


corr_list = [corr_L2_LR, corr_PCA_LR, corr_NN_1layer, corr_NN_5layer, corr_NN_5layer_dense]
min_corr = np.min(corr_list)
print("corr_list:", corr_list)
print("min_corr:", min_corr)

if min_corr > 0.95:
    print("\nThe minimum Pearson correlation is above 0.95. \nYou have passed the validity test!")
else:
    print("\nThe minimum Pearson correlation is not above 0.95. \nYou have failed the validity test. Please check the code or toolbox installation before you run formal models.")


