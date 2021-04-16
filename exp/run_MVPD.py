import os, sys
sys.path.append("..")
from mvpd import data_loading, model_exec

"""
Step 1 - Analysis Specification
"""
# Subject/Participant
sub='sub-01'
# Total number of experimental runs
total_run=8

# Functional Data
filepath_func=[]
filepath_func+=['./testdata/sub-01/sub-01_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run1.nii.gz']
filepath_func+=['./testdata/sub-01/sub-01_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run2.nii.gz']
filepath_func+=['./testdata/sub-01/sub-01_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run3.nii.gz']
filepath_func+=['./testdata/sub-01/sub-01_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run4.nii.gz']
filepath_func+=['./testdata/sub-01/sub-01_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run5.nii.gz']
filepath_func+=['./testdata/sub-01/sub-01_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run6.nii.gz']
filepath_func+=['./testdata/sub-01/sub-01_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run7.nii.gz']
filepath_func+=['./testdata/sub-01/sub-01_movie_bold_space-MNI152NLin2009cAsym_preproc_denoised_run8.nii.gz']

# Predictor ROI Mask
filepath_mask1='./testdata/sub-01/sub-01_FFA_80vox_bin.nii.gz'
# Target ROI Mask
filepath_mask2='./testdata/GM_thr0.1_bin.nii.gz'

base1=os.path.basename(filepath_mask1)
base2=os.path.basename(filepath_mask2)
roi_1_name=base1.split('.nii')[0]
roi_2_name=base2.split('.nii')[0]

# Output Directory
roidata_save_dir='./testdata/roi_data/'
results_save_dir='./results/'

# MVPD Model
model_type='NN_1layer' # ['PCA_LR', 'L2_LR', 'NN_1layer', 'NN_5layer', 'NN_5layer_dense']

# only for PCA_LR
num_pc=3 # number of principal components used

# only for L2_LR
alpha=0.01 # regularization strength

# only for neural networks (NN_1layer, NN_5layer, NN_5layer_dense)
input_size=80 # size of predictor ROI
output_size=53539 # size of target ROI
hidden_size=100 # number of units per hidden layer
num_epochs=5000 # number of epochs for training
save_freq=1000 # checkpoint saving frequency
print_freq=100 # results printing out frequency
batch_size=32 
learning_rate=1e-3
momentum_factor=0.9  
w_decay=0 # weight decay (L2 penalty)

# Save Settings
save_prediction=False # default

"""
Step 2 - Data Loading
"""
data_loading.load_data(sub, total_run, roi_1_name, roi_2_name, filepath_func, filepath_mask1, filepath_mask2, roidata_save_dir)

"""
Step 3 - Analysis Execution
"""
model_exec.MVPD_exec(model_type, sub, total_run, 
                     alpha, num_pc, # reg params
                     input_size, output_size, hidden_size, num_epochs, save_freq, print_freq, batch_size, learning_rate, momentum_factor, w_decay, # nn params 
                     roidata_save_dir, roi_1_name, roi_2_name, filepath_func, filepath_mask1, filepath_mask2, results_save_dir, save_prediction)
