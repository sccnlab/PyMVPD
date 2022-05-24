# Template script to run MVPD analysis.

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
inputinfo.filepath_func+=['path/to/functional/data/run1.nii.gz']
inputinfo.filepath_func+=['path/to/functional/data/run2.nii.gz']
### ......

## predictor ROI mask
inputinfo.filepath_mask1='path/to/predictor/ROI/mask.nii.gz'
## target ROI mask
inputinfo.filepath_mask2='path/to/target/ROI/mask.nii.gz'
## save settings
inputinfo.roidata_save_dir='path/to/save/roidata/' # output data directory
inputinfo.results_save_dir='path/to/save/results/' # output model results directory
inputinfo.save_prediction=False # whether to save the model prediction of the timecourses in the target ROI, False(default)

# MVPD Model Parameters
params=data_loading.structtype()
## general MVPD model class
params.mode_class='LR' # ['LR'(default), 'NN']
## cross validation: leave k run out
params.leave_k=1 # leave one run out(default)

### LR or NN model parameters
### ......

"""
Step 2 - Data Loading
"""
data_loading.load_data(inputinfo)

"""
Step 3 - Analysis Execution
"""
model_exec.MVPD_exec(inputinfo, params)

