# PyMVPD

PyMVPD: MultiVariate Pattern Dependence (MVPD) Analysis in Python

## MVPD Model Family
1. Linear Regression Models
* L2_LR: linear regression model with L2 regularization
* PCA_LR: linear regression model with no regularization after principal component analysis (PCA)

2. Neural Network Models
* NN_1layer: 1-layer fully-connected linear neural network model
* NN_5layer: 5-layer fully-connected linear neural network model
* NN_5layer_dense: 5-layer fully-connected linear neural network model with dense connections

## Workflow
<img src="/PyMVPD_workflow.png" width="750"/>

## Installation & Dependencies
Before installing PyMVPD, you should first install [PyTorch](https://pytorch.org/get-started/locally/) on your system. PyTorch is used to support the construction of neural network models in PyMVPD. If you are only interested in using linear regression models for your MVPD analyses, you can go and check the light version [PyMVPD_LITE](https://github.com/sccnlab/PyMVPD_LITE) where PyTorch is not required.

The easiest way to install the PyMVPD package is to execute (possibly in a [new virtual environment](https://packaging.python.org/tutorials/installing-packages/#creating-and-using-virtual-environments)) the following command:
```
pip install PyMVPD
```
You can also install from the GitHub [repository](https://github.com/sccnlab/PyMVPD) to get the most up-to-date version.
```
git clone https://github.com/sccnlab/PyMVPD.git
pip install -r requirements.txt
```
In addition to PyTorch, the following packages need to be installed to use PyMVPD:
* python >= 3.6
* nibabel>=3.2.1
* numpy>=1.19.3
* scikit-learn>=0.20.1
* scipy>=1.1.0

## Tutorial
### Test Dataset
[Data](https://github.com/sccnlab/PyMVPD_LITE/tree/main/exp/testdata) of one subject from the [_StudyForrest_](http://studyforrest.org) dataset.

Predictor ROI: FFA - fusiform face area, 

Target ROI: GM - gray matter.

* Raw data were first preprocessed using [fMRIPrep](https://fmriprep.readthedocs.io/en/latest/index.html) and then denoised by using CompCor (see more details in [Fang et al. 2019](https://doi.org/10.31234/osf.io/qbx4m)).

### Example Analyses and Scripts
To give a quick try for MVPD analysis, you can directly run our example script [run_MVPD.py](https://github.com/sccnlab/PyMVPD/blob/main/exp/run_MVPD.py):
```
cd exp/
python3 run_MVPD.py
```

We have also provided a [tutorial](https://github.com/sccnlab/PyMVPD/blob/main/exp/PyMVPD_Tutorial.ipynb) in jupyter notebook. Feel free to check it out!

## Customization
To generate your own scripts, please follow the three steps:
```
import os
from mvpd import data_loading, model_exec
```
Step 1 - Analysis Specification
```
sub='sub-01' # subject whose data are to be analyzed
total_run=XX # total number of experimental runs

# Input functional Data
filepath_func=[]
filepath_func+=['path/to/functional/data/run1.nii.gz']
filepath_func+=['path/to/functional/data/run2.nii.gz']
......

# Input predictor ROI mask and target ROI mask
filepath_mask1='path/to/predictor/ROI/mask.nii.gz'
filepath_mask2='path/to/target/ROI/mask.nii.gz'

base1=os.path.basename(filepath_mask1)
base2=os.path.basename(filepath_mask2)
roi_1_name=base1.split('.nii')[0]
roi_2_name=base2.split('.nii')[0]

# Output Directory
roidata_save_dir='path/to/save/roidata/'
results_save_dir='path/to/save/results/'

# Choose MVPD model
model_type='L2_LR' # ['PCA_LR', 'L2_LR', 'NN_1layer', 'NN_5layer', 'NN_5layer_dense']

# Set model parameters
# Only for PCA_LR
num_pc=3 # number of principal components used 

# Only for L2_LR
alpha=0.01 

# Only for neural networks (NN_1layer, NN_5layer, NN_5layer_dense)
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

# Save predicted timecourses
save_prediction=False # default
```
Step 2 - Data Loading
```
data_loading.load_data(sub, total_run, roi_1_name, roi_2_name, filepath_func, filepath_mask1, filepath_mask2, roidata_save_dir)
```
Step 3 - Analysis Execution
```
model_exec.MVPD_exec(model_type, sub, total_run, 
                     alpha, num_pc, # reg params
                     input_size, output_size, hidden_size, num_epochs, save_freq, print_freq, batch_size, learning_rate, momentum_factor, w_decay, # nn params 
                     roidata_save_dir, roi_1_name, roi_2_name, filepath_func, filepath_mask1, filepath_mask2, results_save_dir, save_prediction)
```

## Contact
Reach out to Mengting Fang (mtfang0707@gmail.com) for questions, suggestions and feedback!
