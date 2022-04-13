# PyMVPD

PyMVPD is a Python-based toolbox to model the multivariate interactions between brain regions using fMRI data. You can find a description of the MVPD method in this [article](https://doi.org/10.1371/journal.pcbi.1005799).

[NEW!] We added a preprint with detailed descriptions about the toolbox and example applications. Check it out [here](https://biorxiv.org/cgi/content/short/2021.10.12.464157v1)!

## MVPD Model Family
1. Linear Regression (LR) Models
Available built-in model components:
* Dimensionality reduction: principal component analysis ([PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)), independent component analysis ([ICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html))
* Regularization: [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) (L1), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) (L2), [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) (L2 with build-in cross-validation)
* Cross validation: leave k run out

  Example LR models:
  * L2_LR: linear regression model with L2 regularization
  * PCA_LR: linear regression model with no regularization after principal component analysis (PCA)

2. Neural Network (NN) Models
Available built-in model components:
* NN_standard: fully connected feedforward neural network model
* NN_dense: fully connected feedforward neural network model with dense connections

  Example NN models:
  * NN_1layer: 1-layer fully-connected linear neural network model
  * NN_5layer: 5-layer fully-connected linear neural network model
  * NN_5layer_dense: 5-layer fully-connected linear neural network model with dense connections

In addition to these pre-implemented models, you can also customize your own MVPD models by adding scripts under [mvpd/](https://github.com/sccnlab/PyMVPD/tree/main/mvpd).

## Workflow
<img src="/PyMVPD_workflow.png" width="750"/>

## Installation & Dependencies
Before installing PyMVPD, you should first install [PyTorch](https://pytorch.org/get-started/locally/) on your system. PyTorch is used to support the construction of neural network models in PyMVPD. If you are only interested in using linear regression models for your MVPD analyses, you can go and check the lite version [PyMVPD_LITE](https://github.com/sccnlab/PyMVPD_LITE) where PyTorch is not required.

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
# Model Input Info
inputinfo=data_loading.structtype()
inputinfo.sub='sub-01' # subject whose data are to be analyzed
filepath_func=[] # input functional Data
filepath_func+=['path/to/functional/data/run1.nii.gz']
filepath_func+=['path/to/functional/data/run2.nii.gz']
......

inputinfo.filepath_mask1='path/to/predictor/ROI/mask.nii.gz' # predictor ROI mask
inputinfo.filepath_mask2='path/to/target/ROI/mask.nii.gz' # target ROI mask

inputinfo.roidata_save_dir='path/to/save/roidata/' # output data directory
inputinfo.results_save_dir='path/to/save/results/' # output model results directory
inputinfo.save_prediction=False # whether to save predicted timecourses in the target ROI

# MVPD Model Parameters
params=data_loading.structtype()
params.leave_k=1 # cross validation: leave k run out, default=1

## general MVPD model class
params.mode_class='NN' # ['LR'(default), 'NN']

### LR model parameters
#### dimensionality reduction
params.dim_reduction=True # whether to perform dimensionality reduction on input data
params.dim_type='pca' # ['pca'(default), 'ica']
params.num_dim=3 # number of dimensions after dimensionality reduction, default=3
#### regularization
params.lin_reg=True # whether to add regularization term
params.reg_type='Ridge' # ['Ridge'(default), 'Lasso', 'RidgeCV']
params.reg_strength=0.001 # regularization strength, default=0.001
#params.reg_strength_list=[0.1,1.0,10.0] # only for RidgeCV: array of reg_strength values to try, default=(0.1,1.0,10.0)

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

```
Step 2 - Data Loading
```
data_loading.load_data(inputinfo)
```
Step 3 - Analysis Execution
```
model_exec.MVPD_exec(inputinfo, params)
```

## Citation
PyMVPD has been used in:

- PyMVPD: A toolbox for multivariate pattern dependence. [PDF](https://www.biorxiv.org/content/10.1101/2021.10.12.464157v1.full.pdf) <br/>
Fang, M., Poskanzer, C., Anzellotti, S.

- Identifying hubs that integrate responses across multiple category-selective regions.<br/>
Fang, M., Aglinskas, A., Li, Y., Anzellotti, S. 

If you plan to use the toolbox, please consider citing this.

```
@article{fang2021pymvpd,
  title={PyMVPD: A toolbox for multivariate pattern dependence},
  author={Fang, Mengting and Poskanzer, Craig and Anzellotti, Stefano},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contact
Reach out to Mengting Fang (mtfang0707@gmail.com) for questions, suggestions and feedback!
