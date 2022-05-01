# PyMVPD

PyMVPD is a Python-based toolbox to model the multivariate interactions between brain regions using fMRI data. You can find a description of the MVPD method in this [article](https://doi.org/10.1371/journal.pcbi.1005799).

[NEW!] We added a preprint with detailed descriptions about the toolbox and example applications. Check it out [here](https://biorxiv.org/cgi/content/short/2021.10.12.464157v1)!

## MVPD Model Family
1. Linear Regression (LR) Models

Available built-in model components:
* Dimensionality reduction: principal component analysis ([PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)), independent component analysis ([ICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html))
* Regularization: [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) (L1), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) (L2), [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) (L2 with built-in cross-validation)
* Cross validation: leave k run out

  Example LR models:
  * [L2_LR](https://github.com/sccnlab/PyMVPD/tree/main/exp/run_MVPD_L2_LR.py): linear regression model with L2 regularization
  * [PCA_LR](https://github.com/sccnlab/PyMVPD/tree/main/exp/run_MVPD_PCA_LR.py): linear regression model with no regularization after principal component analysis (PCA)

2. Neural Network (NN) Models

Available built-in model components:
* NN_standard: fully connected feedforward neural network model
* NN_dense: fully connected feedforward neural network model with dense connections

  Example NN models:
  * [NN_1layer](https://github.com/sccnlab/PyMVPD/tree/main/exp/run_MVPD_NN_1layer.py): 1-layer fully-connected linear neural network model
  * [NN_5layer](https://github.com/sccnlab/PyMVPD/tree/main/exp/run_MVPD_NN_5layer.py): 5-layer fully-connected linear neural network model
  * [NN_5layer_dense](https://github.com/sccnlab/PyMVPD/tree/main/exp/run_MVPD_NN_5layer_dense.py): 5-layer fully-connected linear neural network model with dense connections

In addition to the above built-in functions, you can also customize your own functions by adding scripts under [mvpd/custom_func](https://github.com/sccnlab/PyMVPD/tree/main/mvpd/custom_func).

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
To give a quick try for MVPD analysis, you can directly run our example script [run_MVPD_test.py](https://github.com/sccnlab/PyMVPD/blob/main/exp/run_MVPD_test.py) or other example MVPD models under [exp/](https://github.com/sccnlab/PyMVPD/blob/main/exp/) (e.g. run_MVPD_xxx.py):
```
cd exp/
python3 run_MVPD_test.py
```

We have also provided a [tutorial](https://github.com/sccnlab/PyMVPD/blob/main/exp/PyMVPD_Tutorial.ipynb) in jupyter notebook. Feel free to check it out!

## Customization
To customize and run your own MVPD model, please follow the three steps:
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
## general MVPD model class
params.mode_class='NN' # ['LR'(default), 'NN']
## cross validation
params.leave_k=1 # leave k run out, default=1

### LR or NN model parameters
......

```
Step 2 - Data Loading
```
data_loading.load_data(inputinfo)
```
Step 3 - Analysis Execution
```
model_exec.MVPD_exec(inputinfo, params)
```
### Required Input Information 

- **inputinfo.sub**
  - This variable specifies the subject whose data are to be analyzed.
- **input.filepath_func**
  - This variable specifies the list of paths to the directories containing processed functional data.
- **inputinfo.filepath_mask1**
  - This variable specifies the path to the directory containing the predictor ROI mask.
- **inputinfo.filepath_mask2**
  - This variable specifies the path to the directory containing the target ROI mask.
- **inputinfo.roidata_save_dir**
  - This variable specifies the path to the directory where the extracted functional data will be saved.
- **inputinfo.results_save_dir**
  - This variable specifies the path to the directory where the results will be saved.
- **inputinfo.save_prediction** 
  - This variable specifies whether to save predicted timecourses in the target ROI.

### List of Model Parameters

NOTICE: Remember to set the value of the parameter manually if you do not want to use the default value.

- General model parameters
  - **params.mode_class** 
    - This parameter determines the general class of MVPD model to be used.
    - The available classes are 'LR' linear regression models and 'NN' neural network models.
    - The default value is 'LR'.

  - **params.leave_k**
    - This parameter determines the number of leave out runs in cross-validation.
    - The default value is 1 (leave-one-run-out procedure).

- LR model parameters
  - **params.dim_reduction**: 
    - This parameter determines whether dimensionality reduction is applied to the input data.
    - It is only used if you are using a linear regression model by setting params.mode_class='LR'
    - The default value is false.
  - **params.dim_type**: 
    - This parameter determines the type of the dimensionality reduction.
    - It is only used if you are using a linear regression model and you set "params.dim_reduction=True".
    - The available values are 'pca', 'ica', or the name of your custom dimensionality reduction method.
    - The default value is 'pca'.
  - **params.num_dim**:
    - This parameter determines the number of dimensions to keep after dimensionality reduction.
    - It is only used if you are using a linear regression model and you set "params.dim_reduction=True".
    - The default value is 3.
    
  - **params.lin_reg**:
    - This parameter determines whether to add a regularization term to the linear regression model.
    - It is only used if you are using a linear regression model by setting params.mode_class='LR'.
    - The default value is false.
  - **params.reg_type**
    - This parameter determines the type of regularization term that you want to add to the linear regression model.
    - It is only used if you are using a linear regression model with regularization by setting "params.mode_class='LR', params.lin_reg=True".
    - The available values are 'Ridge', 'Lasso', and 'RidgeCV'.
    - The default value is 'Ridge'.
  - **params.reg_strength**
    - This parameter determines the regularization strength of the chosen regularization term.
    - It is only used if you are using a linear regression model with regularization by setting "params.mode_class='LR', params.lin_reg=True".
    - The default value is '0.001'.
  - **params.reg_strength_list**
    - This parameter determines the array of regularization strength values to try in the cross-validation for Ridge regression.
    - It is only used if you are using a linear RidgeCV regression model by setting "params.mode_class='LR', params.lin_reg=True, params.reg_type='RidgeCV'".
    - The default array is [0.001, 0.01, 0.1].

- NN model parameters
  - **params.NN_type**:
    - This parameter determines the type of the neural network model to be used.
    - It is only used if you are using a neural network model by setting params.mode_class='NN'.
    - The available types are 'NN_standard', 'NN_dense', or your custom neural network model.
    - The default type is 'NN_standard'.
  - **params.input_size**:
    - This parameter determines the size of the predictor ROI and must be provided by the user.
  - **params.output_size**:
    - This parameter determines the size of the target ROI and must be provided by the user.
  - **params.hidden_size**:
    - This parameter determines the number of units per hidden layer.
    - The default value is 100.
  - **params.num_hLayer**:
    - This parameter determines the number of hidden layers.
    - The default value is 1.
  - **params.num_epochs**:
    - This parameter determines the number of epochs for training.
    - The default value is 5000.
  - **params.save_freq**:
    - This parameter determines the frequency of saving checkpoints.
    - The default value is the number of epochs for training (i.e. params.num_epochs).
  - **params.print_freq**:
    - This parameter determines the frequency of printing results.
    - The default value is 100.
  - **params.batch_size**:
    - This parameter determines the batch size when training the neural network model.
    - The default value is 32.
  - **params.learning_rate**:
    - This parameter determines the learning rate of the Stochastic Gradient Descent (SGD) optimizer.
    - The default value is 1e-3.
  - **params.momentum_factor**:
    - This parameter determines the momentum_factor of the Stochastic Gradient Descent (SGD) optimizer.
    - The default value is 0.9.
  - **params.params.w_decay**:
    - This parameter determines the weight decay (L2 penalty) of the Stochastic Gradient Descent (SGD) optimizer.
    - The default value is 0.

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
