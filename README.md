# PyMVPD

PyMVPD: MultiVariate Pattern Dependence in Python

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

## Usage
### Example Dataset
Data of one subject from the [_StudyForrest_](http://studyforrest.org) dataset: FFA - fusiform face area, GM - grey matter.

* Raw data were first preprocessed using [fMRIPrep](https://fmriprep.readthedocs.io/en/latest/index.html) and then denoised by using CompCor (see more details in [Fang et al. 2019](https://doi.org/10.31234/osf.io/qbx4m)).

### Example Analyses and Scripts
1. Choose one MVPD model, set model parameters, input functional data and ROI masks, set output directory in [analysis_spec.py](mvpd/analysis_spec.py);
2. Run [data_loading.py](mvpd/data_loading.py) to preprocess functional data;
```
python3 data_loading.py
```
3. Run MVPD model: 
```
sh analysis_exec.sh
```

## Contact
Reach out to mtfang0707@gmail.com for questions, suggestions and feedback.
