import sklearn
from sklearn import linear_model

def lin_reg(params, ROI_1_train_trans, ROI_2_train_trans, ROI_1_test_trans):
    """
    Build a MVPD linear regression model.
    
    INPUT FORMAT
    params - model parameters structure
       params.lin_reg - whether to apply regularization on the linear regression model 
       params.reg_type - the type of regularization method
       
    ROI_1_train_trans - transformed functional data in the predictor ROI for training after dimensionality reduction [t, n_trans]
    ROI_2_train_trans - transformed functional data in the target ROI for training after dimensionality reduction [t, n_trans]
    ROI_1_test_trans - transformed functional data in the predictor ROI for testing after dimensionality reduction [t, n_trans] 
       t - the number of timepoints in the experimental run(s)
       n_trans - the number of voxels in the ROI after dimensionality reduction 

    OUTPUT FORMAT
    predict_ROI_2_test_trans - predicted transformed data in the target ROI for testing after dimensionality reduction [t, n_trans]
       t - the number of timepoints in the experimental run(s)
       n_trans - the number of voxels in the ROI after dimensionality reduction  
    """
    # initialize and fit model on training data
    if params.lin_reg:
       if params.reg_type == "Ridge":
          linear = linear_model.Ridge(params.reg_strength)
       elif params.reg_type == "RidgeCV":
          linear = linear_model.RidgeCV(alphas=params.reg_strength_list)
       elif params.reg_type == "Lasso":
          linear = linear_model.Lasso(params.reg_strength)
       else:
          print("No base reg method satisified! Set as L2 reg as default.")
          linear = linear_model.Ridge(params.reg_strength)
    else: # no regularization
          linear = linear_model.LinearRegression()
    
    linear.fit(ROI_1_train_trans, ROI_2_train_trans)
    # predict on test set
    predict_ROI_2_test_trans = linear.predict(ROI_1_test_trans)
    
    return predict_ROI_2_test_trans



