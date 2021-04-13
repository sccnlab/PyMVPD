"""
MVPD - Linear Regression Model + L2 Regularization 
"""
import sklearn
from sklearn import linear_model

def L2_LR(ROI_1_train, ROI_2_train, ROI_1_test, ROI_2_test, alpha):
    """
    Build a linear regression model with L2 regularization.
    """
    # initialize and fit model on training data
    ridgereg = linear_model.Ridge(alpha)
    ridgereg.fit(ROI_1_train, ROI_2_train)
    # predict on test set
    predict_ROI_2_test = ridgereg.predict(ROI_1_test)
    err_LR = predict_ROI_2_test - ROI_2_test
    # coefficients and intercept
    coef = ridgereg.coef_
    intercept = ridgereg.intercept_

    return predict_ROI_2_test, err_LR



