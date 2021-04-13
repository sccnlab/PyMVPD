"""
MVPD - Linear Regression Model + Principal Component Analysis (PCA)
"""
import sklearn
from sklearn.decomposition import PCA
from sklearn import linear_model

def PCA_LR(ROI_1_train_pca, ROI_2_train_pca, ROI_1_test_pca, ROI_2_train, ROI_2_test, num_pc):
    """
    Build a linear regression model with no regularization after PCA.
    """
    # initialize and fit model on training data
    linear = linear_model.LinearRegression()
    linear.fit(ROI_1_train_pca, ROI_2_train_pca)
    # predict on test set
    predict_ROI_2_test_pca = linear.predict(ROI_1_test_pca)
    # reconstruct ROI_2 prediction from low-dimensional subspace of pc to voxel space
    pca_ROI_2 = PCA(n_components=num_pc)
    pca_ROI_2.fit(ROI_2_train)
    predict_ROI_2_test = pca_ROI_2.inverse_transform(predict_ROI_2_test_pca)
    err_LR = predict_ROI_2_test - ROI_2_test
    return predict_ROI_2_test, err_LR



