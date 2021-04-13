import numpy as np
import sklearn
from sklearn.decomposition import PCA

def DR_PCA(num_pc, ROI_1_train, ROI_2_train, ROI_1_test, ROI_2_test):
    """ 
    Apply principal component analysis.
    """
    # num_pc components are kept
    pca_ROI_1 = PCA(n_components=num_pc)
    pca_ROI_2 = PCA(n_components=num_pc)
    # fit the model with training data
    pca_ROI_1.fit(ROI_1_train)
    pca_ROI_2.fit(ROI_2_train)
    ROI_1_train_pca = pca_ROI_1.transform(ROI_1_train)
    ROI_2_train_pca = pca_ROI_2.transform(ROI_2_train)
    ROI_1_test_pca = pca_ROI_1.transform(ROI_1_test)
    ROI_2_test_pca = pca_ROI_2.transform(ROI_2_test)

    # Percentage of variance explained by each selected component
    ROI_1_train_var_ratio = pca_ROI_1.explained_variance_ratio_
    ROI_2_train_var_ratio = pca_ROI_2.explained_variance_ratio_
    print("ROI_1_var_ratio:", ROI_1_train_var_ratio)
    print("ROI_2_var_ratio:", ROI_2_train_var_ratio)

    return ROI_1_train_pca, ROI_2_train_pca, ROI_1_test_pca, ROI_2_test_pca


