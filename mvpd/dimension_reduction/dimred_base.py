"""
Basic dimensionality reduction methods: PCA and ICA
"""
import numpy as np
import sklearn
from sklearn.decomposition import PCA, FastICA

class pca:
    """ 
    Apply principal component analysis.
    """
    def __init__(self, ROI_1_train_trans=[], ROI_2_train_trans=[], ROI_1_test_trans=[], ROI_2_test_trans=[], predict_ROI_2_test=[]):
        'Initialization'
        self.ROI_1_train_trans = []
        self.ROI_2_train_trans = [] 
        self.ROI_1_test_trans = []
        self.ROI_2_test_trans = []
        self.predict_ROI_2_test = []

    def vox2dim(self, ROI_1_train=[], ROI_2_train=[], ROI_1_test=[], ROI_2_test=[], num_pc=3):
        'Apply PCA on voxel space.'    
        # num_pc components are kept
        pca_ROI_1 = PCA(n_components=num_pc)
        pca_ROI_2 = PCA(n_components=num_pc)
        # fit the model with training data
        pca_ROI_1.fit(ROI_1_train)
        pca_ROI_2.fit(ROI_2_train)
        ROI_1_train_trans = pca_ROI_1.transform(ROI_1_train)
        ROI_2_train_trans = pca_ROI_2.transform(ROI_2_train)
        
        ROI_1_test_trans = pca_ROI_1.transform(ROI_1_test)
        ROI_2_test_trans = pca_ROI_2.transform(ROI_2_test)

        # Percentage of variance explained by each selected component
        ROI_1_train_var_ratio = pca_ROI_1.explained_variance_ratio_
        ROI_2_train_var_ratio = pca_ROI_2.explained_variance_ratio_
        print("ROI_1_var_ratio:", ROI_1_train_var_ratio)
        print("ROI_2_var_ratio:", ROI_2_train_var_ratio)

        return ROI_1_train_trans, ROI_2_train_trans, ROI_1_test_trans, ROI_2_test_trans

    def dim2vox(self, ROI_2_train=[], predict_ROI_2_test_trans=[], num_pc=3):
        'Recover from low-dimensional space to voxel space.'
        pca_ROI_2 = PCA(n_components=num_pc) 
        pca_ROI_2.fit(ROI_2_train)
        predict_ROI_2_test = pca_ROI_2.inverse_transform(predict_ROI_2_test_trans)
        print("inverse shape:", predict_ROI_2_test.shape)
        return predict_ROI_2_test

class ica:
    """ 
    Apply fast independent component analysis.
    """
    def __init__(self, ROI_1_train_trans=[], ROI_2_train_trans=[], ROI_1_test_trans=[], ROI_2_test_trans=[], predict_ROI_2_test=[]):
        'Initialization'
        self.ROI_1_train_trans = []
        self.ROI_2_train_trans = []
        self.ROI_1_test_trans = []
        self.ROI_2_test_trans = []
        self.predict_ROI_2_test = []

    def vox2dim(self, ROI_1_train=[], ROI_2_train=[], ROI_1_test=[], ROI_2_test=[], num_ic=3):
        'Apply ICA on voxel space.' 
        # num_ic components are extracted
        ica_ROI_1 = FastICA(n_components=num_ic)
        ica_ROI_2 = FastICA(n_components=num_ic)
        # fit the model with training data 
        ROI_1_train_trans = ica_ROI_1.fit_transform(ROI_1_train)
        ROI_2_train_trans = ica_ROI_2.fit_transform(ROI_2_train)

        print("ROI_1_train_trans:", ROI_1_train_trans.shape)
        print("ROI_2_test:", ROI_1_test.shape)

        ROI_1_test_trans = ica_ROI_1.fit_transform(ROI_1_test)
        ROI_2_test_trans = ica_ROI_2.fit_transform(ROI_2_test)

        return ROI_1_train_trans, ROI_2_train_trans, ROI_1_test_trans, ROI_2_test_trans

    def dim2vox(self, ROI_2_train=[], predict_ROI_2_test_trans=[], num_ic=3):
        'Recover from low-dimensional space to voxel space.'
        ica_ROI_2 = FastICA(n_components=num_ic)
        ica_ROI_2.fit(ROI_2_train)
        predict_ROI_2_test = ica_ROI_2.inverse_transform(predict_ROI_2_test_trans)
        print("inverse shape:", predict_ROI_2_test.shape)
        return predict_ROI_2_test 
 
