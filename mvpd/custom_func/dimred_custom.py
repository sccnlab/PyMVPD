"""
Custom dimensionality reduction methods.
"""
# Make sure to include vox2dim and dim2vox functions in your custom class.

class mycustom:
    """ 
    Apply my custom dimensionality reduction method.
    """
    def __init__(self, ROI_1_train_trans=[], ROI_2_train_trans=[], ROI_1_test_trans=[], ROI_2_test_trans=[], predict_ROI_2_test=[]):
        'Initialization'
        self.ROI_1_train_trans = []
        self.ROI_2_train_trans = []
        self.ROI_1_test_trans = []
        self.ROI_2_test_trans = []
        self.predict_ROI_2_test = []

### put your custom vox2dim and dim2vox functions below ###

#    def vox2dim(self, ROI_1_train=[], ROI_2_train=[], ROI_1_test=[], ROI_2_test=[], num_rc=3):
#        'Apply mycustom on voxel space.'
#        return ROI_1_train_trans, ROI_2_train_trans, ROI_1_test_trans, ROI_2_test_trans

#    def dim2vox(self, ROI_2_train=[], predict_ROI_2_test_trans=[], num_rc=3):
#        'Recover from low-dimensional space to voxel space.'
#        return predict_ROI_2_test
