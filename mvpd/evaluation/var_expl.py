import numpy as np

def eval_var_expl(error, ROI_2_test):
    """
    Calculate the proportion of variance explained of individual voxels.

    INPUT FORMAT
    error - the error term in the MVPD model for the target ROI (predict_ROI_2_test - ROI_2_test) [t, n]
    ROI_2_test - functional data in the target ROI for testing [t, n]
       t - the number of timepoints in the experimental run(s)
       n - the number of voxels in the ROI

    OUTPUT FORMAT
    var_expl_threshold - the proportion of variance explained of individual voxels with the non-negative threshold [1, n]
    var_expl - the proportion of variance explained of individual voxels [1, n]
       n - the number of voxels in the ROI
    """
    var_expl = 1 - error.var(axis=0)/ROI_2_test.var(axis=0)
    var_expl_threshold = np.maximum(0, var_expl) # replace negative values with zeros
    return var_expl_threshold, var_expl

