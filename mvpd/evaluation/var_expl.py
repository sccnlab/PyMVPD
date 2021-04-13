"""
Calculate the proportion of variance explained.
"""
import numpy as np

def eval_var_expl(error, ROI_2_test):
    """
    Proportion of variance explained of individual voxels.
    """
    var_expl = 1 - error.var(axis=0)/ROI_2_test.var(axis=0)
    var_expl = np.maximum(0, var_expl) # replace negative values with zeros
    return var_expl

