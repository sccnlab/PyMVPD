"""
Preprocess functional data with ROI masks.
"""
import numpy as np
import nibabel as nib

def apply_mask(filepath_func, filepath_mask1, filepath_mask2):
    """ 
    Extract functional data from masks.
    """
    # load functional data 
    func = nib.load(filepath_func)
    # load masks - mask1: predictor mask, mask2: target mask
    mask1 = nib.load(filepath_mask1)
    mask2 = nib.load(filepath_mask2)
    
    # extract func data from masks
    func_values = func.get_fdata()
    mask1_values = mask1.get_fdata()
    mask2_values = mask2.get_fdata()
    mask1_values = np.expand_dims(mask1_values, axis=3)
    mask2_values = np.expand_dims(mask2_values, axis=3)
    
    mask1_values[mask1_values == 0] = np.nan
    mask2_values[mask2_values == 0] = np.nan
    
    func_mask1 = func_values * mask1_values # func data in mask 1
    func_mask2 = func_values * mask2_values # func data in mask 2
    
    return func_mask1, func_mask2


def roi_array(func_mask):
    """
    Convert functional data masked in ROIs to TxN arrays.
    """
    T = np.shape(func_mask)[3] # number of timepoints in func_mask
    func_idx = np.nonzero(~np.isnan(func_mask)) # index of non-nan voxels in mask
    N = int(np.shape(func_idx)[1]/T) # number of non-nan voxels in mask
    func_array = np.zeros([T, N])
    
    for i in range(T*N):
        x = func_idx[0][i]
        y = func_idx[1][i]
        z = func_idx[2][i]
        t = func_idx[3][i]
        n = i//T
        func_array[t, n] = func_mask[x, y, z, t]

    return func_array


