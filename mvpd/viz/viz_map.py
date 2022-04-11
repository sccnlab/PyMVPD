"""
Visualization.
"""
import numpy as np
import nibabel as nib

def cmetric_to_map(filepath_map, c_metric):
    """
    Project connectivity metric to brain map.

    INPUT FORMAT
    filepath_map - the path to the directory where the connectivity metric is to be projected to the brain map
    c_metric - the connectivity metric that is to be projected to the brain map 

    OUTPUT FORMAT
    c_metric_map - the obtained brain map in the format of numpy array  
    c_metric_img - the obtained brain map in the format of NIfTI image
    """
    # load brain map
    brain_map = nib.load(filepath_map)
    brain_map_affine = brain_map.affine
    brain_map_data = brain_map.get_data()
    brain_map_shape = np.shape(brain_map_data) 
    nozero_idx = np.nonzero(brain_map_data)

    c_metric_map = np.zeros(brain_map_shape) 
    for i in range(np.shape(nozero_idx)[1]):
        x = nozero_idx[0][i]
        y = nozero_idx[1][i]
        z = nozero_idx[2][i]
        c_metric_map[x][y][z] = c_metric[i]
    
    c_metric_img = nib.Nifti1Image(c_metric_map, brain_map.affine)
    
    return c_metric_map, c_metric_img
