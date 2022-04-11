import sys
from mvpd.dimension_reduction import dimred_base
from mvpd.custom_func import dimred_custom

def dimred_vox2dim(params, ROI_1_train, ROI_2_train, ROI_1_test, ROI_2_test):
    """
    Apply selected dimensionality reduction method on voxel space.

    INPUT FORMAT
    params - model parameters structure
       params.dim_type - the type of dimensionality reduction method to be used 
       params.num_dim - the number of dimensions of the low-dimensional space after dimensionality reduction
    
    ROI_1_train - functional data in the predictor ROI for training [t, n]
    ROI_2_train - functional data in the target ROI for training [t, n]
    ROI_1_test - functional data in the predictor ROI for testing [t, n]
    ROI_2_test - functional data in the target ROI for testing [t, n]
       t - the number of timepoints in the experimental run(s)
       n - the number of voxels in the ROI

    OUTPUT FORMAT
    ROI_1_train_trans - transformed functional data in the predictor ROI for training after dimensionality reduction [t, n_trans]   
    ROI_2_train_trans - transformed functional data in the target ROI for training after dimensionality reduction [t, n_trans] 
    ROI_1_test_trans - transformed functional data in the predictor ROI for testing after dimensionality reduction [t, n_trans] 
    ROI_2_test_trans - transformed functional data in the target ROI for testing after dimensionality reduction [t, n_trans]
       t - the number of timepoints in the experimental run(s)
       n_trans - the number of voxels in the ROI after dimensionality reduction 
    """
    if params.dim_type in ["ica", "pca"]:
       dimred_to_call = getattr(dimred_base, params.dim_type)
       dimred_func = dimred_to_call()
       print("dimensionality reduction method:", params.dim_type)

    else: # custom dimensionality reduction method
       dimred_to_call = getattr(dimred_custom, params.dim_type)
       dimred_func = dimred_to_call() 
       print("custom dimensionality reduction method:", params.dim_type)
       try:
          dimred_func.vox2dim
       except AttributeError:
          print("Warning: you did not specify the function 'vox2dim' in the selected custom dimensionality reduction method.")
          sys.exit("Model terminated.")
          
    ROI_1_train_trans, ROI_2_train_trans, ROI_1_test_trans, ROI_2_test_trans = dimred_func.vox2dim(ROI_1_train, ROI_2_train, ROI_1_test, ROI_2_test, params.num_dim)

    return ROI_1_train_trans, ROI_2_train_trans, ROI_1_test_trans, ROI_2_test_trans


def dimred_dim2vox(params, ROI_2_test, predict_ROI_2_test_trans):
    """
    Recover from low-dimensional space to voxel space after selected dimensionality reduction.

    INPUT FORMAT
    params - model parameters structure
       params.dim_type - the type of dimensionality reduction method to be used 
       params.num_dim - the number of dimensions of the low-dimensional space after dimensionality reduction
   
    ROI_2_test - functional data in the target ROI for testing [t, n]
    predict_ROI_2_test_trans - predicted low-dimensional data in the target ROI for testing after dimensionality reduction [t, n_trans]
       t - the number of timepoints in the experimental run(s)
       n - the number of voxels in the ROI
       n_trans - the number of voxels in the ROI after dimensionality reduction 

    OUTPUT FORMAT
    predict_ROI_2_test - predicted voxel-wise functional data in the target ROI for testing recovered from dimensionality reduction [t, n]
       t - the number of timepoints in the experimental run(s)
       n - the number of voxels in the ROI
    """
    if params.dim_type in ["ica", "pca"]:
       dimred_to_call = getattr(dimred_base, params.dim_type)
       dimred_func = dimred_to_call() 
    else: # custom dimensionality reduction method
       dimred_to_call = getattr(dimred_custom, params.dim_type)
       dimred_func = dimred_to_call()
       try:
          dimred_func.dim2vox
       except AttributeError:
          print("Warning: you did not specify the function 'dim2vox' in the selected custom dimensionality reduction method.")
          sys.exit("Model terminated.")
 
    predict_ROI_2_test = dimred_func.dim2vox(ROI_2_test, predict_ROI_2_test_trans, params.num_dim)
    return predict_ROI_2_test
