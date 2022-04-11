import os
import numpy as np
import nibabel as nib
import itertools as it
from mvpd.dataloader.loader_regression import ROI_Dataset
from mvpd.dim_red import dimred_vox2dim, dimred_dim2vox
from mvpd.func_regression.lin_reg import lin_reg
from mvpd.evaluation import var_expl
from mvpd.viz import viz_map

def run_lin_reg(inputinfo, params):
    """
    Run the MVPD linear regression model.
    
    INPUT FORMAT
    inputinfo - model input info structure
       inputinfo.sub - subject/participant whose data are to be analyzed
       inputinfo.filepath_mask1 - the path to the directory containing the predictor ROI mask  
       inputinfo.filepath_mask2 - the path to the directory containing the target ROI mask
       inputinfo.roidata_save_dir - the path to the directory where the extracted functional data will be saved
       inputinfo.results_save_dir - the path to the directory where the results will be saved
       inputinfo.save_prediction - whether to save the model prediction of the timecourses in the target ROI

    params - model parameters structure
       params.total_run - the number of total experimental runs params.leave_k - the number of leave-out runs in cross validation
       params.dim_reduction - whether to perform dimensionality reduction
    """
    print("total_run:", params.total_run)
    print("leave_k_run_out:", params.leave_k)
 
    base1 = os.path.basename(inputinfo.filepath_mask1)
    base2 = os.path.basename(inputinfo.filepath_mask2)

    inputinfo.roi_1_name = base1.split('.nii')[0]
    inputinfo.roi_2_name = base2.split('.nii')[0]

    for this_run in range(1, params.total_run-params.leave_k+2):
        print("test run:", np.arange(this_run, this_run+params.leave_k))
        # Load functioanl data and ROI masks
        # Training 
        roi_train = ROI_Dataset()
        roi_train.get_train(inputinfo.roidata_save_dir, inputinfo.roi_1_name, inputinfo.roi_2_name, this_run, params.total_run, params.leave_k)
        ROI_1_train = roi_train[:]['ROI_1']
        ROI_2_train = roi_train[:]['ROI_2']
        # Testing 
        roi_test = ROI_Dataset()
        roi_test.get_test(inputinfo.roidata_save_dir, inputinfo.roi_1_name, inputinfo.roi_2_name, this_run, params.total_run, params.leave_k)
        ROI_1_test = roi_test[:]['ROI_1']
        ROI_2_test = roi_test[:]['ROI_2']
    
        # Dimensionality reduction: PCA
        if params.dim_reduction:
           print("start dimensionality reduction ...")
           ROI_1_train_trans, ROI_2_train_trans, ROI_1_test_trans, ROI_2_test_trans = dimred_vox2dim(params, ROI_1_train, ROI_2_train, ROI_1_test, ROI_2_test) 
        else:
           ROI_1_train_trans, ROI_2_train_trans, ROI_1_test_trans, ROI_2_test_trans = ROI_1_train, ROI_2_train, ROI_1_test, ROI_2_test

        # Linear regresson model 
        predict_ROI_2_test_trans = lin_reg(params, ROI_1_train_trans, ROI_2_train_trans, ROI_1_test_trans)

        if params.dim_reduction:
           print("recover from low-dimension space to voxel space ...")
           predict_ROI_2_test = dimred_dim2vox(params, ROI_2_train, predict_ROI_2_test_trans)
        else:
           predict_ROI_2_test = predict_ROI_2_test_trans

        err_LR = predict_ROI_2_test - ROI_2_test

        if inputinfo.save_prediction:
           np.save(inputinfo.results_save_dir+sub+'_predict_ROI_2_testrun'+str(this_run)+'.npy', predict_ROI_2_test)
 
        # Evaluation: variance explained
        varexpl_threshold, varexpl = var_expl.eval_var_expl(err_LR, ROI_2_test)
    
        # Visualization
        var_expl_map_threshold, var_expl_img_threshold = viz_map.cmetric_to_map(inputinfo.filepath_mask2, varexpl_threshold)
        var_expl_map, var_expl_img = viz_map.cmetric_to_map(inputinfo.filepath_mask2, varexpl)
        nib.save(var_expl_img_threshold, inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_threshold_testrun'+str(this_run)+'.nii.gz')
        nib.save(var_expl_img, inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_testrun'+str(this_run)+'.nii.gz')
    
    
    
