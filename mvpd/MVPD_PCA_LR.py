# MVPD - Principal Component Analysis (PCA) + Linear Regression Model
import os
import numpy as np
import nibabel as nib
import itertools as it
from mvpdlite.dataloader.loader_regression import ROI_Dataset
from mvpdlite.dimension_reduction import pca
from mvpdlite.func_regression.PCA_LR import PCA_LR
from mvpdlite.evaluation import var_expl
from mvpdlite.viz import viz_map

def run_PCA_LR(model_type, sub, total_run, num_pc, roidata_save_dir, roi_1_name, roi_2_name, filepath_func, filepath_mask1, filepath_mask2, results_save_dir, save_prediction):
    # create output folder if not exists
    if not os.path.exists(results_save_dir):
           os.mkdir(results_save_dir)
    
    for this_run in range(1, total_run+1):
        print("test run:", this_run) 
        # Load functioanl data and ROI masks
        # Training 
        roi_train = ROI_Dataset()
        roi_train.get_train(roidata_save_dir, roi_1_name, roi_2_name, this_run, total_run)
        ROI_1_train = roi_train[:]['ROI_1']
        ROI_2_train = roi_train[:]['ROI_2']
        # Testing 
        roi_test = ROI_Dataset()
        roi_test.get_test(roidata_save_dir, roi_1_name, roi_2_name, this_run, total_run)
        ROI_1_test = roi_test[:]['ROI_1']
        ROI_2_test = roi_test[:]['ROI_2']
    
        # Dimensionality reduction: PCA
        ROI_1_train_pca, ROI_2_train_pca, ROI_1_test_pca, ROI_2_test_pca = pca.DR_PCA(num_pc, ROI_1_train, ROI_2_train, ROI_1_test, ROI_2_test)
    
        # Linear regresson model
        predict_ROI_2_test, err_LR = PCA_LR(ROI_1_train_pca, ROI_2_train_pca, ROI_1_test_pca, ROI_2_train, ROI_2_test, num_pc) 
   
        if save_prediction:
           np.save(results_save_dir+sub+'_predict_ROI_2_'+model_type+'_testrun'+str(this_run)+'.npy', predict_ROI_2_test)
 
        # Evaluation: variance explained
        varexpl = var_expl.eval_var_expl(err_LR, ROI_2_test)
    
        # Visualization
        var_expl_map, var_expl_img = viz_map.cmetric_to_map(filepath_mask2, varexpl)
        nib.save(var_expl_img, results_save_dir+sub+'_var_expl_map_'+model_type+'_testrun'+str(this_run)+'.nii.gz')
    
    
    
