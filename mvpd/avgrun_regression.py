"""
Average variance explained across runs for regression models.
"""
import numpy as np
import nibabel as nib

def avgruns_reg(model_type, sub, total_run, leave_k, results_save_dir):
    var_data_total_nonzero = []
    var_data_total = []
    
    for testrun in range(1, total_run-leave_k+2):
        # nonzero variance explained (threshold above zero)
        data_dir_nonzero = results_save_dir+sub+'_var_expl_map_nonzero_'+model_type+'_testrun'+str(testrun)+'.nii.gz'
        var_map_nonzero = nib.load(data_dir_nonzero)
        var_data_nonzero = var_map_nonzero.get_fdata()
        var_data_total_nonzero.append(var_data_nonzero)

        data_dir = results_save_dir+sub+'_var_expl_map_'+model_type+'_testrun'+str(testrun)+'.nii.gz'
        var_map = nib.load(data_dir) 
        var_data = var_map.get_fdata()
        var_data_total.append(var_data)
    
    var_data_avg = np.mean(var_data_total, 0)
    var_map_affine = var_map.affine 
    var_img_avg = nib.Nifti1Image(var_data_avg, var_map_affine)
    nib.save(var_img_avg, results_save_dir+sub+'_var_expl_map_'+model_type+'_avgruns.nii.gz')
   
    var_data_avg_nonzero = np.mean(var_data_total_nonzero, 0)
    var_img_avg_nonzero = nib.Nifti1Image(var_data_avg_nonzero, var_map_affine)
    nib.save(var_img_avg_nonzero, results_save_dir+sub+'_var_expl_map_nonzero_'+model_type+'_avgruns.nii.gz') 
