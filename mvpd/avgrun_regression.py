import numpy as np
import nibabel as nib

def avgruns_reg(inputinfo, params):
    """
    Average variance explained across runs for MVPD linear regression models.
    
    INPUT FORMAT
    inputinfo - model input info structure
       inputinfo.sub - subject/participant whose data are to be analyzed
       inputinfo.results_save_dir - the path to the directory where the results will be saved

    params - model parameters structure
       params.total_run - the total number of experimental runs
       params.leave_k - the number of leave-out runs in cross validation
    """ 
    var_data_total = []    
    var_data_total_threshold = []

    for testrun in range(1, params.total_run-params.leave_k+2):
        data_dir = inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_testrun'+str(testrun)+'.nii.gz'
        var_map = nib.load(data_dir) 
        var_data = var_map.get_fdata()
        var_data_total.append(var_data)

        data_dir = inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_testrun'+str(testrun)+'.nii.gz'
        var_map = nib.load(data_dir)
        var_data = var_map.get_fdata()
        var_data_total.append(var_data)

        # variance explained thresholding above zero
        data_dir_threshold = inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_threshold_testrun'+str(testrun)+'.nii.gz'
        var_map_threshold = nib.load(data_dir_threshold)
        var_data_threshold = var_map_threshold.get_fdata()
        var_data_total_threshold.append(var_data_threshold)
    
    var_data_avg = np.mean(var_data_total, 0)
    var_map_affine = var_map.affine 
    var_img_avg = nib.Nifti1Image(var_data_avg, var_map_affine)
    nib.save(var_img_avg, inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_avgruns.nii.gz')

    var_data_avg_threshold = np.mean(var_data_total_threshold, 0)
    var_img_avg_threshold = nib.Nifti1Image(var_data_avg_threshold, var_map_affine)
    nib.save(var_img_avg_threshold, inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_threshold_avgruns.nii.gz')    

