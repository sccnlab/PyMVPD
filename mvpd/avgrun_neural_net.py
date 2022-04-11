import numpy as np
import nibabel as nib

def avgruns_nn(inputinfo, params):
    """
    Average variance explained across runs for MVPD neural network models.
    
    INPUT FORMAT
    inputinfo - model input info structure
       inputinfo.sub - subject/participant whose data are to be analyzed
       inputinfo.results_save_dir - the path to the directory where the results will be saved

    params - model parameters structure
       params.total_run - the total number of experimental runs
       params.leave_k - the number of leave-out runs in cross validation

       params.NN_type - the type of MVPD NN model to be used
       params.num_hLayer - the number of hidden layers 
       params.num_epochs - the total number of epochs for training 
       params.save_freq - the checkpoint saving frequency
    """
    for epoch in range(params.save_freq, params.num_epochs+1, params.save_freq):
        var_data_total = []
        var_data_total_threshold = []

        for testrun in range(1, params.total_run-params.leave_k+2):
            results_save_dir_run = inputinfo.results_save_dir+inputinfo.sub+'_'+params.NN_type+'_'+str(params.num_hLayer)+'hLayer_testrun'+str(testrun)+'/'
            data_dir = results_save_dir_run+inputinfo.sub+'_var_expl_map_'+params.NN_type+'_testrun'+str(testrun)+'_'+str(epoch)+'epochs.nii.gz'
            var_map = nib.load(data_dir) 
            var_data = var_map.get_fdata()
            var_data_total.append(var_data)
   
            # variance explained thresholding above zero
            data_dir_threshold = results_save_dir_run+inputinfo.sub+'_var_expl_map_threshold_'+params.NN_type+'_testrun'+str(testrun)+'_'+str(epoch)+'epochs.nii.gz'
            var_map_threshold = nib.load(data_dir_threshold)
            var_data_threshold = var_map_threshold.get_fdata()
            var_data_total_threshold.append(var_data_threshold)  

        var_data_avg = np.mean(var_data_total, 0)
        var_map_affine = var_map.affine 
        var_img_avg = nib.Nifti1Image(var_data_avg, var_map_affine) 
        nib.save(var_img_avg, inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_'+params.NN_type+'_avgruns_'+str(epoch)+'epochs.nii.gz')
    
        var_data_avg_threshold = np.mean(var_data_total_threshold, 0)
        var_img_avg_threshold = nib.Nifti1Image(var_data_avg_threshold, var_map_affine)
        nib.save(var_img_avg_threshold, inputinfo.results_save_dir+inputinfo.sub+'_var_expl_map_threshold_'+params.NN_type+'_avgruns_'+str(epoch)+'epochs.nii.gz')
        
