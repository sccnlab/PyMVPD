import os
import numpy as np
from mvpdlite.preprocessing.dataset_processing import apply_mask, roi_array

def load_data(sub, total_run, roi_1_name, roi_2_name, filepath_func, filepath_mask1, filepath_mask2, roidata_save_dir):
    print("start loading data of", sub)
    # create output folder if not exists
    if not os.path.exists(roidata_save_dir):
           os.mkdir(roidata_save_dir)
    
    for run in range(1, total_run+1):
        print("loading data in run", run)
        roidata_save_dir_run = roidata_save_dir + 'roi_run_' + str(run) + '/'
        
        if not os.path.exists(roidata_save_dir_run):
               os.mkdir(roidata_save_dir_run)
         
        filepath_func_run = filepath_func[run-1]
        roi_1_mask, roi_2_mask = apply_mask(filepath_func_run, filepath_mask1, filepath_mask2) # functional data in masks
        roi_1_data_run = roi_array(roi_1_mask)
        roi_2_data_run = roi_array(roi_2_mask)
    
        np.save(roidata_save_dir_run+roi_1_name+'_data_run_'+str(run)+'.npy', roi_1_data_run)
        np.save(roidata_save_dir_run+roi_2_name+'_data_run_'+str(run)+'.npy', roi_2_data_run)
    print("data loading done!") 
    return None    

