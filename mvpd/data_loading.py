import os
import numpy as np
from mvpd.preprocessing.dataset_processing import apply_mask, roi_array

class structtype():
    """
    Data structure for input info and model parameters.
    """
    def __init__(self,**kwargs):
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
    def SetAttr(self,lab,val):
        self.__dict__[lab] = val

def load_data(inputinfo):
    """
    Load input data before running the chose MVPD model.
    
    INPUT FORMAT
    inputinfo - model input info structure
       inputinfo.sub - subject/participant whose data are to be analyzed
       inputinfo.filepath_func - the paths to the directories containing processed functional data 
       inputinfo.roidata_save_dir - the path to the directory where the extracted functional data will be saved 
       inputinfo.filepath_mask1 - the path to the directory containing the predictor ROI mask 
       inputinfo.filepath_mask2 - the path to the directory containing the target ROI mask 
    """
    total_run = len(inputinfo.filepath_func)
    print("total_run:", total_run)
    print("start loading data of", inputinfo.sub)
    # create roi data save folder if not exists
    if not os.path.exists(inputinfo.roidata_save_dir):
           os.mkdir(inputinfo.roidata_save_dir)

    base1 = os.path.basename(inputinfo.filepath_mask1)
    base2 = os.path.basename(inputinfo.filepath_mask2)

    inputinfo.roi_1_name = base1.split('.nii')[0]   
    inputinfo.roi_2_name = base2.split('.nii')[0] 
 
    for run in range(1, total_run+1):
        print("loading data in run", run)
        roidata_save_dir_run = inputinfo.roidata_save_dir+'roi_run_'+str(run)+'/'
        if not os.path.exists(roidata_save_dir_run):
               os.mkdir(roidata_save_dir_run)
         
        filepath_func_run = inputinfo.filepath_func[run-1]
        roi_1_mask, roi_2_mask = apply_mask(filepath_func_run, inputinfo.filepath_mask1, inputinfo.filepath_mask2) # functional data in masks
        roi_1_data_run = roi_array(roi_1_mask)
        roi_2_data_run = roi_array(roi_2_mask)
    
        np.save(roidata_save_dir_run+inputinfo.roi_1_name+'_data_run_'+str(run)+'.npy', roi_1_data_run)
        np.save(roidata_save_dir_run+inputinfo.roi_2_name+'_data_run_'+str(run)+'.npy', roi_2_data_run)
    print("data loading done!") 
    return None    

