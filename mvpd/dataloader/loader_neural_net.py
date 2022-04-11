import numpy as np
import itertools as it
import torch
from torch.utils.data import Dataset

class ROI_Dataset(Dataset):
    """
    Implement the data loader of MVPD neural network models.
    """
    def __init__(self, ROIs_1=[], ROIs_2=[]):
        'Initialization'
        self.ROIs_1 = []
        self.ROIs_2 = []

    def get_train(self, roidata_save_dir=None, roi_1_name=None, roi_2_name=None, this_run=0, total_run=0, leave_k=0):
        NULL = True # dataset is empty
        for run in it.chain(range(1,this_run), range(this_run+leave_k,total_run+1)):
                roidata_save_dir_run = roidata_save_dir+'roi_run_'+str(run)+'/'
                roi_1_data_run = np.load(roidata_save_dir_run+roi_1_name+'_data_run_'+str(run)+'.npy')
                roi_2_data_run = np.load(roidata_save_dir_run+roi_2_name+'_data_run_'+str(run)+'.npy')                
               
                # Concatenate data from each run for training
                if NULL:
                     roi_1_data = roi_1_data_run
                     roi_2_data = roi_2_data_run
                     NULL = False
                else: 
                     roi_1_data = np.concatenate([roi_1_data, roi_1_data_run], 0) 
                     roi_2_data = np.concatenate([roi_2_data, roi_2_data_run], 0)

        # BatchNorm: dataset size modulo batch size is equal to 1  
        num_data = np.shape(roi_1_data)[0]
        del_idx = np.random.randint(0, num_data)
        roi_1_data = np.delete(roi_1_data, del_idx, 0)
        roi_2_data = np.delete(roi_2_data, del_idx, 0)
 
        self.ROIs_1 = torch.from_numpy(roi_1_data)
        self.ROIs_2 = torch.from_numpy(roi_2_data)
        self.ROIs_1 = self.ROIs_1.type(torch.FloatTensor)
        self.ROIs_2 = self.ROIs_2.type(torch.FloatTensor)
      
    def get_test(self, roidata_save_dir=None, roi_1_name=None, roi_2_name=None, this_run=0, total_run=0, leave_k=0):
        NULL = True # dataset is empty  
        for run in range(this_run, this_run+leave_k):
                roidata_save_dir_run = roidata_save_dir+'roi_run_'+str(run)+'/'
                roi_1_data_run = np.load(roidata_save_dir_run+roi_1_name+'_data_run_'+str(run)+'.npy')
                roi_2_data_run = np.load(roidata_save_dir_run+roi_2_name+'_data_run_'+str(run)+'.npy') 
                # Concatenate data from each run for training
                if NULL:
                     roi_1_data = roi_1_data_run
                     roi_2_data = roi_2_data_run
                     NULL = False
                else:
                     roi_1_data = np.concatenate([roi_1_data, roi_1_data_run], 0)
                     roi_2_data = np.concatenate([roi_2_data, roi_2_data_run], 0)
 
        'Convert ndarrays in sample to Tensors'
        self.ROIs_1 = torch.from_numpy(roi_1_data)
        self.ROIs_2 = torch.from_numpy(roi_2_data)
        self.ROIs_1 = self.ROIs_1.type(torch.FloatTensor)
        self.ROIs_2 = self.ROIs_2.type(torch.FloatTensor)

    def __len__(self):
        'Denotes the total number of predictor samples'
        return len(self.ROIs_1)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        ROI_1 = self.ROIs_1[idx]
        ROI_2 = self.ROIs_2[idx]
        sample = {'ROI_1': ROI_1, 'ROI_2': ROI_2}
        return sample

