# MVPD neural network model
import os
import importlib
import nibabel as nib
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mvpd.dataloader.loader_neural_net import ROI_Dataset
from mvpd.evaluation import var_expl
from mvpd.viz import viz_map

np.seterr(divide='ignore', invalid='ignore')

def save_model(net, optim, epoch, ckpt_fname):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        ckpt_fname)

def NN_train(params, net, trainloader, criterion, optimizer, epoch, save_dir):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs
        ROIs_1 = data['ROI_1']
        ROIs_2 = data['ROI_2']
        
        if torch.cuda.is_available():
           ROIs_1, ROIs_2 = Variable(ROIs_1.cuda()), Variable(ROIs_2.cuda())
        else:
           ROIs_1, ROIs_2 = Variable(ROIs_1), Variable(ROIs_2)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(ROIs_1)
        loss = criterion(outputs, ROIs_2)
        loss.backward()
        optimizer.step()
        # print statistics
        loss.item()
        running_loss += loss.data
        # print every print_freq mini-batches 
        if i % params.print_freq == (params.print_freq-1):
            print('[%d, %5d] loss: %.3f' %
                  (epoch, i + 1, running_loss / params.print_freq))
            running_loss = 0.0

    if epoch % params.save_freq == 0:
        save_model(net, optimizer, epoch, os.path.join(save_dir, 'MVPD_'+params.NN_type+'_'+str(params.num_hLayer)+'hLayer_%03d.ckpt' % epoch))
        print("Model saved in file: "+save_dir+"MVPD_"+params.NN_type+"_"+str(params.num_hLayer)+"hLayer_%03d.ckpt" % epoch)

def NN_test(params, net, testloader):
    net.eval()
    score = []
    ROI_2_pred = []
    ROI_2_target = []
    ROI_2_pred = np.reshape(ROI_2_pred, [-1, params.output_size])
    ROI_2_target = np.reshape(ROI_2_target, [-1, params.output_size])

    for i, data in enumerate(testloader):
            # get the inputs
            ROIs_1 = data['ROI_1']
            ROIs_2 = data['ROI_2']
            # wrap them in Variable
            if torch.cuda.is_available():
               ROIs_1, ROIs_2 = Variable(ROIs_1.cuda()), Variable(ROIs_2.cuda())
            else:
               ROIs_1, ROIs_2 = Variable(ROIs_1),  Variable(ROIs_2)
            # forward + backward + optimize
            outputs = net(ROIs_1)
            outputs_numpy = outputs.cpu().data.numpy()
            ROI_2_pred = np.concatenate([ROI_2_pred, outputs_numpy], 0)
            ROIs_2_numpy = ROIs_2.cpu().data.numpy()
            ROI_2_target = np.concatenate([ROI_2_target, ROIs_2_numpy], 0)
    err_NN = ROI_2_pred - ROI_2_target
    return err_NN, ROI_2_target, ROI_2_pred

def run_neural_net(inputinfo, params):
     print("total_run:", params.total_run)
     print("leave_k_run_out:", params.leave_k)

     base1 = os.path.basename(inputinfo.filepath_mask1)
     base2 = os.path.basename(inputinfo.filepath_mask2)

     inputinfo.roi_1_name = base1.split('.nii')[0]
     inputinfo.roi_2_name = base2.split('.nii')[0]

     if params.NN_type in ['NN_standard', 'NN_dense']:
        NN_module = importlib.import_module('mvpd.func_neural_net.%s'%(params.NN_type))
     else: # custom NN model
        NN_module = importlib.import_module('mvpd.custom_func.NN_custom')

     NN_model = getattr(NN_module, params.NN_type)

     # Device configuration
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
     for this_run in range(1, params.total_run-params.leave_k+2):
         print("test run:", np.arange(this_run, this_run+params.leave_k))
         results_save_dir_run = inputinfo.results_save_dir+inputinfo.sub+'_'+params.NN_type+'_'+str(params.num_hLayer)+'hLayer_testrun'+str(this_run)+'/'
         if not os.path.exists(results_save_dir_run):
                os.mkdir(results_save_dir_run)
         # Load functioanl data and ROI masks 
         # Training 
         roi_train = ROI_Dataset()
         roi_train.get_train(inputinfo.roidata_save_dir, inputinfo.roi_1_name, inputinfo.roi_2_name, this_run, params.total_run, params.leave_k)
         trainloader = DataLoader(roi_train, params.batch_size, shuffle=True, num_workers=0, pin_memory=True) 
         # Testing 
         roi_test = ROI_Dataset()
         roi_test.get_test(inputinfo.roidata_save_dir, inputinfo.roi_1_name, inputinfo.roi_2_name, this_run, params.total_run, params.leave_k)
         testloader = DataLoader(roi_test, params.batch_size, shuffle=False, num_workers=0, pin_memory=True) 
   
         net = NN_model(params.input_size, params.hidden_size, params.output_size, params.num_hLayer).to(device)
         
         # Loss and optimizer
         criterion = nn.MSELoss() # mean squared error
         optimizer = optim.SGD(net.parameters(), lr=params.learning_rate, momentum=params.momentum_factor, weight_decay=params.w_decay)

         for epoch in range(params.num_epochs+1):  # loop over the dataset multiple times
             NN_train(params, net, trainloader, criterion, optimizer, epoch, results_save_dir_run)
             if (epoch != 0) & (epoch % params.save_freq == 0):
                 err_NN, ROI_2_test, ROI_2_pred = NN_test(params, net, testloader)
                 
                 if inputinfo.save_prediction:
                    np.save(results_save_dir_run+inputinfo.sub+'_predict_ROI_2_'+params.NN_type+'_testrun'+str(this_run)+'_%depochs.npy' % epoch, ROI_2_pred) 

                 # Evaluation: variance explained
                 varexpl_threshold, varexpl = var_expl.eval_var_expl(err_NN, ROI_2_test)

                 # Visualization
                 var_expl_map_threshold, var_expl_img_threshold = viz_map.cmetric_to_map(inputinfo.filepath_mask2, varexpl_threshold)
                 var_expl_map, var_expl_img = viz_map.cmetric_to_map(inputinfo.filepath_mask2, varexpl)
                 nib.save(var_expl_img_threshold, results_save_dir_run+inputinfo.sub+'_var_expl_map_threshold_'+params.NN_type+'_testrun'+str(this_run)+'_%depochs.nii.gz' % epoch)
                 nib.save(var_expl_img, results_save_dir_run+inputinfo.sub+'_var_expl_map_'+params.NN_type+'_testrun'+str(this_run)+'_%depochs.nii.gz' % epoch)

