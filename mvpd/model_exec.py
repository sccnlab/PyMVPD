import sys
from mvpd.MVPD_L2_LR import run_L2_LR
from mvpd.MVPD_PCA_LR import run_PCA_LR
from mvpd.MVPD_neural_net import run_neural_net
from mvpd.avgrun_regression import avgruns_reg
from mvpd.avgrun_neural_net import avgruns_nn

def MVPD_exec(model_type, sub, total_run, 
              alpha, num_pc, # reg params
              input_size, output_size, hidden_size, num_epochs, save_freq, print_freq, batch_size, learning_rate, momentum_factor, w_decay, # nn params 
              roidata_save_dir, roi_1_name, roi_2_name, filepath_func, filepath_mask1, filepath_mask2, results_save_dir, save_prediction):
    
    if model_type == "L2_LR":
       print("\nstart running L2_LR model for", sub)
       run_L2_LR(model_type, sub, total_run, 
                 alpha, 
                 roidata_save_dir, roi_1_name, roi_2_name, filepath_func, filepath_mask1, filepath_mask2, results_save_dir, save_prediction)
    
    elif model_type == "PCA_LR":
       print("\nstart running PCA_LR model for", sub)
       run_PCA_LR(model_type, sub, total_run, 
                  num_pc, 
                  roidata_save_dir, roi_1_name, roi_2_name, filepath_func, filepath_mask1, filepath_mask2, results_save_dir, save_prediction)

    elif model_type == "NN_1layer" or model_type == "NN_5layer" or model_type == "NN_5layer_dense":
       print("\nstart running "+model_type+" model for", sub)
       run_neural_net(model_type, sub, total_run,
                      input_size, output_size, hidden_size, num_epochs, save_freq, print_freq, batch_size, learning_rate, momentum_factor, w_decay,
                      roidata_save_dir, roi_1_name, roi_2_name, filepath_func, filepath_mask1, filepath_mask2, results_save_dir, save_prediction)
    else:
       print("\nError: model type unavailable!")
       sys.exit()
 
    print("\naverage results across runs")
    if model_type == "L2_LR" or model_type == "PCA_LR": 
       avgruns_reg(model_type, sub, total_run, results_save_dir)
    elif model_type == "NN_1layer" or model_type == "NN_5layer" or model_type == "NN_5layer_dense":
       avgruns_nn(model_type, sub, total_run, num_epochs, save_freq, results_save_dir)

    log_filename = results_save_dir+sub+"_"+model_type+"_log.txt"
    log_file = open(log_filename, 'w')
    log_file.write("sub = "+sub+",\n")
    log_file.write("total_run = "+str(total_run)+",\n")
    log_file.write("predictor_roi: "+roi_1_name+",\n")
    log_file.write("target_roi: "+roi_2_name+",\n")
    log_file.write("model_type = "+model_type+",\n")

    if model_type == "L2_LR":
       log_file.write("alpha = "+str(alpha)+".\n")
    elif model_type == "PCA_LR":
       log_file.write("num_pc = "+str(num_pc)+".\n")
    elif model_type == "NN_1layer" or model_type == "NN_5layer" or model_type == "NN_5layer_dense":
       log_file.write("input_size = "+str(input_size)+",\n")
       log_file.write("output_size = "+str(output_size)+",\n")
       log_file.write("hidden_size = "+str(hidden_size)+",\n")
       log_file.write("num_epochs = "+str(num_epochs)+",\n")
       log_file.write("save_freq = "+str(save_freq)+",\n")
       log_file.write("print_freq = "+str(print_freq)+",\n")
       log_file.write("batch_size = "+str(batch_size)+",\n")
       log_file.write("learning_rate = "+str(learning_rate)+",\n")
       log_file.write("momentum_factor = "+str(momentum_factor)+",\n")
       log_file.write("w_decay = "+str(w_decay)+".\n") 
    log_file.close()

    print("\ndone!")
