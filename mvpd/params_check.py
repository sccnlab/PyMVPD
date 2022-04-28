import sys
from mvpd.custom_func import dimred_custom, NN_custom

def params_check(params):
    """
    Check the validity of model parameters.
    """
    # check general MVPD model class
    try:
       params.mode_class
       if params.mode_class not in ['LR', 'NN']:
          print("Warning: the general class of MVPD model you specified is not valid.")
          print("By default: MVPD linear regression (LR) model will be used.\n")
          params.mode_class='LR'
       # check neural network (NN) model parameters
       elif params.mode_class=='NN':
          # check dimensionality reduction
          if params.dim_reduction==True:
             print("Warning: dimensionality reduction is not available for MVPD neural network models.")
             params.dim_reduction=False
          # check NN model parameters
          try:
             params.NN_type
             if params.NN_type not in ['NN_standard', 'NN_dense']:
                try:
                   getattr(NN_custom, params.NN_type)
                except AttributeError:
                   print("Warning: fail to find the custom MVPD neural network model.")
                   print("By default: NN_standard model will be used.\n")
                   params.NN_type='NN_standard'
          except AttributeError:
             print("Warning: you did not specify the type of the MVPD neural network model.")
             print("By default: NN_standard model will be used.\n")
             params.NN_type='NN_standard'
          
          try:
             params.input_size
          except AttributeError:
             print("Warning: you did not specify the input size of the NN model.")
             sys.exit("Model terminated.")
         
          try:
             params.output_size
          except AttributeError:
             print("Warning: you did not specify the output size of the NN model.")
             sys.exit("Model terminated.")

          try:
             params.hidden_size
          except AttributeError:
             print("Warning: you did not specify the hidden size of the NN model.")
             print("By default: 100 units per hidden layer will be used.\n")
             params.hidden_size=100 

          try:
             params.num_hLayer
          except AttributeError:
             print("Warning: you did not specify the number of hidden layers of the NN model.")  
             print("By default: one hidden layer will be used.\n")
             params.num_hLayer=1

          try:
             params.num_epochs
          except AttributeError:
             print("Warning: you did not specify the number of epochs for training.")
             print("By default: 5000 training epochs will be used.\n")
             params.num_epochs=5000

          try:
             params.save_freq
          except AttributeError:
             print("Warning: you did not specify the checkpoint saving frequency.")
             print("By default: only the last checkpoint will be saved.\n")
             params.save_freq=params.num_epochs

          try:
             params.print_freq
          except AttributeError:
             print("Warning: you did not specify the results printing out frequency.")
             print("By default: results will be printed out for every 100 epochs.\n")
             params.print_freq=100

          try:
             params.batch_size
          except AttributeError:
             print("Warning: you did not specify the batch size during training.")
             print("By default: the batch size of 32 will be used.\n")
             params.batch_size=32

          try:
             params.learning_rate
          except AttributeError:
             print("Warning: you did not specify the learning rate of the SGD optimizer.")
             print("By default: the learning rate of 1e-3 will be used.\n")
             params.learning_rate=1e-3

          try:
             params.momentum_factor
          except AttributeError:
             print("Warning: you did not specify the momentum factor of the SGD optimizer.")
             print("By default: the momentum factor of 0.9 will be used.\n")
             params.momentum_factor=0.9

          try:
             params.w_decay
          except AttributeError:
             print("Warning: you did not specify the weight decay of the SGD optimizer.")
             print("By default: the weight decay of 0 will be used.\n")
             params.w_decay=0 
        
    except AttributeError:
       print("Warning: you did not specify the general class of MVPD model.")
       print("By default: MVPD linear regression (LR) model will be used.\n")
       params.mode_class='LR'

    # check linear regression (LR) model parameters
    if params.mode_class=='LR':
       # check dimensionality reduction
       try:
          params.dim_reduction
       except AttributeError:
          print("Warning: you did not specify whether or not to use dimensionality reduction.")
          print("By default: no dimensionality reduction will be applied.\n")
          params.dim_reduction=False 
       
       if params.dim_reduction==True:
          try:
             params.dim_type
             if params.dim_type not in ['ica', 'pca']:
                print("Attempt to apply a custom dimensionality reduction method.")
                try:
                   getattr(dimred_custom, params.dim_type)
                except AttributeError:
                   print("Warning: fail to find the custom dimensionality reduction method.")
                   print("By default: PCA will be applied.\n")
                   params.dim_type='pca'

             try:
                params.num_dim 
                if params.num_dim<=0:
                   print("Warning: the number of dimensions after dimensionality reduction you specified is not valid.")
                   print("By default: 3 dimensions will be used.\n")
                   params.num_dim=3 
             except AttributeError:
                print("Warning: you did not specify the number of dimensions after dimensionality reduction.")
                print("By default: 3 dimensions will be used.\n")
                params.num_dim=3
          except AttributeError:
             print("Warning: you did not specify a method to perform dimensionality reduction.")
             print("By default: PCA will be applied.\n")
             params.dim_type='pca'
   
       # check regularization term for linear regression
       try:
          params.lin_reg
       except AttributeError:
          print("Warning: you did not specify whether or not to add regularization to the linear regression model.")
          print("By default: no regularization will be added.\n")
          params.lin_reg=False
   
       if params.lin_reg==True: 
          try:
             params.reg_type
             if params.reg_type not in ['Ridge', 'Lasso', 'RidgeCV']:
                print("Warning: the regularization method you specified is not valid.")
                print("By default: Ridge regression (L2 regularization) will be applied.\n") 
                params.reg_type='Ridge'
   
             elif params.reg_type=='RidgeCV':
                try:
                   params.reg_strength_list
                except AttributeError:
                   print("Warning: For RidgeCV, you did not specify the array of regularization strength values to try.")
                   print("By default: the values [0.001,0.01,0.1] will be tried.\n")
                   params.reg_strength_list=[0.001,0.01,0.1] 
          except AttributeError:
             print("Warning: you did not specify the regularization method.")
             print("By default: Ridge regression (L2 regularization) will be applied.\n")
             params.reg_type='Ridge'
   
          try:
             params.reg_strength
          except AttributeError:
             print("Warning: you did not specify the regularization strength.")
             print("By default: the regularization strength 0.001 will be used.\n")
             params.reg_strength=0.001

    # check validity of leave k run out
    try:
       params.leave_k
       if params.leave_k<=0 or params.leave_k>=params.total_run:
          print("Warning: the leave-k-run-out you specified is not valid.")
          print("By default: leave-one-run-out cross-validation will be used.\n")
          params.leave_k=1
    except AttributeError:
       print("Warning: you did not specify the leave-k-run-out cross-validation.")
       print("By default: leave-one-run-out cross-validation will be used.\n")
       params.leave_k=1


