import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser

"""
This file is adapted from:
https://github.com/chengtan9907/OpenSTL/blob/OpenSTLv0.3.0/examples/tutorial.ipynb

Original Authors: CAIRI AI Lab
Modifications by: Michael Trimboli
"""

#Note: If you have a desired model you would like to recall the weights for testing, I would recommend making two copies in your work_dirs: 
    # 1). A backup folder that archives your initial training and testing results. DO NOT USE THIS FOR TESTING! This is your backup in case anything goes wrong.
    # 2). A folder where your new experiment will take place. Rename the folder to your desired experiment name and conduct your testing.

#For this code, the following is required:
    # 1). A dataset of an input and output length that is different from your model. For example, a dataset with 10 input frames and 90 output frames. 
    # NOTE: The input frames must be smaller than or equal to what the model expects (5 data input <= 10 expected input).
    #       The output frames must be bigger than or equal what the model expects OR equal to what the model expects (95 data output > 90 expected output).

    # This next requirement is only necessary for inferring of a pre-trained model. Make sure the exact same model architecture is defined here.
    # 2). A folder including the weights of your desired model for testing. For example, a model trained for taking in 10 frames and predicting 90 frames. This
    # should be the 2nd copy that you create.

class CustomDataset(Dataset):
    def __init__(self, X, Y, normalize=False):
        super(CustomDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.mean = None
        self.std = None

        if normalize:
            # get the mean/std values along the channel dimension
            mean = data.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            std = data.std(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            data = (data - mean) / std
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.X)
        #return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels

def transform_sequences(X, Y, new_input_length=10, new_output_length=90): #Transforms a sequence of inputs and outputs into the intended model lengths.
    num_samples, orig_input_length, C, H, W = X.shape
    _, orig_output_length, _, _, _ = Y.shape

    # Pad X with zeros at the beginning of the sequences
    pad_length = new_input_length - orig_input_length
    if pad_length > 0:
        padding = np.zeros((num_samples, pad_length, C, H, W), dtype=X.dtype)
        X_transformed = np.concatenate((padding, X), axis=1)
    elif pad_length < 0:
        raise ValueError("Invalid input length: input length is larger than the expected value defined in the model.")
    else:
        X_transformed = X  # No padding needed

    # Trim Y to match new output length
    trim = new_output_length - orig_output_length
    if trim < 0:
        Y_transformed = Y[:, :new_output_length]
    elif trim > 0:
        raise ValueError("Invalid output length: ouput length is smaller than the expected value defined in the model.")
    else:
        Y_transformed = Y

    return X_transformed, Y_transformed

# Insert data here.
dataset = np.load('data/spin_data/train.npz')
dataval = np.load('data/spin_data/valid.npz')
testset = np.load('data/spin_data/test_50sequences.npz')

pre_seq_length = 10 # Input length the model expects for training
aft_seq_length = 90 # Output length the model expects for training
batch_size = 1  # Test with 1 clip per batch, or more if so desired.


X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset['X_train'], dataval['X_val'], testset['X_test'], dataset['Y_train'], dataval['Y_val'], testset['Y_test']

# Transform sequences. Be sure to chane the desired new input length and new output length according to what your model expects to recieve (If you put in a sequence of 5-95, this will transform it into 10-90. This function only works if we need a bigger input, and a smaller output).
X_train, Y_train = transform_sequences(X_train, Y_train, new_input_length=pre_seq_length, new_output_length=aft_seq_length)
X_val, Y_val = transform_sequences(X_val, Y_val, new_input_length=pre_seq_length, new_output_length=aft_seq_length)
X_test, Y_test = transform_sequences(X_test, Y_test, new_input_length=pre_seq_length, new_output_length=aft_seq_length)

print(f'X_shape:{X_test.shape}')
print(f'Y_shape:{Y_test.shape}')

train_set = CustomDataset(X=X_train, Y=Y_train)
val_set = CustomDataset(X=X_val, Y=Y_val)
test_set = CustomDataset(X=X_test, Y=Y_test)

dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

custom_training_config = {
    'pre_seq_length': pre_seq_length,
    'aft_seq_length': aft_seq_length,
    'total_length': pre_seq_length + aft_seq_length,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 10,
    'lr': 0.001,   
    'metrics': ['mse', 'mae'],

    'ex_name':  'spin_simvp_output256x256',
    'dataname': 'spinodaldata',
    'in_shape': [pre_seq_length, 1, 64, 64],
}

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'method': 'SimVP',
    # Users can either using a config file or directly set these hyperparameters 
    #'config_file': 'configs/taxibj/E3DLSTM.py',
    
    # Here, we directly set these parameters
    # model
    #'pre_seq_length': 10,
    #'aft_seq_length': 90,
    'total_length': pre_seq_length + aft_seq_length,
    'model_type': 'gSTA',
    'N_S': 4,
    'N_T': 8,
    'hid_S': 64,
    'hid_T': 256
}

args = create_parser().parse_args([])
config = args.__dict__

# update the training config
config.update(custom_training_config)
# update the model config
config.update(custom_model_config)
# fulfill with default values
default_values = default_parser()
# Here are some direct commands to let the network resume training.

# default_values['resume_from'] = "work_dirs/spin_simvp_output256x256_exp/checkpoints/latest.pth"
# config['test'] = True
# config['inference'] = True

for attribute in default_values.keys():
    if config[attribute] is None:
        config[attribute] = default_values[attribute]

exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test))

print('>'*35 + ' training ' + '<'*35)
exp.train()

print('>'*35 + ' testing  ' + '<'*35)
exp.test()