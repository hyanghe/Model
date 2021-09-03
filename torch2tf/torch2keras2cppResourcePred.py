# Keras version 2.2.4-tf  
# Tensorflow version 2.1.0-gpu
# Issue with importing keras: from tensorflow.keras import layers, optimizers, datasets-->
#                             AttributeError: module 'tensorflow' has no attribute 'keras'
# solved by: pip3 install --force-reinstall tensorflow-gpu=2.1.0
# https://www.cxyzjd.com/article/JohnJim0/105937006                                                                                            

#####################################################################
###############       Construct torch model     #####################
#####################################################################
from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import logging
import argparse
from torchsummary import summary


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Backbone(nn.Module):
    def __init__(self, input_size, opt):
        super(Backbone, self).__init__()
        self.activation = opt.activation
        self.input_size = input_size
        self.hidden = opt.hidden

        net = []
        for IdxLayer, nHidden in enumerate(self.hidden): # hidden is like [512, 512, 512]
            if IdxLayer == 0:
                net.append(nn.Linear(self.input_size, nHidden))
            else:
                net.append(nn.Linear(self.previous_hidden, nHidden))

            if opt.activation == "relu":
                net.append(nn.ReLU())

            net.append(nn.BatchNorm1d(nHidden))
            self.previous_hidden = nHidden

        self.net = nn.Sequential(*net)

    def forward(self, x):
        output = self.net(x)
        return output


class NN_cls(nn.Module):
    def __init__(self, input_size, cls_size, opt):
        super(NN_cls, self).__init__()
        self.input_size = input_size
        self.cls_size = cls_size
        self.opt = opt

        self.backbone = Backbone(self.input_size, self.opt)
        self.linear = nn.Linear(opt.hidden[-1], self.cls_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        rep = self.backbone(x)
        output = self.linear(rep)
        logits = self.softmax(output)
        # logits = output
        return logits

input_size = 15
cls_size = 5


def config():
    """build configruation dictionary"""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--activation', dest='activation', default="relu", help='the activate function of model')
    parser.add_argument('--hidden', dest='hidden', type=int, nargs='+', default=None, help='# of middle layers')
    args, unknown = parser.parse_known_args() # only for jupyter notebook
    # args = parser.parse_args() 
    return args
opt = config()
# opt.hidden = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
opt.hidden = [512] * 10

torch_model = NN_cls(input_size, cls_size, opt)
torch_model.to(device)
summary(torch_model, (512, input_size))


# #####################################################################
# ###############       Construct keras model     #####################
# #####################################################################

# # from __future__ import print_function
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import Sequential, Model, optimizers, activations
# from tensorflow.keras.layers import Input, Dense, ReLU, BatchNormalization, Softmax, Lambda
# import argparse
# import numpy as np

# def config():
#     """build configruation dictionary"""
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--activation', dest='activation', default="relu", help='the activate function of model')
#     parser.add_argument('--hidden', dest='hidden', type=int, nargs='+', default=None, help='# of middle layers')
#     args, unknown = parser.parse_known_args() # only for jupyter notebook
#     # args = parser.parse_args() 
#     return args

# opt = config()
# # opt.hidden = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
# opt.hidden = [512] * 10
# input_size, cls_size = 15, 5
# net = []

# for IdxLayer, nHidden in enumerate(opt.hidden): # hidden is like [512, 512, 512]
#     net.append(Dense(nHidden))
#     if opt.activation == "relu":
#         net.append(ReLU())
#     net.append(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-05))
#     # net.append(BatchNormalization(momentum=0.1, epsilon=1e-05))
#     # net.append(BatchNormalization())

# net.append(Dense(cls_size))
# net.append(Softmax(axis=-1))
# net.append(Lambda(lambda x: tf.math.log(x)))
# model_tf = Sequential(net)
# input_data = np.random.rand(512,input_size)
# model_tf(input_data)
# model_tf.summary()


###############################################################################################################
###############       Construct keras model compatible with keras2cpp   #######################################
###############################################################################################################


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model, optimizers, activations
from tensorflow.keras.layers import Input, Dense, ReLU, BatchNormalization, Softmax, Lambda
import argparse
import numpy as np

def config():
    """build configruation dictionary"""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--activation', dest='activation', default="relu", help='the activate function of model')
    parser.add_argument('--hidden', dest='hidden', type=int, nargs='+', default=None, help='# of middle layers')
    args, unknown = parser.parse_known_args() # only for jupyter notebook
    # args = parser.parse_args() 
    return args

opt = config()
# opt.hidden = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
opt.hidden = [512] * 10
input_size, cls_size = 15, 5
net = []

for IdxLayer, nHidden in enumerate(opt.hidden): # hidden is like [512, 512, 512]
    net.append(Dense(nHidden, activation='relu'))
    # if opt.activation == "relu":
    #     net.append(ReLU())
    net.append(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-05))
    # net.append(BatchNormalization(momentum=0.1, epsilon=1e-05))
    # net.append(BatchNormalization())

net.append(Dense(cls_size, activation='softmax'))
# net.append(Softmax(axis=-1))
# net.append(Lambda(lambda x: tf.math.log(x)))
model_tf = Sequential(net)
input_data = np.random.rand(512,input_size)
model_tf(input_data)
model_tf.summary()

# #save model
from keras2cpp import export_model
export_model(model_tf, 'example.model')
print('Save keras2cpp successfully')


#####################################################################
###############       Load trained torch model     ##################
#####################################################################
import torch
import os 
cwd = os.getcwd()
data_ckpt_dir = os.path.dirname(cwd) + '\\data_ckpt\\'
torch_model_weights_dict = torch.load(data_ckpt_dir + 'best_model.pth',  map_location=torch.device('cpu'))['model']
len(torch_model_weights_dict.keys())
tf_layers = model_tf.layers

#################################################################################
############### Modify the state_dict manually (if needed)     ##################
#################################################################################

# for key in list(torch_model_weights_dict.keys()):
#   if key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
#     # print('bn')
#     torch_model_weights_dict[key] = torch.tensor(np.random.uniform(low=0, high=1.,\
#                   size = torch_model_weights_dict[key].shape), dtype=torch.float32)
#   else:
#     torch_model_weights_dict[key] = torch.tensor(np.random.uniform(low=-0.1, high=0.1,\
#                   size = torch_model_weights_dict[key].shape), dtype=torch.float32)
#     # torch_model_weights_dict[key] = torch.tensor(np.zeros(torch_model_weights_dict[key].shape), dtype=torch.float32)

# torch_model.load_state_dict(torch_model_weights_dict, strict=False)




# #############################################################################################################
# ############################ Assign torch weights to keras compatible with keras2cpp     ####################
# #############################################################################################################
tf_layers = model_tf.layers
unit_size_torch = 7
unit_size_tf = 2

for i, j in zip(range(len(torch_model_weights_dict)//unit_size_torch), range(len(tf_layers)//unit_size_tf)):
# for i, j in zip(range(len(torch_model_weights_dict)//unit_size_torch), range((len(tf_layers) - 1)//unit_size_tf)):
  dense_w_s, dense_b_s, w_s, b_s, running_mean_s, running_var_s, _ =\
   list(torch_model_weights_dict.keys())[i*unit_size_torch:(i+1)*unit_size_torch]
  dense_w = torch_model_weights_dict[dense_w_s]
  dense_b = torch_model_weights_dict[dense_b_s]
  w = torch_model_weights_dict[w_s]
  b = torch_model_weights_dict[b_s]
  running_mean = torch_model_weights_dict[running_mean_s]
  running_var = torch_model_weights_dict[running_var_s]
  # print('j is: ', j)
  dense_tf, norm_tf = tf_layers[j*unit_size_tf:(j+1)*unit_size_tf]
  print('dense_tf.weights[0] shape is: ', dense_tf.weights[0].shape)
  print('dense_w shape is: ', dense_w.shape)
  dense_tf.set_weights([dense_w.t(), dense_b])
  # dense_tf.weights[0] = dense_w # Dense Weights
  # dense_tf.weights[1] = dense_b # Dense Bias
  # norm_tf.weights[0] = w # gamma
  # norm_tf.weights[1] = b # beta
  # norm_tf.weights[2] = running_mean # moving_mean
  # norm_tf.weights[3] = running_var # moving_variance

  ## Not sure whether the weights for batch norm need to be transposed
  norm_tf.set_weights([w.t(), b, running_mean, running_var])
  ## Try not transposing the batch norm weight
  # norm_tf.set_weights([w, b, running_mean, running_var])

print(i)
last_layer_torch_w_s = list(torch_model_weights_dict.keys())[-2]
last_layer_torch_w = torch_model_weights_dict[last_layer_torch_w_s]
last_layer_torch_b_s = list(torch_model_weights_dict.keys())[-1]
last_layer_torch_b = torch_model_weights_dict[last_layer_torch_b_s]

last_layer_tf = tf_layers[-1]
# last_layer_tf = tf_layers[-2]
last_layer_tf.set_weights([last_layer_torch_w.t(), last_layer_torch_b])


## Example test
import numpy as np
# random_input = np.random.random((1, 15)).astype('f')
random_input = np.random.uniform(low = 0.0, high = 1.0, size = (1, 15)).astype('f')
# random_input = np.ones((1, 15)).astype('f')

tf_out = model_tf(random_input)
# torch_model.load_state_dict(torch.load('best_model.pth')['model'])
torch_model.load_state_dict(torch.load(data_ckpt_dir + 'best_model.pth', map_location=torch.device('cpu'))['model'])
torch_model.eval()
# torch_model = torch.load('best_model.pth')
torch_input = torch.tensor(random_input).to(device)
torch_out = torch_model(torch_input)

################################################################################################################
########################## Need to add this line to compensate for the Lambda layer#############################
################################################################################################################
print(tf.math.log(tf_out))
################################################################################################################
########################## Need to add this line to compensate for the Lambda layer#############################
################################################################################################################
print(torch_out)


################################################################################################################
########################## Save to (keras2cpp sequential) model ################################################
################################################################################################################
from keras2cpp import export_model
print(model_tf.summary())
export_model(model_tf, 'example.model')




#################################################################################
############################ Test using real data     ###########################
#################################################################################


# import pickle
# with (open(data_ckpt_dir + "time_data_np.pickle", "rb")) as f:
#   test_file = pickle.load(f)['test']
#   test_data = test_file[0]
#   test_label = test_file[1]
#   torch_pred = torch_model(torch.tensor(test_data, dtype=torch.float32).to(device))
#   tf_pred = model_tf(test_data)

# print('Check whether the two models are predicting the same for real data:')
# torch_pred_np = torch.argmax(torch_pred, -1).detach().cpu().numpy()
# tf_pred_np = tf.argmax(tf_pred, -1).numpy()
# num_consistent = np.sum(torch_pred_np == tf_pred_np)
# num_total = tf_pred_np.shape[0]
# print(f'num_consistent is {num_consistent}, num_total is {num_total}')

import pickle
with (open(data_ckpt_dir + "time_data_np.pickle", "rb")) as f:
  test_file = pickle.load(f)['test']
  test_data = test_file[0]
  test_label = test_file[1]
  torch_pred = torch_model(torch.tensor(test_data, dtype=torch.float32).to(device))
  tf_pred = model_tf(test_data)
################################################################################################################
########################## Need to add this line to compensate for the Lambda layer#############################
################################################################################################################
  tf_pred = tf.math.log(tf_pred)
################################################################################################################
########################## Need to add this line to compensate for the Lambda layer#############################
################################################################################################################


for i in range(10):
  print(np.asarray(torch_pred[i].detach().cpu()))
  print(np.asarray(tf_pred[i]))