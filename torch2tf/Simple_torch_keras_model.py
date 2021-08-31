# from __future__ import print_function
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

############################################################################
## Construct a torch model
############################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_hidden = 1
num_neuron = 8
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

    def forward(self, x):
        rep = self.backbone(x)
        output = self.linear(rep)
        logits = self.softmax(output)
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
opt.hidden = [num_neuron] * num_hidden


# torch_model = torch.nn.Sequential(
#     Backbone(input_size, opt),
#     nn.Linear(opt.hidden[-1], cls_size),
#     nn.LogSoftmax(dim=-1))

# with torch.no_grad():
#     print(torch_model[0])
    # torch_model[0].weight = nn.Parameter(torch.ones_like(model[0].weight))
    # model[0].weight[0, 0] = 2.
    # model[0].weight.fill_(3.)

torch_model = NN_cls(input_size, cls_size, opt)
torch_model.to(device)
summary(torch_model, (num_neuron, input_size))



############################################################################
## Construct a keras model
############################################################################
# from __future__ import print_function
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

# opt = config()
# opt.hidden = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
# opt.hidden = [512] * 10
input_size, cls_size = 15, 5
net = []

for IdxLayer, nHidden in enumerate(opt.hidden): # hidden is like [512, 512, 512]
    net.append(Dense(nHidden))
    if opt.activation == "relu":
        net.append(ReLU())
    net.append(BatchNormalization())

net.append(Dense(cls_size))
net.append(Softmax(axis=-1))
net.append(Lambda(lambda x: tf.math.log(x)))
model_tf = Sequential(net)
input_data = np.random.rand(num_neuron,input_size)
model_tf(input_data)
model_tf.summary()



############################################################################
## Assign the same weights to torch and keras models: 

# keras' set_weight
# https://stackoverflow.com/questions/47183159/how-to-set-weights-in-keras-with-a-numpy-array

# Using torch's load_state_dict 
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
############################################################################

# torch_layer_1_w = np.ones((input_size, num_neuron))
# torch_layer_1_b = np.ones((num_neuron))
# torch_bn_w = np.ones((num_neuron))
# torch_bn_b = np.ones((num_neuron))
# torch_bn_mean = np.ones((num_neuron))
# torch_bn_var = np.ones((num_neuron))
torch_layer_2_w = np.ones((num_neuron, cls_size))
# torch_layer_2_b = np.ones((cls_size))

torch_layer_1_w = np.random.uniform(low = 0.0, high = 1.0, size = (input_size, num_neuron))
torch_layer_1_b = np.random.uniform(low = 0.0, high = 1.0, size = (num_neuron))
torch_bn_w = np.random.uniform(low = 0.0, high = 1.0, size = (num_neuron))
torch_bn_b = np.random.uniform(low = 0.0, high = 1.0, size = (num_neuron))
torch_bn_mean = np.random.uniform(low = 0.0, high = 1.0, size = (num_neuron))
torch_bn_var = np.random.uniform(low = 0.0, high = 1.0, size = (num_neuron))
# torch_layer_2_w = np.random.uniform(low = 0.0, high = 1.0, size = (num_neuron, cls_size))
torch_layer_2_b = np.random.uniform(low = 0.0, high = 1.0, size = (cls_size))

## Assign weights to keras layers
tf_layer_1 = model_tf.layers[0]
tf_norm = model_tf.layers[2]
tf_layer_2 = model_tf.layers[3]
tf_layer_1.set_weights([torch_layer_1_w, torch_layer_1_b])
tf_norm.set_weights([torch_bn_w, torch_bn_b, torch_bn_mean, torch_bn_var])
tf_layer_2.set_weights([torch_layer_2_w, torch_layer_2_b])

model_tf.layers

## Assign weights to torch layers

torch_layers = list(torch_model.state_dict().keys())
from collections import OrderedDict

new_dict = {
    torch_layers[0]: torch.tensor(torch_layer_1_w.transpose(), dtype=torch.float32).to(device),
    torch_layers[1]: torch.tensor(torch_layer_1_b, dtype=torch.float32).to(device),
    torch_layers[2]: torch.tensor(torch_bn_w, dtype=torch.float32).to(device),
    torch_layers[3]: torch.tensor(torch_bn_b, dtype=torch.float32).to(device),
    torch_layers[4]:  torch.tensor(torch_bn_mean, dtype=torch.float32).to(device),
    torch_layers[5]:  torch.tensor(torch_bn_var, dtype=torch.float32).to(device),
    # torch_layers[6]:  torch.tensor([], dtype=torch.float32).to(device),  # Extra parameter for BN1d
    torch_layers[7]:  torch.tensor(torch_layer_2_w.transpose(), dtype=torch.float32).to(device),
    torch_layers[8]:  torch.tensor(torch_layer_2_b, dtype=torch.float32).to(device),
}

new_dict = OrderedDict(new_dict)
torch_model.load_state_dict(new_dict, strict=False)

# Create trivial input for testing the two models
input_test = np.arange(input_size).reshape(1, -1)
input_test.shape
torch_model.eval()
torch_output, tf_output = torch_model(torch.tensor(input_test, dtype=torch.float32).to(device)), model_tf(input_test)
print('torch output is: ', torch_output)
print('tf output is: ', tf_output)