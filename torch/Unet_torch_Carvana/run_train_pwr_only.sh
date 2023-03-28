#!/bin/bash

mode="train"
# variable="condition"
gpu=6
load='./checkpoints/checkpoint.pth'
# batch_size=128
batch_size=32

# plot_solution=True

# python ./generator_train_sln.py -m ${mode} --variable ${variable} --gpu ${gpu} --load_model ${load_model} --plot_solution ${plot_solution}
python ./train_pwr_org_data.py -m ${mode} --gpu ${gpu} --load ${load} -b ${batch_size}
# python ./train_pwr_org_data.py -m ${mode} --gpu ${gpu} -b ${batch_size}

                                 
