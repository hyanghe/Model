#!/bin/bash

mode="test"
# variable="condition"
gpu=1
# load='./checkpoints/checkpoint.pth'
# load = False
# plot_solution=True
batch_size=100
learning_rate=0.0003
epochs=5000

python ./U.py -m ${mode} --gpu ${gpu} -b ${batch_size} -l ${learning_rate} -e ${epochs}

                                 
