#!/bin/bash

mode="train"
# variable="condition"
gpu=1
# load='./checkpoints/checkpoint.pth'
# load = False
# plot_solution=True
batch_size=256
learning_rate=0.00003
epochs=5000

python ./U.py -m ${mode} --gpu ${gpu} -b ${batch_size} -l ${learning_rate} -e ${epochs}

                                 
