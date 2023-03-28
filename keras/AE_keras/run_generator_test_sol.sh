#!/bin/bash

mode="testing"
variable="solution"
gpu=5
load_model=True
# plot_solution=True
process_data=False
# python ./generator_train_sln.py -m ${mode} --variable ${variable} --gpu ${gpu} --load_model ${load_model} --plot_solution ${plot_solution}
# python ./flux_conservation_train.py -m ${mode} --gpu ${gpu} --load_model ${load_model}
                                 
python ./generator_train.py -m testing -v solution --gpu ${gpu} --load_model ${load_model}  --process_data ${process_data}