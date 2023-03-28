#!/bin/bash

mode="training"
variable="solution"
gpu=4
load_model=False
process_data=False
# plot_solution=True

# python ./generator_train_sln.py -m ${mode} --variable ${variable} --gpu ${gpu} --load_model ${load_model} --plot_solution ${plot_solution}
# python ./flux_conservation_train.py -m ${mode} --gpu ${gpu} --load_model ${load_model}
                                 
# python ./generator_train.py -m ${mode} -v ${variable} --gpu ${gpu} --load_model ${load_model} --plot_solution ${plot_solution}

python ./generator_train.py -m ${mode} -v ${variable} --gpu ${gpu} --load_model ${load_model} --process_data ${process_data}