import tensorflow as tf
#tf.enable_eager_execution()
import os
import sys
import argparse
num_examples = [50, 100, 200, 400, 800, 1600, 3200]

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--Experiment', help = 'n mame of the experiment, available: Addition, RandomRNN, Wind', default='Wind')
parser.add_argument('-var', '--noise', help = 'variance of the guassian noise', default= 0.1)
parser.add_argument('-nr', '--number_runs', help = 'number of runs you want to excecute', default=1)
#parser.add_argument('-m','--methods', help = 'methods you want to run', nargs='+', default=['TIHT', 'TIHT+SGD'])
args = parser.parse_args()
if args.Experiment != None:
    Experiment = args.Experiment
else:
    raise Exception('Did not initialize Experiment, try set up after -e argument')

if args.noise != None:
    noise = args.noise

if args.number_runs != None:
    N_runs = args.number_runs

#methods = args.methods

assert Experiment in ['Wind', 'RandomRNN', 'Addition'], 'Invalid name for experiments, can only be Wind, RandomRNN and Addition'

if Experiment == 'Wind':
    os.system(
        "python Wind_EXP.py -lm TIHT+SGD TIHT -ns 10 -mi 10000 -epo2 2000 -nr " + str(N_runs)
    )
else:
    methods = 'LSTM TIHT TIHT+SGD'
    for num_example in num_examples:
        if Experiment == 'Addition':
            num_states = 2
            os.system("python Addition_EXP.py -var "+str(noise) + " -ns "+str(num_states) + " -a 1 -nr " +str(N_runs) +" -ld -lm "+ methods)
            num_states = 4
            os.system(
                "python Addition_EXP.py -var " + str(noise) + " -ns " + str(num_states) + " -a 1 -nr " +str(N_runs) +" -ld -lm " + methods)
        elif Experiment == 'RandomRNN':
            num_states = 5
            os.system(
                "python RandomRNN_EXP.py -var " + str(noise) + " -ns " + str(num_states) + " -a 1 -nr " +str(N_runs) +" -ld -lm " + methods)
            num_states = 6
            os.system(
                "python RandomRNN_EXP.py -var " + str(noise) + " -ns " + str(num_states) + " -a 1 -nr " +str(N_runs) +" -ld -lm " + methods)



