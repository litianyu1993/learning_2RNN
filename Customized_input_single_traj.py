import numpy as np
import learning
import synthetic_data
import pickle
import os
import sys
from shutil import copyfile
import time
import matplotlib.pyplot as plt
import argparse
import pickle
import torch
import shlex

def get_data_from_file(train_file_path, test_file_path, normalize = False):
    train_data = np.genfromtxt(train_file_path, delimiter=',')
    if train_data.ndim == 1:
        feature_dim = 1
    else:
        feature_dim = train_data.shape[1]
    train_data = train_data.reshape(-1, feature_dim)
    test_data = np.genfromtxt(test_file_path, delimiter=',')
    test_data  = test_data.reshape(-1, feature_dim)
    if normalize == True:
        mean = np.mean(train_data, axis = 0)
        std = np.std(train_data, axis = 0)
        train_data = (train_data - mean)/std
        test_data = (test_data - mean)/std
        return train_data, test_data, mean, std
    else:
        mean = np.zeros(feature_dim)
        std = np.ones(feature_dim)
        return train_data, test_data, mean, std
def train_vali_split(train_data, split_ratio):
    total_length = len(train_data)
    vali_data = train_data[:int(total_length*split_ratio)]
    new_train_data = train_data[int(total_length*split_ratio):]
    return new_train_data, vali_data

def add_time_dimension_base_on_length(data, length):
    X = []
    Y = []
    for i in range(len(data) - length - 1):
        temp = []
        for j in range(length):
            temp_data = np.asarray(data[i+j])
            temp.append(np.insert(temp_data, 0, 1.))
        temp = np.asarray(temp)
        X.append(temp)
        Y.append(np.asarray(data[i + length]))

    return np.asarray(X), np.asarray(Y)

def cal_RMSE(pred, ytest, mean_data, std_data):
    pred = (pred * std_data) + mean_data
    ytest = (ytest * std_data) + mean_data
    print(pred[0:5])
    print(ytest[0:5])
    return np.sqrt(np.mean((pred - ytest) ** 2))


def cal_MAPE(pred, ytest, mean_data, std_data):
    pred = (pred * std_data) + mean_data
    ytest = (ytest * std_data) + mean_data
    return np.mean(np.abs(pred - ytest) / ytest)


def cal_MAE(pred, ytest, mean_data, std_data):
    pred = (pred * std_data) + mean_data
    ytest = (ytest * std_data) + mean_data
    return np.mean(np.abs(pred - ytest))

normalize = False
parser = argparse.ArgumentParser()
parser.add_argument('-nr', '--number_runs', help='number of runs', type=int, default= 1)
parser.add_argument('-le', '--length', help='minimum training length', type=int, default=2)
parser.add_argument('-tle', '--testing_length', help='testing length', type=int, default=6)
parser.add_argument('-lm', '--method_list', nargs='+', help="List of methods to use", default=['TIHT', 'TIHT+SGD'])
parser.add_argument('-eps', '--HT_epsilon', help='epsilon for TIHT and IHT', type=float, default=1e-20)
parser.add_argument('-lr', '--HT_learning_rate', help='learning rate for TIHT and IHT', type=float, default=1e-1)
parser.add_argument('-mi', '--HT_max_iter', help='number of max iterations for TIHT and IHT', type=int, default=10000)

parser.add_argument('-xp', '--xp_path', help='experiment folder path', default='./')
parser.add_argument('-lr2', help='learning rate for sgd 2rnn', type=float, default=0.001)
parser.add_argument('-epo2', help='number of epochs for sgd 2rnn', type=int, default=50)
parser.add_argument('-b2', '--batch_size', help='batch size for sgd 2rnn', type=int, default=1000)
parser.add_argument('-t', '--tolerance', help='tolerance for sgd 2rnn', type=int, default=50)

parser.add_argument('-ns', '--states_number', help='number of states for the model', type=int, default=5)
parser.add_argument('-norm', '--normalize', help = 'if normalize the data', action = 'store_true')
parser.add_argument('-trf', '--train_file', help = 'file path for training data', type = str, default='train_data.csv')
parser.add_argument('-tef', '--test_file', help = 'file path for testing data', type = str, default='test_data.csv')
parser.add_argument('-tvs', '--train_validation_split', help = 'split ratio for training and validation', type = float, default=0.2)
args = parser.parse_args()

if args.states_number:
    num_states = args.states_number
if args.number_runs:
    N_runs = args.number_runs
if args.length:
    length = args.length
if args.testing_length:
    test_length = args.testing_length
if args.method_list:
    methods = args.method_list
if args.HT_epsilon:
    TIHT_epsilon = args.HT_epsilon
if args.HT_learning_rate:
    TIHT_learning_rate = args.HT_learning_rate
if args.HT_max_iter:
    TIHT_max_iters = args.HT_max_iter
if args.xp_path:
    xp_path = args.xp_path
if args.lr2:
    lr2 = args.lr2
if args.epo2:
    epo2 = args.epo2
if args.batch_size:
    b2 = args.batch_size
if args.tolerance:
    tol = args.tolerance
if args.normalize:
    normalize = args.normalize
if args.train_file:
    train_file = args.train_file
if args.test_file:
    test_file = args.test_file
if args.train_validation_split:
    split_ratio = args.train_validation_split
if not os.path.exists(xp_path):
    os.makedirs(xp_path)


train_data, test_data, mean_data, std_data = get_data_from_file(train_file, test_file, normalize)
train_data, vali_data = train_vali_split(train_data, split_ratio)
data_function_train = lambda l: add_time_dimension_base_on_length(train_data, l)
data_function_vali = lambda l: add_time_dimension_base_on_length(vali_data, l)
data_function_test = lambda l: add_time_dimension_base_on_length(test_data, l)

X = []
Y = []
for i in range(length*2+2):
   tempx, tempy = data_function_train(i)
   X.append(tempx)
   Y.append(tempy)
Xtest, ytest= data_function_test(test_length)
Xvali, yvali = data_function_vali(test_length)
verbose = 0
for run in range(N_runs):
    print("test MSE of zero function", np.mean(ytest ** 2))
    print('\n\n', '*' * 80, '\nrun', run)
    ph = 1
    for k in range(len(methods)):
        method = methods[k]
        if method == 'TIHT+SGD':
            model = learning.TIHT_SGD_torch(X, Y, num_states, length, verbose, TIHT_epsilon, TIHT_learning_rate, TIHT_max_iters,
                                            lr2, epo2, b2, tol, Xtest, ytest, alpha = 1., lifting = True)

            Xtest_temp = torch.from_numpy(Xtest).float()
            pred = model(Xtest_temp)
            pred_numpy = pred.detach().numpy().reshape(-1, )

            pred_numpy = pred_numpy.reshape(ytest.shape)
            rmse1 = cal_RMSE(pred_numpy, ytest, mean_data, std_data)
            mape1 = cal_MAPE(pred_numpy, ytest, mean_data, std_data)
            mae1 = cal_MAE(pred_numpy, ytest, mean_data, std_data)

        else:
            X_vec, y_vec = [], []
            for i in range(0, len(X)):
                X_vec.append(learning.sequence_to_tensor(X[i]))
                y_vec.append(Y[i])
            TL_vec = learning.get_all_TLs(X_vec, y_vec, rank=num_states, eps=TIHT_epsilon,
                                          learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                          method=method, verbose=0, alpha_ini_value=1.)

            Hl = learning.approximate_hankel_l(TL_vec=TL_vec, length=length)
            Hl_plus = learning.approximate_hankel_plus(TL_vec=TL_vec, length=length)
            Hl_minus = learning.approximate_hankel_minus(TL_vec=TL_vec, length=length)
            model = learning.spectral_learning_multiple_length(num_states, Hl_minus, Hl, Hl_plus)

            pred = []
            for i in range(len(Xtest)):
                pred.append(model.predict(Xtest[i]))
            pred = np.asarray(pred).reshape(ytest.shape)

            rmse1 = cal_RMSE(pred, ytest, mean_data, std_data)
            mape1 = cal_MAPE(pred, ytest, mean_data, std_data)
            mae1 = cal_MAE(pred, ytest, mean_data, std_data)

        print('RMSE:', rmse1)
        print('MAPE:', mape1)
        print('MAE:', mae1)






