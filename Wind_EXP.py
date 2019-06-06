import numpy as np
import tensorflow as tf
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

def tic():
    return time.clock()

def toc(t):
    return time.clock() - t

def generate_wind_speed(train_file_path, test_file_path, mean_window_size = 12):
    data1 = np.genfromtxt(train_file_path, max_rows=111659, delimiter=',')
    data1 = data1[:, 13]
    data2 = np.genfromtxt(train_file_path, skip_header=111659, delimiter=',')
    data2 = data2[:, 13]
    data3 = np.genfromtxt(test_file_path, delimiter=',')
    data3 = data3[:, 13]
    data = np.insert(data1, len(data1), data2)
    data = np.insert(data, len(data), data3)
    train_test_split = (len(data1) + len(data2)) / len(data)
    nan_count = 0
    for i in range(len(data)):
        if type(data[i]) is str:
            data[i] = float(data[i])
        if type(data[i]) is str:
            print(data[i])
        if np.isnan(data[i]):
            nan_count += 1
            temp_sum = []
            for j in range(1, 6):
                if not np.isnan(data[i - j]):
                    temp_sum.append(data[i - j])
                    break
            for j in range(1, 6):
                if not np.isnan(data[i + j]):
                    temp_sum.append(data[i + j])
                    break
            data[i] = np.mean(np.asarray(temp_sum))

    temp_data = []
    for i in range(int((len(data) - mean_window_size) / mean_window_size)):
        temp_data.append(np.mean(data[i * mean_window_size:(i * mean_window_size + mean_window_size)]))
    data = temp_data
    return data, train_test_split

def generate_wind_speed_preprocess(data, length):
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

def generate_wind_train(data, train_test_split, length):
    X, Y= generate_wind_speed_preprocess(data, length)
    train_X = X[:int(train_test_split * len(X))]
    train_Y = Y[:int(train_test_split * len(X))]
    return train_X, train_Y

def generate_wind_test(data, train_test_split, length):
    X, Y= generate_wind_speed_preprocess(data, length)
    test_X = X[int(train_test_split * len(X)):]
    test_Y = Y[int(train_test_split * len(X)):]
    return test_X, test_Y


def pred_k_more(model, test_X, test_Y, pred, ph, if_tc = False, if_tf = False):
    if ph <=0:return pred, test_Y

    for j in range(ph):
        temp_test_x = np.zeros((test_X.shape[0], test_X.shape[1]+1, test_X.shape[2]))

        for i in range(test_X.shape[0]):
            temp_test_x[i] = np.insert(test_X[i], test_X[i].shape[0]*test_X[i].shape[1],
                                       np.asarray([ 1., pred[i]])).reshape(test_X.shape[1]+1, test_X.shape[2])
            test_X[i] = temp_test_x[i][1:]
        pred2 = []
        if if_tc:
            Xtest_temp = torch.from_numpy(test_X).float()
            pred2 = model(Xtest_temp).detach().numpy()
        for i in range(len(test_X)):
            if if_tf == False and if_tc == False:
                pred2.append(model.predict(test_X[i]))
            elif if_tf == True:
                Xtest_temp = tf.convert_to_tensor(test_X[i], np.float32)
                pred2.append(model.predict(Xtest_temp))
        pred = np.asarray(pred2)
    return pred[:-(ph)], test_Y[(ph):]


data, train_test_split = generate_wind_speed('./Data/Wind_Speed/train.csv', './Data/Wind_Speed/test.csv')
mean_data = np.mean(data)
std_data = np.std(data)
data = (data-mean_data)/std_data
data_function_train = lambda l: generate_wind_train(data, train_test_split, l)
data_function_test = lambda l: generate_wind_test(data, train_test_split, l)
N_runs = 1
length = 3
test_length = 4
methods = ['TIHT', 'TIHT+SGD']
TIHT_epsilon = 1e-20
TIHT_learning_rate = 1e-1
TIHT_max_iters = 50000
xp_path = './SP_Wind/'


parser = argparse.ArgumentParser()
parser.add_argument('-nr', '--number_runs', help='number of runs', type=int)
parser.add_argument('-le', '--length', help='minimum training length', type=int)
parser.add_argument('-tle', '--testing_length', help='testing length', type=int)
parser.add_argument('-lm', '--method_list', nargs='+', help="List of methods to use")
parser.add_argument('-eps', '--HT_epsilon', help='epsilon for TIHT and IHT', type=float)
parser.add_argument('-lr', '--HT_learning_rate', help='learning rate for TIHT and IHT', type=float)
parser.add_argument('-mi', '--HT_max_iter', help='number of max iterations for TIHT and IHT', type=int)

parser.add_argument('-xp', '--xp_path', help='experiment folder path')
parser.add_argument('-lr2', help = 'learning rate for sgd 2rnn', type = float, default=0.001)
parser.add_argument('-epo2', help = 'number of epochs for sgd 2rnn', type = int, default=50)
parser.add_argument('-b2', '--batch_size', help = 'batch size for sgd 2rnn', type = int, default=1000)
parser.add_argument('-t', '--tolerance', help = 'tolerance for sgd 2rnn', type = int, default=50)


parser.add_argument('-ns', '--states_number', help='number of states for the model', type=int)
args = parser.parse_args()

if args.states_number != None:
    num_states = args.states_number
else:
    raise Exception('Did not initialize state numbers, try set up after -ns argument')


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

if not os.path.exists(xp_path):
    os.makedirs(xp_path)


def cal_RMSE(pred, ytest, mean_data, std_data):
    pred = (pred * std_data) + mean_data
    ytest = (ytest * std_data) + mean_data
    return np.sqrt(np.mean((pred - ytest) ** 2))


def cal_MAPE(pred, ytest, mean_data, std_data):
    pred = (pred * std_data) + mean_data
    ytest = (ytest * std_data) + mean_data
    return np.mean(np.abs(pred - ytest) / ytest)


def cal_MAE(pred, ytest, mean_data, std_data):
    pred = (pred * std_data) + mean_data
    ytest = (ytest * std_data) + mean_data
    return np.mean(np.abs(pred - ytest))


X = []
Y = []
for i in range(length*2+2):
   tempx, tempy = data_function_train(i)
   X.append(tempx)
   Y.append(tempy)
Xtest, ytest= data_function_test(test_length)
#print(ytest.shape)
#print(Xtest[0:5], Xtest.shape)
verbose = 0
Experiment = 'Wind'
mse = np.zeros((2, N_runs, 3))
times = np.zeros((2, N_runs))
for run in range(N_runs):
    print("test MSE of zero function", np.mean(ytest ** 2))

    print('\n\n', '*' * 80, '\nrun', run)
    print('Current Experiment: Wind with ' + str(num_states)+' states')
    ph = 1
    for k in range(len(methods)):
        method = methods[k]
        if method == 'TIHT+SGD':
            t = tic()
            model = learning.TIHT_SGD_torch(X, Y, num_states, length, verbose, TIHT_epsilon, TIHT_learning_rate, TIHT_max_iters,
                                            lr2, epo2, b2, tol, Xtest, ytest, alpha = 1., lifting = True)
            T = toc(t)
            if ytest.ndim ==1:
                out_dim = 1
            else:
                out_dim = ytest.shape[1]
            Xtest_temp = torch.from_numpy(Xtest).float()
            pred = model(Xtest_temp)
            pred_numpy = pred.detach().numpy().reshape(-1, )
            if out_dim ==1:
                pred_numpy = pred_numpy.reshape(-1,)
            rmse1 = cal_RMSE(pred_numpy, ytest, mean_data, std_data)
            mape1 = cal_MAPE(pred_numpy, ytest, mean_data, std_data)
            mae1 = cal_MAE(pred_numpy, ytest, mean_data, std_data)
            print(method, 'Window size 1 time:', T)
            print('RMSE:', rmse1)
            print('MAPE:', mape1)
            print('MAE:', mae1)

            pred = pred_numpy.ravel()
            X_test_temp = []
            y_test_temp = []
            for i in range(len(Xtest)):
                X_test_temp.append(Xtest[i])
                y_test_temp.append(ytest[i])
            X_test_temp = np.asarray(X_test_temp)
            y_test_temp = np.asarray(y_test_temp)
            pred_k, ytest_k = pred_k_more(model, X_test_temp, y_test_temp, pred, 2, if_tc= True)
            pred_numpy = pred_k
            if out_dim ==1:
                pred_numpy = pred_numpy.reshape(-1,)
            rmse3 = cal_RMSE(pred_numpy, ytest_k, mean_data, std_data)
            mape3 = cal_MAPE(pred_numpy, ytest_k, mean_data, std_data)
            mae3 = cal_MAE(pred_numpy, ytest_k, mean_data, std_data)

            pred_k, ytest_k = pred_k_more(model, X_test_temp, y_test_temp, pred, 5, if_tc=True)
            pred_numpy = pred_k
            if out_dim ==1:
                pred_numpy = pred_numpy.reshape(-1,)
            rmse6 = cal_RMSE(pred_numpy, ytest_k, mean_data, std_data)
            mape6 = cal_MAPE(pred_numpy, ytest_k, mean_data, std_data)
            mae6 = cal_MAE(pred_numpy, ytest_k, mean_data, std_data)


            print(method, 'Window size 3 time:', T)
            print('RMSE:', rmse3)
            print('MAPE:', mape3)
            print('MAE:', mae3)

            print(method, 'Window size 6 time:', T)
            print('RMSE:', rmse6)
            print('MAPE:', mape6)
            print('MAE:', mae6)

            mse[k][run][0] = rmse1
            mse[k][run][1] = rmse3
            mse[k][run][2] = rmse6
            times[k][run] = T

        else:
            t = tic()
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
            #for i in range(len(model.A)):
            #    print(model.A[i].shape)
            T = toc(t)
            pred = []
            for i in range(len(Xtest)):
                pred.append(model.predict(Xtest[i]))
            pred = np.asarray(pred)

            rmse1 = cal_RMSE(pred, ytest, mean_data, std_data)
            mape1 = cal_MAPE(pred, ytest, mean_data, std_data)
            mae1 = cal_MAE(pred, ytest, mean_data, std_data)

            X_test_temp = []
            y_test_temp = []
            for i in range(len(Xtest)):
                X_test_temp.append(Xtest[i])
                y_test_temp.append(ytest[i])
            X_test_temp = np.asarray(X_test_temp)
            y_test_temp = np.asarray(y_test_temp)

            pred_k, ytest_k = pred_k_more(model, X_test_temp, y_test_temp, pred, 2)
            rmse3 = cal_RMSE(pred_k, ytest_k, mean_data, std_data)
            mape3 = cal_MAPE(pred_k, ytest_k, mean_data, std_data)
            mae3 = cal_MAE(pred_k, ytest_k, mean_data, std_data)

            pred_k, ytest_k = pred_k_more(model, X_test_temp, y_test_temp, pred, 5)
            rmse6 = cal_RMSE(pred_k, ytest_k, mean_data, std_data)
            mape6 = cal_MAPE(pred_k, ytest_k, mean_data, std_data)
            mae6 = cal_MAE(pred_k, ytest_k, mean_data, std_data)
            print(method, 'Window size 1 time:', T)
            print('RMSE:', rmse1)
            print('MAPE:', mape1)
            print('MAE:', mae1)

            print(method, 'Window size 3 time:', T)
            print('RMSE:', rmse3)
            print('MAPE:', mape3)
            print('MAE:', mae3)

            print(method, 'Window size 6 time:', T)
            print('RMSE:', rmse6)
            print('MAPE:', mape6)
            print('MAE:', mae6)

            mse[k][run][0] = rmse1
            mse[k][run][1] = rmse3
            mse[k][run][2] = rmse6
            times[k][run] = T


if not os.path.exists(xp_path):
    os.makedirs(xp_path)
file = open(xp_path + '/Error.pickle', 'wb')
pickle.dump(mse, file)
file.close()
file = open(xp_path + '/Time.pickle', 'wb')
pickle.dump(times, file)
file.close()
