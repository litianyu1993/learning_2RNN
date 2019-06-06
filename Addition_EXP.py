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

def tic():
    return time.clock()

def toc(t):
    return time.clock() - t


def generate_data_simple_addition(N_samples, seq_length, noise=0.):
    X = []
    Y = []
    for i in range(N_samples):
        if seq_length > 0:
            X.append(np.hstack((np.random.normal(0, 1, [seq_length, 2]),np.ones([seq_length,1]))))
            y = X[-1][:,0].sum()-X[-1][:,1].sum() + np.random.normal(0,noise)
        else:
            X.append([])
            y = 0
        Y.append(y)
    return np.asarray(X),np.asarray(Y).squeeze()
def sum_untill_length(length, dim):
    temp = 0
    for i in range(1, length+1):
        temp+= dim**i
    return temp


if __name__ == '__main__':
    '''
    python Addition_EXP.py 'launch' './new_examples_add' 0.1  2 0.1
    '''

    L_num_examples = [20, 40, 80, 160, 320, 640, 1500, 2560, 5000]
    N_runs = 1
    #print(N_runs)
    length = 2
    test_length = 6
    methods = ['NuclearNorm', 'TIHT', 'IHT', 'OLS', 'LSTM']
    TIHT_epsilon = 1e-15
    TIHT_learning_rate = 1e-1
    TIHT_max_iters = 5000
    xp_path = './SP_Addition/'

    b2 = 100
    lr2 = 0.001
    epo2 = 1000
    tol = 50
    verbose = False

    parser = argparse.ArgumentParser()
    parser.add_argument('-lne', '--list_number_examples', nargs = '+', help='list of examples numbers', type=int)
    parser.add_argument('-nr', '--number_runs', help='number of runs', type=int)
    parser.add_argument('-le', '--length', help='minimum training length', type=int)
    parser.add_argument('-tle', '--testing_length', help='testing length', type=int)
    parser.add_argument('-lm', '--method_list', nargs='+', help="List of methods to use")
    parser.add_argument('-eps', '--HT_epsilon', help='epsilon for TIHT and IHT', type=float)
    parser.add_argument('-lr', '--HT_learning_rate', help='learning rate for TIHT and IHT', type=float)
    parser.add_argument('-mi', '--HT_max_iter', help='number of max iterations for TIHT and IHT', type=int)
    parser.add_argument('-xp', '--xp_path', help='experiment folder path')

    parser.add_argument('-var', '--noise', help='variance of the gaussian noise', type=float)
    parser.add_argument('-ns', '--states_number', help='number of states for the model', type=int)
    parser.add_argument('-a', '--alpha', help='hyperparameter for nuclear norm method', type=float)
    parser.add_argument('-ld', '--load_data', help='load the previously created data', action='store_true')

    parser.add_argument('-lr2', help='learning rate for sgd 2rnn', type=float)
    parser.add_argument('-epo2', help='number of epochs for sgd 2rnn', type=int)
    parser.add_argument('-b2', '--batch_size', help='batch size for sgd 2rnn', type=int)
    parser.add_argument('-t', '--tolerance', help='tolerance for sgd 2rnn', type=int)
    args = parser.parse_args()

    if args.noise != None:
        noise_level = args.noise
    else:
        raise Exception('Did not initialize noise_level, try set up after -var argument')
    if args.states_number != None:
        num_states = args.states_number
    else:
        raise Exception('Did not initialize state numbers, try set up after -ns argument')
    if args.alpha != None:
        alpha = args.alpha
    else:
        raise Exception('Did not initialize alpha, try set up after -a argument')
    load_data = True

    if args.list_number_examples:
        L_num_examples = args.list_number_examples
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
    if not os.path.exists(xp_path + 'noise_' + str(noise_level)):
        os.makedirs(xp_path + 'noise_' + str(noise_level))
    xp_path = xp_path + 'noise_' + str(noise_level)+'/'

    if not os.path.exists(xp_path):
        os.makedirs(xp_path)

    results = dict([(m, {}) for m in methods])

    for num_examples in L_num_examples:
        for m in methods:
            results[m][num_examples] = []

    results['NUM_EXAMPLES'] = L_num_examples

    times = dict([(m, {}) for m in methods])

    for num_examples in L_num_examples:
        for m in methods:
            times[m][num_examples] = []

    times['NUM_EXAMPLES'] = L_num_examples



    for run in range(N_runs):


        if load_data == False:
            data_function = lambda l: generate_data_simple_addition(1000, l, noise=noise_level)
            Xtest, ytest = data_function(test_length)
            with open('./Data/Addition/noise_' + str(noise_level) + '/Test.pickle', 'wb') as f:
                pickle.dump([Xtest, ytest], f)

        elif load_data == True:
            data_function = lambda l: generate_data_simple_addition(1000, l, noise=noise_level)
            with open('./Data/Addition/noise_' + str(noise_level) + '/Test.pickle', 'rb') as f:
                [Xtest, ytest] = pickle.load(f)

        print("test MSE of zero function", np.mean(ytest ** 2))

        print('\n\n','*'*80,'\nrun',run)
        for num_examples in L_num_examples:
            print('______\nsample size:', num_examples)
            print('Current Experiment: Addition with noise '+str(noise_level) +' and ' + str(num_states)+' states')
            data_function = lambda l: generate_data_simple_addition(num_examples, l, noise=noise_level)
            Xl, yl = data_function(length)
            X2l, y2l = data_function(length * 2)
            X2l1, y2l1 = data_function(length * 2 + 1)

            for method in methods:

                if method != 'LSTM' and method != 'TIHT+SGD':
                    Tl = learning.sequence_to_tensor(Xl)
                    T2l = learning.sequence_to_tensor(X2l)
                    T2l1 = learning.sequence_to_tensor(X2l1)
                    t=tic()
                    Hl = learning.approximate_hankel(Tl, yl, alpha_ini_value=alpha,
                                                     rank=num_states, eps=TIHT_epsilon,
                                                     learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                                     method=method, verbose=-1)
                    H2l = learning.approximate_hankel(T2l, y2l, alpha_ini_value=alpha,
                                                      rank=num_states, eps=TIHT_epsilon,
                                                      learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                                      method=method, verbose=-1)
                    H2l1 = learning.approximate_hankel(T2l1, y2l1, alpha_ini_value=alpha,  rank=num_states, eps=TIHT_epsilon,
                                                       learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                                       method=method, verbose=-1)

                    learned_model = learning.spectral_learning(num_states, H2l, H2l1, Hl)

                    test_mse = learning.compute_mse(learned_model, Xtest, ytest)
                    train_mse = learning.compute_mse(learned_model, X2l1, y2l1)
                    #print(test_mse)
                    if train_mse > np.mean(y2l1 ** 2):
                        test_mse = np.mean(ytest ** 2)
                    print(method,"test MSE:", test_mse, "\t\ttime:",toc(t))
                    results[method][num_examples].append(test_mse)
                    times[method][num_examples].append(toc(t))
                elif method == 'LSTM':

                    def padding_function(x, desired_length):
                        if desired_length <= x.shape[1]:
                            return x
                        x = np.insert(x, x.shape[1], np.zeros((desired_length - x.shape[1], 1, x.shape[2])), axis=1)
                        return x

                    Xl_padded = padding_function(Xl, test_length)
                    X2l_padded = padding_function(X2l, test_length)
                    X2l1_padded = padding_function(X2l1, test_length)
                    #Xtest = padding_function(Xtest, test_length)
                    X = np.concatenate((Xl_padded, X2l_padded, X2l1_padded))
                    Y = np.concatenate((yl, y2l, y2l1))
                    t = tic()
                    learned_model = learning.RNN_LSTM(X, Y, test_length, num_states, noise_level, 'RandomRNN')
                    test_mse = learning.compute_mse(learned_model, Xtest, ytest, lstm = True)
                    train_mse = learning.compute_mse(learned_model, X2l1_padded, y2l1, lstm = True)
                    #if train_mse > np.mean(y2l1 ** 2):
                    #   test_mse = np.mean(ytest ** 2)
                    print(method, "test MSE:", test_mse, "\t\ttime:", toc(t))
                    results[method][num_examples].append(test_mse)
                    times[method][num_examples].append(toc(t))
                elif method == 'TIHT+SGD':
                    X = []
                    Y = []
                    for i in range(length * 2 + 2):
                        tempx, tempy = data_function(i)
                        X.append(tempx)
                        Y.append(tempy)
                    t = tic()
                    if noise_level == 0.:
                        TIHT_learning_rate = 0.000001
                    learned_model = learning.TIHT_SGD_torch(X, Y, num_states, length, verbose, TIHT_epsilon, TIHT_learning_rate,
                                              TIHT_max_iters,
                                              lr2, epo2, b2, tol, alpha=1., lifting=False)
                    test_mse = learning.compute_mse(learned_model, Xtest, ytest, if_tc = True)
                    train_mse = learning.compute_mse(learned_model, X2l1, y2l1, if_tc = True)
                    if train_mse > np.mean(y2l1 ** 2):
                        test_mse = np.mean(ytest ** 2)
                    print(method, "test MSE:", test_mse, "\t\ttime:", toc(t))
                    results[method][num_examples].append(test_mse)
                    times[method][num_examples].append(toc(t))


            with open(xp_path + 'results_'+str(num_states)+'_states.pickle','wb') as f:
                pickle.dump(results,f)
            with open(xp_path + 'times'+str(num_states)+'_states.pickle', 'wb') as f:
                pickle.dump(times, f)



