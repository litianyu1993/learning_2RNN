
'''
Some conventions for the comments:
s: number of states
n: number of examples
l: length of the sequence taken
d_x: number of variables of input data
d_y: number of variables of output data
'''

from LinRNN import LinRNN
import numpy as np
import tt
from sys import stdout
from collections import Counter
import cvxpy
import tensorflow as tf
import scipy.sparse.linalg as splinalg
import sys
import torch
import TT_learning
from keras.layers import Dense, Activation, Embedding, Masking, Input, LSTM, SimpleRNN
from keras import optimizers
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras



def compute_mse(mdl, X, y, print_first_five_results = False, lstm = False, if_tf = False, if_tc = False):
    '''
    Compute the mean squared errors for the predictions computed by model 'mdl'
    given the input data X, and the desired output y.
    :param mdl: LinRNN model, from the class LinRNN.py
    :param X: Desired testing input data, should be of the dimension of n*l*d_x
    :param y: Desired testing output data, should be of the dimension of n*d_y
    :param print_first_five_results: Set it as True if you want to do so, False otherwise
    :return: Mean squared error between predictions and true values
    '''
    pred = []
    if lstm:
        pred = mdl.predict(X)
        pred = pred.reshape(y.shape)
    elif if_tf:
        Xtest_temp = tf.convert_to_tensor(X, np.float32)
        pred = mdl.predict(Xtest_temp).numpy().reshape(-1, )
        #print(pred.shape)
        #print(y.shape)
    elif if_tc:
        Xtest_temp = torch.from_numpy(X).float()
        if y.ndim == 1:
            out_dim = 1
        else:
            out_dim = y.shape[1]
        pred = mdl(Xtest_temp).detach().numpy().reshape(-1, out_dim)
        y = y.reshape(-1, out_dim)
        #print(pred.shape, y.shape)
    else:
        for o in X:
            pred.append(mdl.predict(o))
        pred = np.array(pred)
    #if print_first_five_results == True:
        #print(X[0:5])
        #print(pred[0:1])
        #print(y[0:1])

    return np.mean((pred - y) ** 2)

def sequence_to_tensor(X):
    '''
    Transforming X from numpy array mode to tensor mode
    :param X: Input data, should be of the dimension of n*l*d_x
    :return: Tensor form of the input data X
    '''
    if X.shape[1] == 0: # empty sequences
        return X.reshape(10,0)
    news = []
    for i in range(X.shape[0]):
        temp = X[i][0]
        for j in range(1, X.shape[1]):
            temp = np.tensordot(temp, X[i][j], axes=0)
        news.append(temp)
    news = np.asarray(news)
    return np.asarray(news)




def dyn_print(s):
    '''
    Dynamical printing function
    :param s: String you want to display
    :return: NA
    '''
    stdout.write("\r%s" % s)
    stdout.flush()

def project_TT(T, rank):
    '''
    Vectorizing tensor
    :param T: Input tensor T
    :param rank: Number of rank you want to preserve
    :return: Vectorized tensor
    '''
    T = tt.vector(T).round(1e-10, rmax=rank)
    return T.full()

def project_SVD(T, rank, svd='dense'):
    '''
    Decompose and reconstruct the tensor using SVD (First reshape the tensor to a second order tensor, i.e. matrix)
    :param T: Input tensor
    :param rank: Number of rank you want to preserve
    :param svd: the mode for SVD (dense or thin), default as dense
    :return: Reconstructed tensor
    '''
    m,n = np.prod(T.shape[:T.ndim//2]),np.prod(T.shape[T.ndim//2:])
    if svd == 'dense' or rank > min(m,n)-1:
        U,s,V  = np.linalg.svd(T.reshape([m,n]),full_matrices=False)
        idx = np.argsort(s)[::-1][:rank]
        U = U[:,idx]
        s = s[idx]
        V = V[idx,:]
    elif svd == 'sparse':
        U,s,V  = splinalg.svds(T.reshape([m,n]),rank)


    return U.dot(np.diag(s).dot(V)).reshape(T.shape)


def TIHT(X, Y, rank, max_iters=1000, learning_rate=1e-4, targ=None, verbose=1, eps=1e-10, method='TT'):
    '''
    Iterative hard thresholding method, given the options of using tensor form or not
    :param X: Input data X, should be of dimension n*l*d_x
    :param Y: Output data Y, should be of dimension n*d_y
    :param rank: Desired rank
    :param max_iters: Max number of iterations for the hard thresholding method
    :param learning_rate: Learning rate
    :param targ: The desired tensor (matrix) recovered, using for debugging
    :param verbose: Verbose = 0: none display
    :param eps: Eposilon parameter for the hard thresholding method, used to determine when to stop the iteration
    :param method: use 'TT' to apply TIHT, use 'SVD' to apply IHT
    :return: Recovered tensor
    '''
    if Y.ndim == 1:
        Y = Y.reshape((Y.shape[0], 1))

    N = len(X)
    dim = X[0].shape[0]
    l = X[0].ndim
    p = Y.shape[1]


    matshape = (dim**l,p)
    tensorshape = [dim]*l + [p]
    X = np.array(X).reshape((N,dim**l))
    T_old = np.zeros(matshape)

    for it in range(max_iters):
        if method == 'TT':
            T = project_TT((T_old + learning_rate * X.T.dot(Y - X.dot(T_old))).reshape(tensorshape), rank).reshape(matshape)
        elif method == 'SVD':
            T = project_SVD((T_old + learning_rate * X.T.dot(Y - X.dot(T_old))).reshape(tensorshape), rank).reshape(matshape)
        else:
            raise NotImplementedError("available methods are 'TT' and 'SVD'")

        if verbose > 0 and it%20 == 0:
            dyn_print(str((np.linalg.norm(T_old-T),np.linalg.norm(targ-T) if targ is not None else '-')))
        if np.linalg.norm(T_old-T) > 100:
            if verbose > 0:
                print ("TIHT/IHT divergence")
            return None
        if np.linalg.norm(T_old-T) < eps:
            #print( tensorshape)
            if verbose > 0: print("")
            return T.reshape(tensorshape)
        T_old = T
    if verbose >0:
        print("TIHT/IHT: reached max_iters")
    return T.reshape(tensorshape)


def OLS(X,Y):
    '''
    OLS method to recover the tensor
    :param X: Input data, of dimension n*l*d_x
    :param Y: Output data, of dimension n*d_y
    :return: recovered tensor
    '''
    if Y.ndim == 1:
        Y = Y.reshape((Y.shape[0], 1))

    N = len(X)
    dim = X[0].shape[0]
    l = X[0].ndim
    p = Y.shape[1]
    return np.linalg.lstsq(X.reshape(N,dim**l),Y, rcond=None)[0].reshape([dim]*l + [p])

def nuclear_norm_cv_alpha(X, y, alpha_ini_value = 0.1, alpha_vec = []):
    #cross validation for nuclear norm's alpha
    #default alpha_vec can be either input through alpha_vec argument
    #or if alpha_ini_value is given, alpha_vec will be
    # [alpha_ini_value/100, alpha_ini_value/10, alpha_ini_value, alpha_ini_value*10, alpha_ini_value*100]

    if not alpha_vec and alpha_ini_value!=0.:
        mulitplier = 5.
        alpha_vec = [alpha_ini_value / mulitplier**3, alpha_ini_value /mulitplier**2,
                     alpha_ini_value / mulitplier**1,
                     alpha_ini_value,
                     alpha_ini_value * mulitplier**1, alpha_ini_value * mulitplier**2,
                     alpha_ini_value * mulitplier**3]
    if y.ndim == 1:
        y = y.reshape((y.shape[0],1))

    best_alpha_vec = []
    for i in range(1):
        index = np.random.permutation(np.arange(len(X)))

        X_train = X[index[0:np.int(0.8*len(X))]]
        X_vali = X[index[np.int(0.8*len(X)):]]
        y_train = y[index[0:np.int(0.8*len(X))]]
        y_vali = y[index[np.int(0.8*len(X)):]]

        dim = X_vali.shape[1]
        N = X_vali.shape[0]
        l = X_vali.ndim - 1
        p = y_vali.shape[1]

        best_mse = sys.float_info.max
        best_alpha = -1.
        for alpha in alpha_vec:
            T = nuclear_norm_minimization(X_train, y_train, alpha)
            if T is None:
                continue
            temp_x = X_vali.reshape([N,dim**l])
            temp_t = T.reshape(dim**l,p)
            temp_mse = np.mean((np.dot(temp_x, temp_t) - y_vali)**2)
            if temp_mse < best_mse:
                best_mse = temp_mse
                best_alpha = alpha
        best_alpha_vec.append(best_alpha)
    best_alpha_vec = np.asarray(best_alpha_vec)
    counter_alpha = Counter(best_alpha_vec)
    best_alpha = counter_alpha.most_common(1)[0][0]
    #print('best alpha is', best_alpha)
    return best_alpha


def nuclear_norm_minimization(X,y, alpha = 0.1):
    '''
    OLS method to recover the tensor
    :param X: Input data, of dimension n*l*d_x
    :param Y: Output data, of dimension n*d_y
    :param alpha: Parameter to adjust the bias variance trade-off (Noisy data tend to imply lower alpha)
    :return: Recovered tensor
    '''
    if y.ndim == 1:
        y = y.reshape((y.shape[0],1))
    dim = X.shape[1]
    N = X.shape[0]
    l = X.ndim - 1
    p = y.shape[1]
    l1 = l//2
    l2 = l-l1
    T = cvxpy.Variable(dim**l,p)
    if p == 1:
        T = cvxpy.reshape(T, dim**l, 1)
    X = X.reshape([N,dim**l])
    if alpha != 0.:
        alpha = ((len(X) * y.shape[1]) ** 0.5) / alpha
        obj = cvxpy.Minimize(cvxpy.norm(cvxpy.reshape(T,dim ** l1,dim ** l2 * p), 'nuc') + cvxpy.pnorm((X*T - y),p = 2)/alpha)
        prob = cvxpy.Problem(obj)
    else:
        constraint = [(X * T == y)]
        obj = cvxpy.Minimize(cvxpy.norm(cvxpy.reshape(T, dim ** l1, dim ** l2 * p), 'nuc'))
        prob = cvxpy.Problem(obj, constraint)
    prob.solve(solver=cvxpy.SCS)
    if T.value is None:
        return None
    return np.array(T.value).reshape([dim]*l + [p])

def approximate_hankel(X, y, rank=10, alpha_ini_value = 0.1, alpha_vec = [], max_iters = 1000,
                       learning_rate=1e-4,targ=None,verbose=1,eps=1e-10, method='TIHT',minibatch_size=100):
    hyp = None
    while hyp is None:
        if method == 'TIHT':
            hyp = TIHT(X, y, rank, max_iters, learning_rate, targ, verbose, eps, method = 'TT')
        elif method == 'IHT':
            hyp = TIHT(X, y, rank, max_iters, learning_rate, targ, verbose, eps, method = 'SVD')
        elif method == 'TIHT_lowmem':
            hyp = TT_learning.TT_TIHT(X, y, rank, max_iters, learning_rate, targ, verbose, eps, minibatch_size)
        elif method == 'NuclearNorm':
            if alpha_ini_value != 0:
                best_alpha = nuclear_norm_cv_alpha(X, y, alpha_ini_value, alpha_vec)
                hyp = nuclear_norm_minimization(X, y, best_alpha)
            else:
                hyp = nuclear_norm_minimization(X, y, 0.)
        elif method == 'OLS':
            hyp = OLS(X,y)
        else:
            raise NotImplementedError("available methods are 'OLS', IHT', 'TIHT', 'TIHT_lowmem' and 'NuclearNorm")
        learning_rate /= 2.
    return hyp



def approximate_hankel_l(TL_vec, length):

    xdim = TL_vec[1].shape[0]
    ydim = TL_vec[1].shape[-1]
    temp_list = [0]
    for i in range(0, length+1):
        temp_list.append(xdim ** i)
    temp_list = np.asarray(temp_list)
    HL_dim = np.sum(temp_list)
    HL = np.zeros((HL_dim, HL_dim, ydim))
    for i in range(length+1):
        for j in range(length+1):
            index1 = index_converter_hl(i, xdim)
            index2 = index_converter_hl(j, xdim)
            tempH = TL_vec[i+j]
            tempH = tempH.reshape(xdim**(i), xdim**(j), ydim)
            HL[index1[0]:index1[1], index2[0]:index2[1], :] = tempH
    return HL

def approximate_hankel_plus(TL_vec, length):

    xdim = TL_vec[1].shape[0]
    ydim = TL_vec[1].shape[-1]
    temp_list = [0]
    for i in range(0, length + 1):
        temp_list.append(xdim ** i)
    temp_list = np.asarray(temp_list)
    HL_dim = np.sum(temp_list)
    H_plus = np.zeros((HL_dim, xdim, HL_dim, ydim))
    for i in range(length+1):
        for j in range(length+1):
            index1 = index_converter_hl(i, xdim)
            index2 = index_converter_hl(j, xdim)
            tempH = TL_vec[i + j +1]
            tempH = tempH.reshape(xdim ** (i), xdim,  xdim ** (j), ydim)
            H_plus[index1[0]:index1[1], :, index2[0]:index2[1], :] = tempH
    return H_plus

def approximate_hankel_minus(TL_vec, length):

    xdim = TL_vec[1].shape[0]
    ydim = TL_vec[1].shape[-1]
    temp_list = [0]
    for i in range(0, length + 1):
        temp_list.append(xdim ** i)
    temp_list = np.asarray(temp_list)
    HL_dim = np.sum(temp_list)
    if ydim == 1:
        H_minus = np.zeros((HL_dim, ))
    else:
        H_minus = np.zeros((HL_dim, ydim))
    for i in range(length+1):
        tempH = TL_vec[i]
        if ydim == 1:
            tempH = tempH.reshape(-1, )
        else:
            tempH = tempH.reshape(-1, ydim)
        index = index_converter_hl(i, xdim)
        H_minus[index[0]:index[1]] = tempH

    return H_minus



def index_converter_hl(index, xdim):
    start = int(np.sum([xdim**i for i in range(index)]))
    return [start, start + xdim**index]

def get_all_TLs(X_vec, y_vec, rank=10, alpha_ini_value = 0.1, alpha_vec = [],
                            max_iters = 1000, learning_rate=1e-4,targ=None,
                            verbose=1,eps=1e-10, method='TIHT'):

    TL_vec = [np.array(y_vec[0].mean(axis = 0))]  # initialize with empty sequence Hankel
    for i in range(1, len(X_vec)):
        X = X_vec[i]
        y = y_vec[i]
        TL_vec.append(approximate_hankel(X, y, rank=rank, alpha_ini_value=alpha_ini_value, alpha_vec=alpha_vec,
                                         max_iters=max_iters, learning_rate=learning_rate, targ=targ,
                                         verbose=verbose, eps=eps, method=method))
    return TL_vec


def spectral_learning_multiple_length(rank, H_minus, Hl, H_plus):
    U, s, V = np.linalg.svd(Hl.reshape(Hl.shape[0], Hl.shape[1]*Hl.shape[2]))
    U = U[:, :rank]
    V = V[:rank]
    s = s[:rank]

    Pinv = np.diag(1. / s).dot(U.T)
    Sinv = V.T

    H_plus = H_plus.reshape(H_plus.shape[0], H_plus.shape[1], H_plus.shape[2]*H_plus.shape[3])
    A = np.tensordot(Pinv, H_plus, axes=(1, 0))
    A = np.tensordot(A, Sinv, axes=[2, 0])
    alpha = Sinv.T.dot(H_minus.reshape(-1, 1)).reshape(-1, )
    omega = Pinv.dot(H_minus)
    model = LinRNN(alpha, A, omega)
    return model





def spectral_learning(rank, H_2l, H_2l1, H_l):
    if H_2l.ndim % 2 == 0: # scalar outputs
        out_dim = 1
        l = H_l.ndim
    else:
        out_dim = H_2l.shape[-1]
        l = H_l.ndim - 1

    d = H_l.shape[0]
    U, s, V = np.linalg.svd(H_2l.reshape([d ** l, d ** l * out_dim]))
    idx = np.argsort(s)[::-1]
    U = U[:, idx[:rank]]
    V = V[idx[:rank], :]
    s = s[idx[:rank]]

    Pinv = np.diag(1. / s).dot(U.T)
    Sinv = V.T

    A = np.tensordot(Pinv, H_2l1.reshape([d ** l, d, d ** l * out_dim]), axes=(1, 0))
    A = np.tensordot(A, Sinv, axes=[2, 0])
    h_l = H_l.ravel()
    if out_dim == 1:
        omega = Pinv.dot(h_l)
    else:
        omega = (Pinv.dot(H_l.reshape([d**l,out_dim])))
    alpha = Sinv.T.dot(h_l)
    model = LinRNN(alpha, A, omega)
    return model

def RNN_LSTM(X, Y, test_length, num_state, noise_level, Experiment, if_linear = False):


    xavia = keras.initializers.glorot_normal(seed=None)
    l2 = keras.regularizers.l2(0.01)
    adam = optimizers.Adam(lr=0.01)



    checkpointer = ModelCheckpoint(filepath='./Data/Addition/noise_' + str(noise_level) +
                                            '/bestmodel' + Experiment +
                                            str(noise_level) + str(num_state) + str(if_linear) +
                                            ".hdf5",
                                   verbose=0,
                                   save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=200)
    reduceonplateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                        factor=0.1, patience=50, verbose=0,
                                                        mode='auto', min_delta=0.0001,
                                                        cooldown=0, min_lr=0)

    inputs = Input(shape=(X.shape[1], X.shape[2]))
    mask = Masking(mask_value=0.0)(inputs)
    if if_linear == 'True':
        lstm_out = LSTM(num_state, activation='linear', recurrent_activation='linear', kernel_initializer=xavia)(mask)

    else:
        lstm_out = LSTM(num_state)(mask)
    x = Dense(Y.ndim, kernel_initializer=xavia)(lstm_out)

    model = Model(inputs=inputs, outputs=x)

    model.compile(optimizer=adam, loss='mean_squared_error')
    try:
        model.fit(X, Y, epochs=1000, batch_size=1280, verbose=0,  validation_split=0.1,
                            callbacks=[checkpointer, early_stopping, reduceonplateau])
    except KeyboardInterrupt:
        print("stop training...")

    model.load_weights('./Data/Addition/noise_' + str(noise_level) + '/bestmodel' +
                       Experiment + str(noise_level) + str(num_state) + str(if_linear) + ".hdf5")
    return model

def TIHT_SGD_torch(X, Y, num_states, length, verbose, TIHT_epsilon, TIHT_learning_rate, TIHT_max_iters,
             sgd_2rnn_learning_rate, sgd_2rnn_epochs, sgd_2rnn_batch_size, sgd_2rnn_tolerance, X_vali = None, Y_vali = None, alpha = 1.,
                   lifting = True):
    if lifting == False:
        X_vec, y_vec = [], []
        for i in range(0, len(X)):
            X_vec.append(sequence_to_tensor(X[i]))
            y_vec.append(Y[i])
        H_vec = []
        temp_index_list = [length, 2*length, 2*length+1]
        for i in temp_index_list:
            X_temp = sequence_to_tensor(X[i])
            Y_temp = Y[i]

            H_temp = approximate_hankel(X_temp, Y_temp, alpha_ini_value=alpha,
                                                 rank=num_states, eps=TIHT_epsilon,
                                                 learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                                 method='TIHT', verbose=-1)
            H_vec.append(H_temp)
        sp_model = spectral_learning(num_states, H_vec[1], H_vec[2], H_vec[0])
    else:
        X_vec, y_vec = [], []
        for i in range(0, len(X)):
            X_vec.append(sequence_to_tensor(X[i]))
            y_vec.append(Y[i])
        TL_vec = get_all_TLs(X_vec, y_vec, rank=num_states, eps=TIHT_epsilon,
                                      learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                      method='TIHT', verbose=0, alpha_ini_value=alpha)

        Hl = approximate_hankel_l(TL_vec=TL_vec, length=length)
        Hl_plus = approximate_hankel_plus(TL_vec=TL_vec, length=length)
        Hl_minus = approximate_hankel_minus(TL_vec=TL_vec, length=length)
        #print(Hl.shape, Hl_plus.shape, Hl_minus.shape)
        #print(num_states, Hl.shape)
        sp_model = spectral_learning_multiple_length(num_states, Hl_minus, Hl, Hl_plus)
        #print(sp_model.A.shape)
        #for i in range(len(sp_model.A)):
        #    print(sp_model.A[i].shape)
    def enough_iter(iter_count, max_iter):
        for count in iter_count:
            if count >= max_iter:
                return True
        return False


    X_vec, y_vec = [], []
    for i in range(0, len(X)):
        X_vec.append(sequence_to_tensor(X[i]))
        y_vec.append(Y[i])

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch_2rnn import Net
    #print(X[1].shape, Y[1].shape)
    if Y[1].ndim == 1:
        out_dim =1
    else:
        out_dim = Y[1].shape[1]
    input_dim = X[1].shape[2]
    #print(sp_model.A, sp_model.alpha, sp_model.Omega)

    net = Net(num_states, input_dim, out_dim, A=sp_model.A, alpha=sp_model.alpha.reshape(1, -1), Omega=sp_model.Omega.reshape(-1, out_dim),
              if_initialize=True)
    optimizer = optim.Adamax(net.parameters(), lr=sgd_2rnn_learning_rate)
    criterion = nn.MSELoss()

    #print('maxiter', max_iter)
    #sgd_2rnn_epochs = 0
    #print(sp_model.A)
    #print(sp_model.alpha)
    #print(sp_model.Omega)
    if X_vali is not None:
        X_vali = torch.from_numpy(X_vali).float()
        Y_vali = torch.from_numpy(Y_vali.reshape(-1, out_dim)).float()
    for j in range(sgd_2rnn_epochs):
        target = torch.from_numpy(Y[-1].reshape(-1, out_dim)).float()
        input = torch.from_numpy(X[-1]).float()
        output = net(input).view(-1, out_dim)
        #print(output.shape, target.shape)
        loss = criterion(output, target)
        pre_loss = 100000000
        index_vec = []
        index_reached = []
        iter_count = []
        for i in range(len(X)):
            index_vec.append(np.arange(0, int(X[i].shape[0])))
            index_vec[-1] = np.random.permutation(index_vec[-1])
            index_reached.append(0)
            iter_count.append(0)
        max_iter = len(index_vec[-1]) * 1. / sgd_2rnn_batch_size
        while not enough_iter(iter_count, max_iter):
            for index1 in range(1, len(X)):
                optimizer.zero_grad()  # zero the gradient buffers
                temp_loss = 0.
                index_list = index_vec[index1][index_reached[index1]:index_reached[index1] + sgd_2rnn_batch_size]
                index_reached[index1] = index_reached[index1] + sgd_2rnn_batch_size
                iter_count[index1] += 1
                #print(Y[index1].shape)
                target = torch.from_numpy(Y[index1][index_list].reshape(-1, out_dim)).float()
                input =  torch.from_numpy(X[index1][index_list]).float()
                output = net(input).view(-1, out_dim)
                loss = criterion(output, target)
                #print(loss)
                loss.backward()
                optimizer.step()
            target = torch.from_numpy(Y[-1].reshape(-1, out_dim)).float()
            input = torch.from_numpy(X[-1]).float()
            output = net(input).view(-1, out_dim)
            loss = criterion(output, target)
            if np.abs((loss - pre_loss).detach().numpy()) < 0.001:
                break
            pre_loss = loss
        if verbose == True:
            if X_vali is not None:
                pred_vali = net(X_vali).view(-1, out_dim)
                vali_loss = criterion(pred_vali, Y_vali)
                print('Epoch'+str(j)+' Training Loss: '+str(pre_loss)+' Vali Loss: '+str(vali_loss))
            else:
                print('Epoch' + str(j) + ' Training Loss: ' + str(pre_loss))
    return net

def TIHT_SGD(X, Y, num_states, length, verbose, TIHT_epsilon, TIHT_learning_rate, TIHT_max_iters,
             sgd_2rnn_learning_rate, sgd_2rnn_epochs, sgd_2rnn_batch_size, sgd_2rnn_tolerance, alpha = 1., lifting = True):
    import second_RNN_model
    from second_RNN_model import Model as Model2RNN
    if lifting == False:
        X_vec, y_vec = [], []
        for i in range(0, len(X)):
            X_vec.append(sequence_to_tensor(X[i]))
            y_vec.append(Y[i])
        H_vec = []
        temp_index_list = [length, 2*length, 2*length+1]
        for i in temp_index_list:
            X_temp = sequence_to_tensor(X[i])
            Y_temp = Y[i]

            H_temp = approximate_hankel(X_temp, Y_temp, alpha_ini_value=alpha,
                                                 rank=num_states, eps=TIHT_epsilon,
                                                 learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                                 method='TIHT', verbose=-1)
            H_vec.append(H_temp)
        sp_model = spectral_learning(num_states, H_vec[1], H_vec[2], H_vec[0])
    else:
        X_vec, y_vec = [], []
        for i in range(0, len(X)):
            X_vec.append(sequence_to_tensor(X[i]))
            y_vec.append(Y[i])
        TL_vec = get_all_TLs(X_vec, y_vec, rank=num_states, eps=TIHT_epsilon,
                                      learning_rate=TIHT_learning_rate, max_iters=TIHT_max_iters,
                                      method='TIHT', verbose=0, alpha_ini_value=alpha)

        Hl = approximate_hankel_l(TL_vec=TL_vec, length=length)
        Hl_plus = approximate_hankel_plus(TL_vec=TL_vec, length=length)
        Hl_minus = approximate_hankel_minus(TL_vec=TL_vec, length=length)

        sp_model = spectral_learning_multiple_length(num_states, Hl_minus, Hl, Hl_plus)

    def early_stopping(vali_error_vec, tolerance, step):
        if len(vali_error_vec) < 3:
            return False, step
        else:
            min_error = np.min(np.asarray(vali_error_vec[:-2]))
            if vali_error_vec[-1] < min_error:
                return False, step
            elif vali_error_vec[-1] >= min_error and step < tolerance:
                step += 1
                return False, step
            elif vali_error_vec[-1] >= min_error and step >= tolerance:
                return True, step

    def convert_X_y(X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        T_X = tf.convert_to_tensor(X, np.float32)
        T_y = tf.convert_to_tensor(y, np.float32)
        return T_X, T_y

    def enough_iter(iter_count, max_iter):
        for count in iter_count:
            if count >= max_iter:
                return True
        return False

    X_vec, y_vec = [], []
    for i in range(0, len(X)):
        X_vec.append(sequence_to_tensor(X[i]))
        y_vec.append(Y[i])

    T_X = []
    T_y = []
    for i in range(0, len(X)):
        if i < len(X) - 1:
            temp_x, temp_y = convert_X_y(X[i], Y[i])
            T_X.append(temp_x)
            T_y.append(temp_y)
        else:
            temp_x, temp_y = convert_X_y(X[i][0:int(0.8 * X[i].shape[0])], Y[i][0:int(0.8 * X[i].shape[0])])
            T_X.append(temp_x)
            T_y.append(temp_y)
            temp_x, temp_y = convert_X_y(X[i][int(0.8 * X[i].shape[0]):], Y[i][int(0.8 * X[i].shape[0]):])
            Tx_vali = temp_x
            Ty_vali = temp_y
        #print(T_X[i])
    training_error = []
    vali_error = []
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(sgd_2rnn_learning_rate, global_step,
                                               sgd_2rnn_epochs, 0.97, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_loss_results = []

    model = Model2RNN(num_units=num_states, input_dim=X[0].ndim, output_dim=Y[0].ndim - 1, A=sp_model.A,
                      alpha=sp_model.alpha, Omega=sp_model.Omega, if_initialize=True)

    step = 0
    for epoch in range(sgd_2rnn_epochs):
        epoch_loss_avg = []
        epoch_vali = []

        index_vec = []
        index_reached = []
        iter_count = []
        for i in range(len(T_X)):
            index_vec.append(np.arange(0, int(T_X[i].shape[0])))
            index_vec[-1] = np.random.permutation(index_vec[-1])
            index_reached.append(0)
            iter_count.append(0)
        max_iter = len(index_vec[-1]) * 1. /sgd_2rnn_batch_size

        while not enough_iter(iter_count, max_iter):
            for index1 in range(1, len(T_X)):
                temp_loss = 0.
                index_list = index_vec[index1][index_reached[index1]:index_reached[index1] + sgd_2rnn_batch_size]
                index_reached[index1] = index_reached[index1] + sgd_2rnn_batch_size
                iter_count[index1] += 1
                x = tf.gather(T_X[index1], index_list)
                y = tf.gather(T_y[index1], index_list)
                #print(x)
                #print(T_X[index1])
                loss_value, grads_train = second_RNN_model.grad(model, x, y)
                temp_loss += loss_value.numpy()
                optimizer.apply_gradients(zip(grads_train, [model.A, model.alpha, model.Omega]),
                                          global_step)
            epoch_loss_avg.append(temp_loss / len(T_X))
            #verbose = 1
            #if epoch % 1 == 0 and verbose == 1:
            #    print("Epoch {:03d}: Training loss: {:.4f}".format(epoch,np.mean(np.asarray(epoch_loss_avg))))
            #    print(max_iter)
        for i in range(np.int(int(Tx_vali.shape[0]) / 2)):
            index = i * 2
            index_list = list(range(index, index + 2))
            x_vali_temp = tf.gather(Tx_vali, index_list)
            y_vali_temp = tf.gather(Ty_vali, index_list)
            vali_loss = second_RNN_model.loss(model, x_vali_temp, y_vali_temp)
            epoch_vali.append(vali_loss.numpy())

        train_loss_results.append(np.mean(np.asarray(epoch_loss_avg)))

        if epoch % 1 == 0 and verbose == 1:
            print("Epoch {:03d}: Training loss: {:.4f}, Vali_loss: {:.7f}".format(epoch,
                                                                                  np.mean(np.asarray(epoch_loss_avg)),
                                                                                  np.mean(np.asarray(vali_loss))))
        training_error.append(np.mean(np.asarray(epoch_loss_avg)))
        vali_error.append(np.mean(np.asarray(vali_loss)))
        if np.isnan(vali_error[-1]):
            break
        stop, step = early_stopping(vali_error, sgd_2rnn_tolerance, step)
        #if stop:
        #    break
    return model




