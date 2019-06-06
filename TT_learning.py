"""
WARNING: the stopping criterion we use now is not good for the mini-batch setting...
"""

from LinRNN import LinRNN
import numpy as np
import tt
from sys import stdout
from collections import Counter

import cvxpy

import scipy.sparse.linalg as splinalg
import sys

import TT_learning


def TT_TIHT(X, Y, rank, max_iters=1000, learning_rate=1e-4, targ=None, verbose=1, eps=1e-10, minibatch_size=100):
    '''
    Iterative hard thresholding method, given the options of using tensor form or not
    :param X: Input data X, should be a list of N matrices of size d x l 
    :param Y: Output data Y, should be of dimension N*d_y
    :param rank: Desired rank
    :param max_iters: Max number of iterations for the hard thresholding method
    :param learning_rate: Learning rate
    :param targ: The desired tensor (matrix) recovered, using for debugging
    :param verbose: Verbose = 0: none display
    :param eps: Eposilon parameter for the hard thresholding method, used to determine when to stop the iteration
    :return: Recovered tensor
    '''
    #print(X[0].shape)
    assert  X[0].ndim == 2, '[Error] TT_TIHT takes a list of N matrices of size l*d as input (not a list of l-th order\
    	d-dimensional tensors like TIHT'

    if Y.ndim == 1:
        Y = Y.reshape((Y.shape[0], 1))
    

    N = len(X)
    l,d = X[0].shape
    p = Y.shape[1]
    ranks = [1] + ([rank] * l) + [1]

    X = np.array(X)


    # Split the data in batches and build the corresonding TT decomposition for the inputs
    X_all,Y_all = X,Y
    if minibatch_size < 0:
        minibatch_size = N
    X_batches = []
    Y_batches = []
    xcores_batches = []
    batch_idx = 0
    while batch_idx < N:
        X_batches.append(X[range(batch_idx,min(N,batch_idx+minibatch_size))])
        Y_batches.append(Y[range(batch_idx,min(N,batch_idx+minibatch_size))])

        x_cores = [X_batches[-1][:,0,:,np.newaxis].T]
        for i in range(1,l):
            x_cores.append(X_batches[-1][:,i,:,np.newaxis] *  np.eye(len(X_batches[-1]))[:,np.newaxis,:])
        xcores_batches.append(x_cores)
        batch_idx += minibatch_size



    # initialze TT cores
    cores = []
    for i in range(1,l+2):
        dim = d if i < l+1 else p
        cores.append(np.random.normal(0,0.000001,[ranks[i-1],dim,ranks[i]]))
    old_cores = [G.copy() for G in cores]
    T = tt.vector.from_list(cores)

    # helper function for later
    def prod_H_x(H_cores,x_cores):
        """
        Product of H (of order l+1) and \sum_n x_1^n \otimes ... \otimes x_l^n (of order l)
        along the first l modes when both tensors are given in TT format 
        """
        res = np.tensordot(x_cores[0],H_cores[0],axes=(1,1)).transpose([0,2,1,3])
        for c1,c2 in zip(x_cores[1:],H_cores[1:-1]):
            c1c2 =  np.tensordot(c1,c2,axes=(1,1))
            res = np.tensordot(res,c1c2,axes=((3,2),(2,0)))
        res = np.tensordot(res,H_cores[-1],axes=(3,0)).squeeze()
        if res.ndim == 1:
            res = res[:, np.newaxis]
        #print(res.shape)
        res = res[:,:,np.newaxis]
        return res


    grad = tt.vector.from_list(x_cores+[Y[:,:,np.newaxis]]) 

    for it in range(max_iters):
        X = X_batches[it % len(X_batches)]
        Y = Y_batches[it % len(Y_batches)]
        x_cores = xcores_batches[it % len(xcores_batches)]

        res = prod_H_x(cores,x_cores)
        # Equivalent but slower:
        #res = np.sum([TT_tenvecs_product(cores,xs).squeeze() for xs,y in zip(X,Y)])

        tmp = Y[:,:,np.newaxis] - res
        grad = tt.vector.from_list(x_cores+[tmp])  # gradient to respect to the whole H, not the independent cores...

        # we first approximate grad with tensor with low TT rank, otherwise sum is costly for large batches...
        round_grad = grad#.round(1e-10,rmax=rank)
        T = T + learning_rate * round_grad

        T = T.round(1e-10, rmax=rank)
        cores = tt.vector.to_list(T)

        # we put that in the a try block cause sometimes on the first iter the TT-decomposition is of smaller rank than
        # the initialization...
        try:
            progress = np.sum([np.linalg.norm(G1-G2) for G1,G2 in zip(old_cores,cores)])
        except:
            progress = 1
            
        if verbose > 0 and it%200 == 0:
            loss = 0.5/N*np.sum([np.linalg.norm(TT_tenvecs_product(cores,xs).squeeze() - y)**2 for xs,y in
                zip(X_all,Y_all)]) 
            print("%i: %.10f %.10f" % (it,progress,loss))
        if progress > 100:
            if verbose > 0:
                print ("\TIHT divergence",learning_rate)
            return None

        if progress < eps:
            if verbose > 0: print("")
            return cores

        old_cores = [G.copy() for G in cores]

    if verbose >0:
        print("\nTIHT: reached max_iters")
    return cores



def TT_product(a_cores, b_cores):
  """
  performs the TT chain of product between the two set of cores:
  -- a1 -- a2 -- a3 -- ...
     |     |     |
  -- b1 -- b2 -- b3 -- ... 
  """
  for i,(c1,c2) in enumerate(zip(a_cores,b_cores)):
    if i == 0:
        res = np.tensordot(c1,c2,(1,1)).transpose([0,2,1,3])
    else:
        res = np.tensordot(res,c1,(2,0))
        res = np.tensordot(res,c2,((2,3),(0,1)))
  return res


def TT_tenvecs_product(cores, xs):
    """
    perform the prodcut H \times_1 x1 \times_2 x2 \times_3 x3 ...
    where cores are the cores of the TT decomposition of H and
    xs = [x1,x2,x3,...] is the input sequence
    """
    res = None
    for c,x in zip(cores[:len(xs)],xs):
        if res is None:
            res = np.tensordot(c,x,axes=(1,0))
        else:
            res = np.tensordot(res,c,axes=(1,0))
            res = np.tensordot(res,x,axes=(1,0))
    if len(cores) == len(xs) + 1:
        return np.tensordot(res,cores[-1],axes=(1,0))
    else:
        return res

def TT_tenvecs_product_omit_one(cores, xs, idx):
    """
    same as TT_tenvecs_product but without perfomring the idx-th product 
    """
    res1 = TT_tenvecs_product(cores[:idx],xs[:idx]) if idx > 0 else 1
    if idx < len(cores) - 2:
        res2 = TT_tenvecs_product(cores[idx+1:],xs[idx+1:]) 
    elif idx == len(cores) - 2:
        res2 = cores[-1]
    else:
        res2 = np.eye(cores[-1].shape[1])
    res = np.tensordot(res1, xs[idx], axes=0) if idx < len(xs) else res1
    if idx == 0:
        res = np.expand_dims(res,0)
    if idx == len(cores) - 1:
        res2 = np.expand_dims(res2,-1)
    return np.tensordot(res, res2, axes=0)

def TT_compute_gradient(X,Y,cores,i):
    """
    [INNEFICIENT]
    compute the gradient of the square loss (1/2N * \sum_n || ... ||^2)
    w.r.t. to the i-th core of the TT-decomposition of H 
    """
    N = len(X)
    l,d = X[0].shape
    p = Y.shape[1]
    grad = np.zeros(cores[i].shape)
    for xs,y in zip(X,Y):
        dfdg = TT_tenvecs_product_omit_one(cores, xs, i)
        if 0 < i < l:
            dfdg = dfdg.squeeze(0)
        f = TT_tenvecs_product(cores,xs).squeeze()
        #print(dfdg.shape,f.shape,Y[n,:].shape)
        grad_n = np.tensordot(dfdg,f-y,axes=(3,0))
        grad_n = grad_n.squeeze(-1) if i < l else grad_n.squeeze(0)
        #print(grad_n.shape)
        grad += grad_n
    return grad



def _rightorth(a, b):
    """
    right orthonormalisation of core a. After this we have
    np.tensordot(a,a,axes=((0,1),(0,1))) = I
    while
    np.tensordot(a,b,axes=(2,0) 
    remains unchanged
    """
    adim = np.array(a.shape)
    bdim = np.array(b.shape)
    cr = np.reshape(a, (adim[0]*adim[1], adim[2]), order='F')
    cr2 = np.reshape(b, (bdim[0],bdim[1]*bdim[2]), order='F')
    [cr, R] = np.linalg.qr(cr)
    cr2 = np.dot(R, cr2)

    adim[2] = cr.shape[1]
    bdim[0] = cr2.shape[0]
    a = np.reshape(cr, adim, order='F')
    b = np.reshape(cr2, bdim, order='F')
    return a,b

def _leftorth(a,b):
    """
    left orthonormalisation of core a. After this we have
    np.tensordot(b,b,axes=((1,2),(1,2))) = I
    while
    np.tensordot(a,b,axes=(2,0) 
    remains unchanged
    """
    adim = a.shape
    bdim = b.shape
    cr = np.reshape(b, (bdim[0],bdim[1]*bdim[2]),order='F').T
    [cr, R] = np.linalg.qr(cr)
    cr2 = np.reshape(a, (adim[0]*adim[1],adim[2]),order='F').T
    cr2 = np.dot(R, cr2)

    a = np.reshape(cr2.T, adim, order='F')
    b = np.reshape(cr.T, bdim, order='F')
    return a,b

def _orthcores(a,dir='right'):
    """
    right (resp. left) orthonormalize all cores of a except for
    the last (resp. first) one 
    """
    if isinstance(a,list):
        d = len(a)
        ry = [X.shape[0] for X in a] + [1]
        L = a
        #a = tt.vector.from_list(L)
    elif isinstance(a,tt.vector):
        d = a.d
        ry = a.r
        L = tt.vector.to_list(a)
    else:
        raise NotImplementedError()

    if dir=='right':
        for i in range(d - 1):
            L[i:i+2] = _rightorth(L[i], L[i + 1])
            ry[i + 1] = L[i].shape[2]
    elif dir=='left':
        for i in range(d-1,0,-1):
            L[i-1:i+1] = _leftorth(L[i-1],L[i])
            ry[i] = L[i-1].shape[2]

    return tt.vector.from_list(L) if isinstance(a,tt.vector) else L



def TT_factorisation_pinv(cores,n_row_modes):
    """
    assuming cores are the cores of the TT decomposition of H and n_row modes is the number
    of modes corresponding to 'prefixes' (i.e. l for H^{(2l)}), this returns the cores
    of the TT-decompositions of the pseudo-inverses of P and S, where P and S are such that
    H = PS
    """
    for i in range(n_row_modes):
        cores[i:i+2] = _rightorth(cores[i],cores[i+1])
    #print(len(cores[n_row_modes:]))
    print(isinstance(cores, list))
    S_cores = _orthcores(cores[n_row_modes:], dir='left')
    P_cores = cores[:n_row_modes]

    c = S_cores[0]
    U,s,V = np.linalg.svd(c.reshape((c.shape[0],np.prod(c.shape[1:])),order='F'),full_matrices=False)
    P_cores[-1] = np.tensordot(P_cores[-1],U,axes=(2,0))
    S_cores[0] = (np.diag(1./s).dot(V)).reshape(c.shape,order='F')

    return P_cores,S_cores

def TT_spectral_learning(H_2l_cores, H_2l1_cores, H_l_cores):
    l = len(H_l_cores) - 1
    P_cores, S_cores = TT_factorisation_pinv(H_2l_cores,l)
    
    # compute alpha
    alpha = TT_product(H_l_cores,S_cores)

    # compute A
    A_left = TT_product(H_2l1_cores[:l],P_cores)
    A_right = TT_product(H_2l1_cores[l+1:],S_cores)
    A = np.tensordot(A_left,H_2l1_cores[l],(2,0))
    A = np.tensordot(A,A_right,(4,0))

    # compute  omega
    omega = TT_product(H_l_cores[:l],P_cores)
    omega = np.tensordot(omega,H_l_cores[l],(2,0))


    model = LinRNN(alpha.squeeze(), A.squeeze(), omega.squeeze())
    return model
















