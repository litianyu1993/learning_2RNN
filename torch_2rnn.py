import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def torch_kahti_rao(C, B):
    temp = []
    for i in range(len(B)):
        if len(C) == 1:
            temp.append(torch.ger(C[0], B[i]))
        else:
            temp.append(torch.ger(C[i], B[i]))
    temp = torch.stack(temp)
    return temp

class Net(nn.Module):

    def __init__(self, num_units, input_dim, output_dim, A = None, alpha=None, Omega=None, if_initialize = False):
        super(Net, self).__init__()
        self.num_units = num_units
        self.input_dim = input_dim
        self.output_dim = output_dim

        #torch.set_default_dtype(torch.float64)
        #torch.set_printoptions(precision=10)
        self.A = nn.Parameter(torch.rand(num_units*input_dim, num_units, requires_grad= True))
        self.alpha = nn.Parameter(torch.rand(1, num_units, requires_grad = False))
        self.Omega = nn.Parameter(torch.rand(num_units, output_dim, requires_grad = False))

        if if_initialize:
            print(A.shape)
            self.A = nn.Parameter(torch.from_numpy(A.reshape(num_units*input_dim, num_units)).float(), requires_grad=True)
            self.alpha = nn.Parameter(torch.from_numpy(alpha).float(), requires_grad=True)
            self.Omega = nn.Parameter(torch.from_numpy(Omega.reshape(num_units, output_dim)).float(), requires_grad=True)



    def forward(self, x):
        x = x.float()
        temp = torch_kahti_rao(self.alpha, x[:, 0, :]).view(x.shape[0], -1)
        for i in range(1, x.shape[1]):
            temp = temp @ self.A
            temp = torch_kahti_rao(temp, x[:, i, :]).view(x.shape[0], -1)
        temp = temp @ self.A
        temp = temp @ self.Omega
        return temp

