#####################################################
### Generating synthetic data with a 2-RNN dynamic ###
######################################################
import numpy as np
import tt
from LinRNN import LinRNN


def generate_random_LinRNN(num_states,input_dim,output_dim, alpha_variance = 1., Omega_variance = 1., A_variance = 1.):
    alpha = np.random.normal(0, alpha_variance, num_states)
    Omega = np.random.normal(0, Omega_variance, [num_states, output_dim])
    A = np.random.normal(0, A_variance, [num_states, input_dim, num_states])

    mdl = LinRNN(alpha,A,Omega)
    X,y = generate_data(mdl, 1000, 4,noise_variance=0.)

    mdl.alpha /= (np.mean(y**2)*10)
    return mdl



def generate_data(mdl, N_samples, seq_length,noise_variance=0.):
    X = []
    Y = []
    for i in range(N_samples):
        X.append(np.random.normal(0, 1, [seq_length, mdl.input_dim]))
        Y.append(mdl.predict(X[-1]) + np.random.normal(0, noise_variance))

    return np.asarray(X),np.asarray(Y).squeeze()




##### The remaining is only here for backward compatibility but should be removed at some point... #####
class synthetic_data_generator():
    def __init__(self, num_states, num_examples, input_dim, output_dim, noise_variance=0.):
        self.num_states = num_states
        self.num_examples = num_examples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_variance = noise_variance
        self.initialize_everything()

        self.target = LinRNN(self.alpha, self.A, self.omega)

    def initialize_alpha(self):
        alpha = np.random.normal(0, 1, self.num_states)
        self.alpha = alpha

    def initialize_omega(self):
        omega = np.random.normal(0, 1, [self.num_states, self.output_dim])
        self.omega = omega

    def initialize_A(self):
        A = np.random.normal(0, 1, [self.num_states, self.input_dim, self.num_states])
        self.A = A

    def initialize_everything(self):
        self.initialize_A()
        self.initialize_alpha()
        self.initialize_omega()

    def generate_training_data(self, length):

        X = []
        Y = []
        for i in range(self.num_examples):
            X.append(np.random.normal(0,1,[length,self.input_dim]))
            Y.append(self.target.predict(X[-1]) + np.random.normal(0, self.noise_variance))
        Y = np.asarray(Y)
        return np.asarray(X),Y.squeeze()



