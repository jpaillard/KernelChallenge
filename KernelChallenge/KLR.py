
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from tqdm import tqdm

import cvxpy as cp
from tqdm import tqdm
from scipy.optimize import minimize
from KernelChallenge.kernels import gramMatrix


class KernelLogisticRegression:

    def __init__(self, kernel, reg_param = 0, epsilon=1e-8):
        self.alpha = None
        self.reg_param = reg_param
        self.beta = None
        self.kernel = kernel
        self.eps = epsilon
        

    def fit(self, X,y):
        N = X.shape[0]

        features_X = self.kernel.fit_subtree(X)
        k = gramMatrix(features_X, features_X)
        alpha = np.zeros(N)
        alpha_old = alpha + np.inf
        sig = np.vectorize(sigmoid)
        logp = np.vectorize(logistic_prime)
        logpp = np.vectorize(logistic_prime2)
        while (np.abs(alpha-alpha_old)> self.eps).any(): 
            # Update coefs
            m = k@alpha
            P = np.diag(logp(y*m))
            W = np.diag(logpp(y*m))
            z = m + y/ sig(y*m)

            #Solve Weighted KRR

            sqrt_W =np.sqrt(W)
            
            
            alpha_old = alpha 
            alpha = sqrt_W@np.linalg.inv(sqrt_W@k@sqrt_W+ N*self.reg_param*np.eye(N))@sqrt_W@z

        self.alpha = alpha
        self.features = features_X
        

    def predict(self,X):
        features_pred = self.kernel.predict(X)
        print(features_pred.shape, self.features.shape)
        K_Xx = gramMatrix(features_pred, self.features)
        predictions = sigmoid(np.einsum('i, ij->j', self.alpha, K_Xx.T)) 
        return - predictions # *-1 because inverted prediction on 1 et -1

def logistic(u):
    return np.log(1 + np.exp(-u))

def logistic_prime(u):
    return -sigmoid(-u)

def logistic_prime2(u):
    return sigmoid(u)*sigmoid(-u)

def sigmoid(u):
    return 1 / (1 + np.exp(-u))