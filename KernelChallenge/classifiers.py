import numpy as np
from scipy import optimize

from KernelChallenge.kernels import gramMatrix


class KernelSVC:

    def __init__(self, C, kernel, epsilon=1e-3):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None

    def fit(self, X, y):
        # You might define here any variable needed for the rest of the code
        N = len(y)
        # k = self.kernel(X, X)
        features_X = self.kernel.fit_subtree(X)
        k = gramMatrix(features_X, features_X)
        M = np.einsum('i,j,ij->ij', y, y, k)
        u = np.ones(N)
        A = np.vstack((np.diag(-np.ones(N)), np.diag(np.ones(N))))
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))

        # Lagrange dual problem
        def loss(alpha):
            return alpha.T @ M @ alpha / 2 - \
                alpha.sum()  # dual loss

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return M @ alpha - \
                u  # --partial derivative of the dual loss wrt alpha

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        def fun_eq(alpha):
            return np.dot(
                alpha, y)  # --function defining the equality constraint

        def jac_eq(alpha):
            return y  # --jacobian wrt alpha of the  equality constraint

        def fun_ineq(alpha):
            return b - A @ alpha  # function defining the inequality constraint

        def jac_ineq(alpha):
            return -A  # -jacobian wrt alpha of the  inequality constraint

        constraints = ({'type': 'eq', 'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq',
                        'fun': fun_ineq,
                        'jac': jac_ineq})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints)
        self.alpha = optRes.x

        # Assign the required attributes

        active_idx = np.where(self.alpha > self.epsilon)
        margin_idx = (self.alpha > self.epsilon) * \
            (self.C - self.alpha > self.epsilon)
        self.active_alphaY = self.alpha[active_idx] * y[active_idx]
        self.active = X[active_idx]
        # A matrix with each row corresponding to a point that falls on the
        # margin -
        self.support = X[margin_idx]

        f = self.separating_function(self.active)
        self.b = (y[active_idx] - f).mean()  # offset of the classifier-

    # Implementation of the separting function $f$

    def separating_function(self, x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        features_X = self.kernel.predict(self.active)
        features_x = self.kernel.predict(x)
        K_Xx = gramMatrix(features_x, features_X)
        return np.einsum('i, ji -> j', self.active_alphaY, K_Xx)

    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d + self.b > 0) - 1
