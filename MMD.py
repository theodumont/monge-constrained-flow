import numpy as np

class MMD(object):
    def __init__(self,k):
        self.k = k

    def K(self,X,Y):
        scal = np.dot(X, Y.T)  # Shape (N, M)
        X2 = np.sum(X**2, axis=1, keepdims=True)  # Shape (N, 1)
        Y2 = np.sum(Y**2, axis=1, keepdims=True).T  # Shape (1, M)
        d2 = np.maximum(X2 + Y2 - 2 * scal, 0)  # Ensures non-negative distances (like ReLU)
        return self.k(d2)

    def eval(self,X,Y):
        n = X.shape[0]
        m = Y.shape[0]
        return 1/n**2 * np.sum(self.K(X, X)) + 1/m**2 * np.sum(self.K(Y, Y)) - 2/(n*m) * np.sum(self.K(X, Y))


def ED_k():
    return lambda d2: -np.sqrt(d2)


def get_error_MMD(rho_infty, gamma, k):
    mmd = MMD(k)
    return mmd.eval(rho_infty, gamma).item()
