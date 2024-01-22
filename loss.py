import numpy as np


def loss(X, Y, D, w):
    dw = np.zeros((Y.shape[0], 3))
    for d in range(3):
        dw[:, d] = np.matmul(D[:, :, d], w)
    return np.sum((Y - X - dw)**2)  # + lambda_s * np.sum(w**2)


def loss_no_correction(X, Y):
    return np.sum((Y - X)**2) / Y.shape[0]
