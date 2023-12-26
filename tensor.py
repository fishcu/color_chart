import timeit

import numpy as np
from scipy.optimize import minimize

N = 24
M = 18
num_dim = 3

Y = np.random.normal(size=(N, num_dim))
X = np.random.normal(size=(N, num_dim))

D = np.random.normal(size=(N, M, num_dim)) * 0.1

# print("X:", X)
# print("Y:", Y)
# print("D:", D)

# Prefer smaller weights
lambda_s = 1.0



def analytic_solution():
    lhs = lambda_s * np.identity(M)
    rhs = np.zeros((M,))
    for d in range(num_dim):
        lhs += np.matmul(np.transpose(D[:, :, d]), D[:, :, d])
        rhs += np.dot(np.transpose(D[:, :, d]), Y[:, d] - X[:, d])
    return np.linalg.solve(lhs, rhs)


def loss(w):
    dw = np.zeros((N, num_dim))
    for d in range(num_dim):
        dw[:, d] = np.matmul(D[:, :, d], w)
    return np.sum((Y - X - dw)**2) + lambda_s * np.sum(w**2)




def approx_round(x, deg=5.0):
    rounded = x.round()
    return rounded + np.power(x - rounded, deg)


def numeric_solution():
    # Initial guess for W
    initial_w = np.zeros(M)
    result = minimize(loss, initial_w)
    return result.x


def penalized_loss(w, lambda_i):
    dw = np.zeros((N, num_dim))
    for d in range(num_dim):
        dw[:, d] = np.matmul(D[:, :, d], w)
    return np.sum((Y - X - dw)**2) + lambda_s * np.sum(w**2) + lambda_i * np.sum((w - w.round())**2)


def penalty_method():
    lambda_i = 0.001
    num_iter = 100
    gamma = 1.2
    w = np.zeros(M)
    for i in range(num_iter):
        w = minimize(penalized_loss, w, args=lambda_i).x
        lambda_i *= gamma
        # print(f"w at step {i}: {w}\n")
    return w
        


analytic_w = analytic_solution()
# numeric_w = numeric_solution()

print("analytic w:")
print(analytic_w)
# print("numeric w:")
# print(numeric_w)

print("final losses of each:")
print(loss(analytic_w))
# print(loss(numeric_w))

# n_prof_runs = 100
# analytic_time = timeit.timeit(analytic_solution, number=n_prof_runs)
# print(f"analytic runtime: {analytic_time / n_prof_runs * 1000.0} ms")
# numeric_time = timeit.timeit(numeric_solution, number=n_prof_runs)
# print(f"numeric runtime: {numeric_time / n_prof_runs * 1000.0} ms")

print("penalty method w:")
penalty_w = penalty_method()
print(penalty_w.round(decimals = 4))
# print(penalty_w.round())
print("compare with round of unconstrained solution:")
print(analytic_w.round())