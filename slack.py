import numpy as np

M = 7
N = 5
d = 3

x = np.random.normal(size=(M, 3))
y = np.random.normal(size=(M, 3))
D = np.random.normal(size=(M, N, 3))

print("x:", x)
print("y:", y)
print("D:", D)

lam = 2.0
# There's probably a fancier way to do this without a for loop
w = np.zeros((N, 3))
for d in range(3):
    lhs = np.matmul(np.transpose(D[:,:,d]), D[:,:,d]) + lam * np.identity(N)
    rhs = np.dot(np.transpose(D[:,:,d]), y[:,d] - x[:,d])
    w[:,d] = np.linalg.solve(lhs, rhs)

print(w)

# Compute the loss function at this point:
def loss(w):
    # I think the computation of dw is the same as np.einsum('ijk,jk->ik', D, w),
    # but I'm not 100% sure.
    dw = np.zeros((M, 3))
    for d in range(3):
        dw[:,d] = np.matmul(D[:,:,d], w[:,d])
    return np.sum((y - x - dw)**2) + lam * np.sum(w**2)

my_loss = loss(w)
print("Loss: ", my_loss)

# Also check this is really the minimum by searching in a sphere around this:
eps = 0.01
for i in range(20):
    other_loss = loss(w + eps * np.random.normal(size=w.shape))
    print(other_loss - my_loss) # If it's negative, this isn't the minimum!
    