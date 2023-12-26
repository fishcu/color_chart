import numpy as np
import matplotlib.pyplot as plt

def bump_function(r):
    if abs(r) > 1:
        return 0
    return np.exp(-1 / (1 - r**2))

def radial_basis_function(x, center, epsilon=20):
    r = np.linalg.norm(x - center)
    return bump_function(r / epsilon)

def construct_linear_system(A, B):
    M = len(A)
    system_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            system_matrix[i, j] = radial_basis_function(A[i], A[j])
    return system_matrix, B

def mapping_function(x, centers, coefficients, epsilon=20):
    # Calculate the mapped point for each row in x
    mapped_points = []
    for row in x:
        mapped_point = np.sum([radial_basis_function(row, center, epsilon) * coeff for center, coeff in zip(centers, coefficients)])
        mapped_points.append(mapped_point)

    # Stack the mapped points into a 2D array
    return np.vstack(mapped_points)

# Define toy example data for 2D points
A = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]])
B = np.array([[-1, 0], [10, 1], [8, 11], [1, 9], [4, 6]])  # Corresponding values we want to map to

# Construct the linear system
system_matrix, target_values = construct_linear_system(A, B)

# Solve the linear system
coefficients = np.linalg.solve(system_matrix, target_values)

# Visualize the mapping on a regular grid
grid_size = 100
x_grid = np.linspace(-3, 13, grid_size)
y_grid = np.linspace(-3, 13, grid_size)
mapped_grid = np.zeros((grid_size, grid_size, 2))

for i, x in enumerate(x_grid):
    for j, y in enumerate(y_grid):
        mapped_point = mapping_function(np.array([x, y]), A, coefficients, epsilon=5)
        mapped_grid[i, j, :] = mapped_point.squeeze()

# Plot the original and mapped points
plt.scatter(A[:, 0], A[:, 1], label='Original Points', color='blue')
plt.scatter(B[:, 0], B[:, 1], label='Target Points', color='red')
for i in range(grid_size):
    plt.plot(mapped_grid[i, :, 0], mapped_grid[i, :, 1], 'g-', alpha=0.5)
    plt.plot(mapped_grid[:, i, 0], mapped_grid[:, i, 1], 'g-', alpha=0.5)

plt.title('Mapping of 2D Points')
plt.legend()
plt.show()
