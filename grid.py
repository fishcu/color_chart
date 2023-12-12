import matplotlib.pyplot as plt
import numpy as np

# Generate a dense 2D grid
dense_grid_size = 101
dense_x = np.linspace(0, 1, dense_grid_size)
dense_y = np.linspace(0, 1, dense_grid_size)
dense_x_grid, dense_y_grid = np.meshgrid(dense_x, dense_y)

# Subsample the dense grid
subsample_factor = 5
x_points = dense_x_grid.flatten()[::subsample_factor]
y_points = dense_y_grid.flatten()[::subsample_factor]

# Connect each point [i, j] with [i+1, j] and [i, j+1], but only for subsampled points
for i in range(dense_grid_size):
    for j in range(dense_grid_size):
        # Connect with [i+1, j] if j is a subsampled point
        if j % subsample_factor == 0 and i < dense_grid_size - 1:
            plt.plot([dense_x_grid[i, j], dense_x_grid[i + 1, j]], [dense_y_grid[i, j],
                     dense_y_grid[i + 1, j]], color='black', linestyle='-', linewidth=1)

        # Connect with [i, j+1] if i is a subsampled point
        if i % subsample_factor == 0 and j < dense_grid_size - 1:
            plt.plot([dense_x_grid[i, j], dense_x_grid[i, j + 1]], [dense_y_grid[i, j],
                     dense_y_grid[i, j + 1]], color='black', linestyle='-', linewidth=1)

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Grid with Connected Lines')

# Show legend
plt.legend()

# Display the plot
plt.show()
