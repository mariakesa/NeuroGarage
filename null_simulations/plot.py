import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

nn=np.load('nn.npy')
# Assume nn is the learned embedding variable of shape (num_points, embedding_dim)
# Here, num_points can be the number of neurons, and embedding_dim is the dimension of the embedding

# For the 3D plot, we'll only use the first 3 dimensions of nn
# Ensure nn has at least 3 dimensions to plot
if nn.shape[1] >= 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the first 3 dimensions of the embedding
    ax.scatter(nn[:, 0], nn[:, 1], nn[:, 2], c='b', marker='o', s=50, alpha=0.8)

    # Set labels for axes
    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.set_zlabel('Embedding Dimension 3')

    # Set plot title
    ax.set_title('3D Visualization of Learned Embedding')

    # Show the plot
    plt.show()
else:
    print("The embedding nn has fewer than 3 dimensions. Cannot plot in 3D.")

    