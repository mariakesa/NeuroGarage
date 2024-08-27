import pickle
import numpy as np
with open('/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/google_vit-base-patch16-224-in21k_embeddings.pkl', 'rb') as file:
    embs = pickle.load(file)


# Assuming natural_movie_one is your original array with shape (900, 768)
natural_movie_one = embs['natural_movie_one'] # Replace this with your actual data

# Generate a random projection matrix of shape (768, 768)
random_projection_matrix = np.random.randn(768, 768)

# Perform the random projection
projected_data = np.dot(natural_movie_one, random_projection_matrix)

# The shape of projected_data should be (900, 768)
print("Shape of the projected data:", projected_data.shape)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=3)
projected_data_3d = pca.fit_transform(projected_data)

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the 3D projected data
ax.scatter(projected_data_3d[:, 0], projected_data_3d[:, 1], projected_data_3d[:, 2], c='blue', s=50, alpha=0.6)

# Label the axes
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Set the title
ax.set_title('3D PCA Projection of Natural Movie Data')

# Show the plot
plt.show()