
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
with open('/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/google_vit-base-patch16-224-in21k_embeddings.pkl', 'rb') as file:
    embs = pickle.load(file)


# Assuming natural_movie_one is your original array with shape (900, 768)
natural_movie_one = embs['natural_movie_one']


# Standardize the data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(natural_movie_one)

# Apply PCA to reduce the dimensionality and whiten the data
pca = PCA(whiten=True)
whitened_data = pca.fit_transform(standardized_data)

# Select a subset of components for visualization (optional)
# In this case, reducing to 3D for plotting purposes
pca_3d = PCA(n_components=3)
whitened_data_3d = pca_3d.fit_transform(whitened_data)
print(whitened_data.shape)

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the 3D whitened data
ax.scatter(whitened_data_3d[:, 0], whitened_data_3d[:, 1], whitened_data_3d[:, 2], c='blue', s=50, alpha=0.6)

# Label the axes
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Set the title
ax.set_title('3D PCA Projection of Whitened Data')

# Show the plot
plt.show()
