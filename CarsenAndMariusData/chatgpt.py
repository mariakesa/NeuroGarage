from lib import EnsemblePursuit
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.distance import pdist
import warnings
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# File paths
images_path = '/home/maria/Documents/CarsenMariusData/6845348/images_natimg2800_all.mat'
stim_path = "/home/maria/Documents/CarsenMariusData/6845348/natimg2800_M161025_MP030_2017-05-29.mat"
components_path = 'ep_components.npy'

# Load data
images = scipy.io.loadmat(images_path)
mat = scipy.io.loadmat(stim_path)
components = np.load(components_path)

# Extract relevant data from the .mat files
events = mat['stim'][0][0][1].T  # Assuming shape (n_events, ...)
sequences = mat['stim'][0][0][2].flatten() - 1  # Adjusting indices if necessary

# Filter out sequences equal to 2800
valid_mask = sequences != 2800
sequences = sequences[valid_mask]
images_nonempty = events[:, valid_mask]
components = components[sequences]

# Debugging prints to check shapes
print(f"Images shape: {images['imgs'].shape}")          # Expected: (height, width, 2800)
print(f"Components shape: {components.shape}")         # Expected: (n_valid_sequences, n_components)
print(f"Sequences shape: {sequences.shape}")           # Expected: (n_valid_sequences,)

# Prepare stimuli: Extract a subset of the image (e.g., columns 90 to 180)
# Adjust the slicing based on actual data dimensions
stims = images['imgs'][:, 90:180, :2800]  # Shape: (height, selected_width, 2800)
n_stims = stims.shape[2]

# Reshape stimuli for PCA: (n_samples, n_features)
stims_reshaped = stims.reshape(stims.shape[0] * stims.shape[1], n_stims).T  # Shape: (2800, height*selected_width)

# Apply PCA
n_pca_components = 100  # You can adjust this number based on variance explained
pca = PCA(n_components=n_pca_components, random_state=42)
pca.fit(stims_reshaped)

# Reconstruct the stimuli from the PCA components
stims_transformed = pca.transform(stims_reshaped)            # Shape: (2800, n_pca_components)
stims_reconstructed = pca.inverse_transform(stims_transformed)  # Shape: (2800, height*selected_width)

print(f"PCA components shape: {pca.components_.shape}")     # Expected: (100, height*selected_width)
print(stims_reconstructed.shape)  # Expected: (2800, height*selected_width)
ims=stims_reconstructed[sequences]
plt.imshow(stims_reconstructed[0].reshape(stims.shape[0], stims.shape[1]), cmap='gray')
plt.show()
# Fit Ridge Regression
ridge_alpha = 1.0  # Regularization strength; adjust as needed
ridge = Ridge(alpha=ridge_alpha, random_state=42)
ridge.fit(ims, components)  # Assuming components shape: (2800, n_receptive_fields)

print("Ridge regression model fitted successfully.")

# Extract regression coefficients (receptive fields)
# Each row in ridge.coef_ corresponds to a receptive field
receptive_fields = ridge.coef_  # Shape: (n_receptive_fields, height*selected_width)

# Determine the number of receptive fields to display
n_receptive_fields = receptive_fields.shape[0]
print(f"Number of receptive fields: {n_receptive_fields}")

# Determine the layout for plotting (e.g., grid size)
n_cols = 5
n_rows = int(np.ceil(n_receptive_fields / n_cols))

# Plot each receptive field
plt.figure(figsize=(15, 3 * n_rows))
for i in range(n_receptive_fields):
    plt.subplot(n_rows, n_cols, i + 1)
    rf_image = receptive_fields[i].reshape(stims.shape[0], stims.shape[1])
    plt.imshow(rf_image, cmap='viridis', aspect='auto')
    plt.title(f'Receptive Field {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Optionally, visualize explained variance by PCA
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100)
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()
