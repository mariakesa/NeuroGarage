import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

# Load the existing embeddings from the pickle file
with open('/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/google_vit-base-patch16-224-in21k_embeddings.pkl', 'rb') as file:
    embs = pickle.load(file)

# Extract the original arrays
natural_movie_one = embs['natural_movie_one']
natural_movie_two = embs['natural_movie_two']
natural_movie_three = embs['natural_movie_three']

# Create a dictionary with the original arrays
dct = {
    'natural_movie_one': natural_movie_one,
    'natural_movie_two': natural_movie_two,
    'natural_movie_three': natural_movie_three
}

# Standardize the data and apply PCA
for i in dct.keys():
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(dct[i])

    # Apply PCA to reduce the dimensionality and whiten the data
    pca = PCA(whiten=True)
    whitened_data = pca.fit_transform(standardized_data)
    
    # Update the embeddings dictionary with the new whitened data
    embs[i] = whitened_data

# Overwrite the pickle file with the new dictionary
with open('/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/google_vit-base-patch16-224-in21k_embeddings.pkl', 'wb') as file:
    pickle.dump(embs, file)

print("Pickle file overwritten with new embeddings.")
