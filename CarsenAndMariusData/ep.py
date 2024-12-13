from lib import EnsemblePursuit
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.distance import pdist
import warnings

ep=EnsemblePursuit(n_components=100,lam=0.01,n_kmeans=100)


# 1. Load the Data
path = "/home/maria/Documents/CarsenMariusData/6845348/natimg2800_M161025_MP030_2017-05-29.mat"
mat = scipy.io.loadmat(path)

# Extract the 'events' matrix
# Assuming 'stim' is structured as described: stim[0][0][1].T
events = mat['stim'][0][0][1]#[:100,:]

print(events.shape)

ep.fit(events)

print(ep.components_.shape)

np.save('ep_components.npy',ep.components_)

