
from lib import EnsemblePursuit
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.distance import pdist
import warnings
from sklearn.linear_model import Ridge
images='/home/maria/Documents/CarsenMariusData/6845348/images_natimg2800_all.mat'
path = "/home/maria/Documents/CarsenMariusData/6845348/natimg2800_M161025_MP030_2017-05-29.mat"
mat = scipy.io.loadmat(path)
images=scipy.io.loadmat(images)

components=np.load('ep_components.npy')

events=mat['stim'][0][0][1].T
sequences=mat['stim'][0][0][2].flatten()-1
images_nonempty=events[:,sequences!=2800]
sequences=sequences[sequences!=2800]
components=components[sequences]
print(images['imgs'].shape)
print(components.shape)
print(sequences.shape)
stims=np.array([images['imgs'][:,90:180,i] for i in range(2800)])

from sklearn.decomposition import PCA

pca=PCA(n_components=100)

pca.fit(stims.reshape(2800,-1))

#reconstruct the stims here
#reconstructed_stims

print(pca.components_.shape)

ridge=Ridge(alpha=1.0)

ridge.fit(reconstruted_stims,components)
