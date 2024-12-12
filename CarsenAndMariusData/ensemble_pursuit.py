import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.distance import pdist
import warnings

# Suppress potential warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Load the Data
path = "/home/maria/Documents/CarsenMariusData/6845348/natimg2800_M161025_MP030_2017-05-29.mat"
mat = scipy.io.loadmat(path)

# Extract the 'events' matrix
# Assuming 'stim' is structured as described: stim[0][0][1].T
events = mat['stim'][0][0][1].T
num_neurons, num_time_points = events.shape
print(f"Events shape: {events.shape}")  # Should print (14062, 5658)

# 2. Define Parameters
time_points_list = list(range(500, 5501, 500))  # [500, 1000, 1500, ..., 5500]
average_distances = []  # To store average distances for each time point

# 3. Iterate Over Specified Time Points
for n in time_points_list:
    print(f"\nProcessing first {n} time points...")
    start_time = time.time()
    
    # Extract the first 'n' time points
    data_subset = events[:, :n]  # Shape: (14062, n)
    
    # Optional: Normalize or Standardize the Data
    # Uncomment the following lines if normalization is desired
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # data_subset = scaler.fit_transform(data_subset)
    
    # Compute Pairwise Euclidean Distances
    print("Computing pairwise distances...")
    try:
        # pdist returns a condensed distance matrix
        distances = pdist(data_subset, metric='euclidean')
    except MemoryError:
        print("MemoryError: Unable to compute all pairwise distances. Consider using sampling.")
        break  # Exit the loop or handle accordingly
    
    # Calculate the Average Distance
    avg_distance = distances.mean()
    average_distances.append(avg_distance)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Average pairwise distance: {avg_distance:.4f}")
    print(f"Time taken: {elapsed:.2f} seconds")

# 4. Plot the Results
plt.figure(figsize=(10, 6))
plt.plot(time_points_list, average_distances, marker='o', linestyle='-', color='b')
plt.title('Average Pairwise Distance Between Neurons vs. Number of Time Points')
plt.xlabel('Number of Time Points')
plt.ylabel('Average Euclidean Distance')
plt.grid(True)
plt.xticks(time_points_list)
plt.tight_layout()
plt.show()
