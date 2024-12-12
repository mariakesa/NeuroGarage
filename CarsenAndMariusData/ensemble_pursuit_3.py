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
# Adjust the indices if your data structure differs
events = mat['stim'][0][0][1].T
num_neurons, num_time_points = events.shape
print(f"Events shape: {events.shape}")  # Expected: (14062, 5658)

# 2. Define Parameters
time_points_list = list(range(500, 5501, 500))  # [500, 1000, 1500, ..., 5500]
average_distances = []  # To store mean distances
std_distances = []      # To store standard deviations
cv_distances = []       # To store Coefficient of Variation (CV)

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
    
    # Calculate the Mean Distance and Standard Deviation
    avg_distance = distances.mean()
    std_distance = distances.std()
    average_distances.append(avg_distance)
    std_distances.append(std_distance)
    
    # Calculate Coefficient of Variation (CV)
    cv = std_distance / avg_distance
    cv_distances.append(cv)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Average pairwise distance: {avg_distance:.4f}")
    print(f"Standard deviation of distances: {std_distance:.4f}")
    print(f"Coefficient of Variation (CV): {cv:.4f}")
    print(f"Time taken: {elapsed:.2f} seconds")

# 4. Plot the Results
plt.figure(figsize=(12, 7))

# Plot Mean Distance
plt.plot(time_points_list, average_distances, marker='o', linestyle='-', color='b', label='Mean Distance')

# Plot Shaded Area for ±1 Std Dev
plt.fill_between(time_points_list,
                 np.array(average_distances) - np.array(std_distances),
                 np.array(average_distances) + np.array(std_distances),
                 color='b', alpha=0.2, label='±1 Std Dev')

# Create a twin axis to plot CV
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot Coefficient of Variation
ax2.plot(time_points_list, cv_distances, marker='s', linestyle='--', color='r', label='Coefficient of Variation (CV)')

# Formatting the twin plot
ax2.set_ylabel('Coefficient of Variation (CV)', color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')

# Titles and Labels
plt.title('Average Pairwise Distance & Coefficient of Variation vs. Number of Time Points')
ax1.set_xlabel('Number of Time Points')
ax1.set_ylabel('Average Euclidean Distance')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.grid(True)
plt.xticks(time_points_list)
plt.tight_layout()
plt.show()
