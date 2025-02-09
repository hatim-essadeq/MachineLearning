import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from unsupervised_learning.kmeans import run_k_means, init_centroids, find_closest_centroids, compute_centroids

# load data
data = loadmat('data/clustering_data.mat')
print(data)
print(data['X'])
print(data['X'].shape)

# classify points
X = data['X']
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
initial_centroids = np.array([[8, 0], [8, 6], [0, 3]])
initial_centroids = init_centroids(X, 3)
print(initial_centroids)

idx = find_closest_centroids(X, initial_centroids)
print(idx)

# calculate new centroid
c = compute_centroids(X, idx, 3)
print(c)

for x in range(6):
    # apply k means
    idx, centroids = run_k_means(X, initial_centroids, x)
    print(idx)
    print()
    print(centroids)

    # draw it
    cluster1 = X[np.where(idx == 0)[0], :]
    cluster2 = X[np.where(idx == 1)[0], :]
    cluster3 = X[np.where(idx == 2)[0], :]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
    ax.scatter(centroids[0, 0], centroids[0, 1], s=300, color='r')

    ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
    ax.scatter(centroids[1, 0], centroids[1, 1], s=300, color='g')

    ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
    ax.scatter(centroids[2, 0], centroids[2, 1], s=300, color='b')

    ax.legend()