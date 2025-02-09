import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from unsupervised_learning.kmeans import init_centroids, run_k_means, find_closest_centroids, project_data, \
    recover_data, pca


# we need to compress the image
image_data = loadmat('./data/bird_small.mat')

print(image_data)

A = image_data['A']
print(A.shape)
plt.imshow(A)

# normalize value ranges
A = A / 255.

# reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
print(X.shape)

# randomly initialize the centroids
initial_centroids = init_centroids(X, 16)
print(initial_centroids)

## run the algorithm
idx, centroids = run_k_means(X, initial_centroids, 10)

## get the closest centroids one last time
idx = find_closest_centroids(X, centroids)

## map each pixel to the centroid value
X_recovered = centroids[idx.astype(int), :]

## reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

plt.imshow(X_recovered)

# -----------------------------------------------------------------------

# Apply PCA
data = loadmat('data/pca_data.mat')
X = data['X']
print(X.shape)
print(X)
print()

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:, 0], X[:, 1])

U, S, V = pca(X)
print(U)
print()
print(S)
print()
print(V)

Z = project_data(X, U, 1)
print(Z)

X_recovered = recover_data(Z, U, 1)
print(X_recovered)
print(X_recovered.shape)

# -----------------------------------------------------------------------

# Apply PCA on faces

faces = loadmat('data/pca_faces.mat')
X = faces['X']
print(X.shape)
plt.imshow(X)

# show one face
face = np.reshape(X[41, :], (32, 32))
plt.imshow(face)

U, S, V = pca(X)
Z = project_data(X, U, 100)

X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[41, :], (32, 32))
plt.imshow(face)