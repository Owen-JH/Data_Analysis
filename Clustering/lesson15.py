import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
filename = 'K-means.mat'
data = loadmat(filename)
X = np.array(data['X'])

x = X[:, 0]
y = X[:, 1]
plt.scatter(x, y, color='blue', marker='o', label='Points')
plt.legend()
plt.show()

m = X.shape[0]
K = 3
centroids = np.array([[3.0, 3.0], [6.0, 2.0], [8.0, 5.0]])
distances = np.zeros((m, K))
centroids_history = []
for i in range(5):
    for k in range(m):
        for j in range(K):
            distances[k, j] = np.sum((X[k, :] - centroids[j]) ** 2)
    cluster = np.argmin(distances, axis=1)

    for k in range(K):
        indices = np.where(cluster == k)
        cluster_points = X[indices]
        centroids[k] = np.mean(cluster_points, axis=0)
    centroids_history.append(np.copy(centroids))

for k in range(K):
        indices = np.where(cluster == k)
        cluster_points = X[indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color='red' if k == 0 else 'green' if k == 1 else 'blue', marker='o', label=f'Cluster {k + 1} Points')

        trajectory_x = [history[k][0] for history in centroids_history]
        trajectory_y = [history[k][1] for history in centroids_history]
        plt.plot(trajectory_x, trajectory_y, marker='o', linestyle='-', color='black')

plt.title('Iteration number 10')
plt.legend()
plt.show()





