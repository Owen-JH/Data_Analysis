
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

filename1 = r'C:\大学\数据分析\lesson11\hw11data.mat'
filename2 = r'C:\大学\数据分析\lesson11\hw11weights.mat'
data = loadmat(filename1)
X = data['X']
Y = data['y']

random_indices = np.random.choice(X.shape[0], 100, replace=False)

selected_images = X[random_indices]
selected_labels = Y[random_indices]


def show_images(images, labels):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(20, 20), cmap='gray')
        ax.text(0.05, 0.05, str(labels[i]), transform=ax.transAxes, color='red', fontsize=8, verticalalignment='top')

    plt.show()


show_images(selected_images, selected_labels)

weights = loadmat(filename2)
theta1 = weights['Theta1']
theta2 = weights['Theta2']


def sigmoid(z):
    if z <= -4.6:
        return 0
    elif z >= 4.6:
        return 1
    else:
        return 1 / (1 + np.exp(-z))


ones = np.ones((X.shape[0], 1))
X = np.hstack((ones, X))

matrix1 = np.zeros((5000, 25))
for i in range(X.shape[0]):
    for j in range(theta1.shape[0]):
        z = np.matmul(X[i, :].T, theta1[j, :])
        matrix1[i, j] = sigmoid(z)
ones = np.ones((matrix1.shape[0], 1))
matrix1 = np.hstack((ones, matrix1))

matrix2 = np.zeros((5000, 10))
for i in range(matrix1.shape[0]):
    for j in range(theta2.shape[0]):
        z = np.matmul(matrix1[i, :].T, theta2[j, :])
        matrix2[i, j] = sigmoid(z)

max_indices = np.argmax(matrix2, axis=1)
result_matrix = np.zeros_like(matrix2)

# 将每行最大值的位置设置为1
result_matrix[np.arange(matrix2.shape[0]), max_indices] = 1

Y_encoded = np.eye(10)[Y.flatten() - 1]
row_equality = np.all(result_matrix == Y_encoded, axis=1)
Yi = np.zeros(5000)
for i in range(5000):
    if row_equality[i]:  # 使用布尔值比较
        Yi[i] = 1
    else:
        Yi[i] = 0
total_identical_rows = np.sum(row_equality)
percentage = total_identical_rows/X.shape[0]
print(f"预测精度为: {percentage}")




