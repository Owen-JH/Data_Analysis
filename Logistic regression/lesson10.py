%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

file_path1 = "C:\大学\数据分析\lesson10\lesson10.txt"

try:
    X = []
    Y = []
    with open(file_path1, "r") as file:
        for line in file:
            parts = line.split(',')
            if len(parts) == 2:
                X.append(float(parts[0]))
                Y.append(float(parts[1]))

except FileNotFoundError:
    print(f"文件 '{file_path1}' 未找到.")
except Exception as e:
    print(f"发生错误: {e}")
X = np.array(X).reshape(-1, 1)
Y = np.array(Y).reshape(-1, 1)

degree = 5
X_poly = np.hstack([X ** i for i in range(1, degree + 1)])  # 创建多项式特征
print(X_poly)

def gradient_descent(X, Y, alpha, n_iterations, min_error, lambda_reg):
    m, n = X.shape
    theta = np.random.randn(n, 1)

    for iteration in range(n_iterations):
        prediction = X.dot(theta)
        error = prediction - Y
        gradient = (1 / m) * X.T.dot(error) + (lambda_reg / m) * theta
        theta = theta*(1-alpha*(lambda_reg/m))-alpha * gradient

        mse = (1 / (2 * m)) * np.sum(error ** 2)
        mse_history.append(mse)
        if mse < min_error:
            break

    return theta, mse

lambda_reg = 0
learning_rate = 0.001
n_iterations = 50000
min_error = 1e-6
mse_history = []
optimal_theta, mse = gradient_descent(X_poly, Y, learning_rate, n_iterations, min_error, lambda_reg)

print("Optimal Parameters (theta):")
print(optimal_theta)
print("Final Mean Squared Error (MSE):")
print(mse)

plt.scatter(X, Y, label="Data Points")
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = np.hstack([X_range ** i for i in range(1, degree + 1)])
Y_fit = X_range_poly.dot(optimal_theta)
plt.plot(X_range, Y_fit, color='red', label="Fitted Curve")
plt.xlabel("X")
plt.ylabel("Y")
plt.title('Name:OuYang JH,λ=0')
plt.legend()
plt.show()
-----------------------------------------------------------------------------------------
file_path2 = "C:\大学\数据分析\lesson10\microchips.txt"
try:
    X1 = []
    X2 = []
    Y = []

    with open(file_path2, "r") as file:
        for line in file:
            parts = line.split(',')
            if len(parts) == 3:
                X1.append(float(parts[0]))
                X2.append(float(parts[1]))
                Y.append(float(parts[2]))
                X = np.column_stack((X1, X2))

except FileNotFoundError:
    print(f"文件 '{file_path2}' 未找到.")
except Exception as e:
    print(f"发生错误: {e}")
X = np.array(X)
Y = np.array(Y)

degree = 27
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

alpha = 0
model = Ridge(alpha=alpha)
model.fit(X_poly, Y)

Y_pred = model.predict(X_poly)
mse = mean_squared_error(Y, Y_pred)
print("Mean Squared Error (MSE):", mse)

X_range1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
X_range2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
X1, X2 = np.meshgrid(X_range1, X_range2)
X_grid = np.c_[X1.ravel(), X2.ravel()]
X_grid_poly = poly.transform(X_grid)
Y_grid = model.predict(X_grid_poly)
Y_grid = Y_grid.reshape(X1.shape)

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, marker='o', label='y=0 (Circle), y=1 (Cross)')
plt.contour(X1, X2, Y_grid, levels=[0.5], colors='red')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Name:OuYang JH,λ=0')
plt.legend()
plt.show()





