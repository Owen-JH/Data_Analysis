%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
filename1 = 'mathmetician.xls'
data1 = pd.read_excel(filename1)
n = data1.shape[0]
p = data1.shape[1]
ones = np.ones((n, 1))
X = data1[['x1', 'x2', 'x3']].values
X = np.hstack((ones, X))
y = data1['y'].values.reshape(n, 1)

def gradient_descent(X, y, alpha, n_iterations, min_error):
    m = len(X)
    theta = np.random.randn(4, 1)

    for iteration in range(n_iterations):
        sum_theta1 = np.zeros((1, 4))
        sum_theta2 = 0
        for k in range(m):
            prediction = np.matmul(X[k, :], theta)
            error = prediction - y[k, 0]
            sum_theta1 += error * X[k, :]
            sum_theta2 += error ** 2
        theta -= alpha * (1 / m) * sum_theta1.T

        mse = (1 / (2 * m)) * sum_theta2
        mse_history.append(mse)
        if mse < min_error:
            break

    return theta, mse

learning_rate = 0.001
n_iterations = 270000
min_error = 1e-6
mse_history = []
optimal_theta, mse = gradient_descent(X, y, learning_rate, n_iterations, min_error)

plt.plot(range(len(mse_history)), mse_history)
plt.xlabel('No. of iterations')
plt.ylabel('Cost function')
plt.title('alpha=0.001,tol=1e-06,k=270000')
plt.ylim(0, 6)
plt.show()

print("θ:", optimal_theta)
print("Cost function:", mse)
-----------------------------------------------------------------------------------------
file_path = "department.txt"

try:
    X1 = []
    X2 = []
    y = []

    with open(file_path, "r") as file:
        for line in file:
            parts = line.split(',')
            if len(parts) == 3:
                X1.append(float(parts[0]))
                X2.append(float(parts[1]))
                y.append(float(parts[2]))
                X = np.column_stack((X1, X2))

except FileNotFoundError:
    print(f"文件 '{file_path}' 未找到.")
except Exception as e:
    print(f"发生错误: {e}")
n = X.shape[0]
ones = np.ones((n, 1))
X = np.hstack((ones, X))

x_data = X[:, 1]
y_data = X[:, 2]

plt.figure()
plt.scatter(x_data, y_data, c=y, cmap=plt.cm.Paired, marker='o', label='y=0 (Circle), y=1 (Cross)')
plt.xlabel('Exam1 score')
plt.ylabel('Exam2 score')
plt.xlim(30, 100)
plt.ylim(30, 100)
plt.legend()
plt.show()
-------------------------------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros((n, 1))
    loss_history = []

    for i in range(num_iterations):
        gradient = np.zeros((1, 3))
        sum_loss = 0
        for k in range(m):
            z = np.matmul(X[k, :], theta)
            h = sigmoid(z)
            gradient += ((h - y[k])*X[k, :]) / m
            theta -= learning_rate * gradient.T
            sum_loss += y[k]*np.log(h)+(1-y[k])*np.log(1-h)

        loss = (-1 / m) * sum_loss
        loss_history.append(loss)

    return theta, loss_history, loss
learning_rate = 0.001
num_iterations = 22500
min_error = 1e-6
optimal_theta, loss_history, loss = logistic_regression(X, y, learning_rate, num_iterations)

plt.plot(range(len(loss_history)), loss_history)
plt.xlabel('No. of iterations')
plt.ylabel('Cost function')
plt.title('alpha=0.001,tol=1e-06,k=22500')
plt.ylim(0, 1)
plt.show()

print("θ:", optimal_theta)
print("Cost function:", loss)
-----------------------------------------------------------------------------------------
plt.figure()
plt.scatter(x_data, y_data, c=y, cmap=plt.cm.Paired, marker='o', label='y=0 (Circle), y=1 (Cross)')
plt.xlabel('Exam1 score')
plt.ylabel('Exam2 score')
plt.xlim(30, 100)
plt.ylim(30, 100)
plt.legend()

x_line = np.linspace(min(x_data), max(x_data), 100)
y_line = (-optimal_theta[1] * x_line - optimal_theta[0]) / optimal_theta[2]
plt.plot(x_line, y_line, label='Decision Boundary', color='red')
plt.show()



