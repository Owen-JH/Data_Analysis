%matplotlib inline
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
file_path = "C:\大学\数据分析\lesson8\cities.txt"

try:
    X1 = []
    y1 = []

    with open(file_path, "r") as file:
        for line in file:
            parts = line.split(',')
            if len(parts) == 2:
                X1.append(float(parts[0]))
                y1.append(float(parts[1]))

except FileNotFoundError:
    print(f"文件 '{file_path}' 未找到.")
except Exception as e:
    print(f"发生错误: {e}")

scaler = StandardScaler()
X = scaler.fit_transform(np.array(X1).reshape(-1, 1))
y = scaler.fit_transform(np.array(y1).reshape(-1, 1))

def gradient_descent(X, y, alpha, n_iterations, min_error):
    m = len(X)
    theta = np.random.randn(2, 1)
    theta_history = []
    d = 0

    for iteration in range(n_iterations):
        sum_theta0 = 0
        sum_theta1 = 0
        sum_theta2 = 0
        for k in range(m):
            prediction = theta[0] + theta[1] * X[k]
            error = prediction - y[k]
            sum_theta0 += error
            sum_theta1 += error * X[k]
            sum_theta2 += error**2
        theta[0] = theta[0] - alpha * (1 / m) * sum_theta0
        theta[1] = theta[1] - alpha * (1 / m) * sum_theta1

        theta_history.append(theta.copy())
        mse = (1 / (2 * m)) * sum_theta2
        d += 1
        if mse < min_error:
            break

    return theta, theta_history, d

learning_rate = 0.05
n_iterations = 10000
min_error = 1e-6

optimal_theta, theta_history, d = gradient_descent(X, y, learning_rate, n_iterations, min_error)

print("最终参数 (theta0, theta1):", optimal_theta)

theta_1 = theta_history[0]
theta_500 = theta_history[499]
theta_1000 = theta_history[999]
theta_5000 = theta_history[4999]
theta_last = theta_history[9999]

xk = np.linspace(0, 25, 1000)

yk_1 = theta_1[1] * xk + theta_1[0]
plt.plot(xk, yk_1, label='Line', color='blue')
plt.scatter(X1, y1, label='Data')
plt.xlim(5, 25)
plt.ylim(-5, 25)
plt.xlabel('population')
plt.ylabel('profit')
plt.title('Name:Ouyang JH, k=1')
plt.legend()
plt.show()

theta0 = np.linspace(-2, 2, 100)
theta1 = np.linspace(-2, 2, 100)

Theta0, Theta1 = np.meshgrid(theta0, theta1)

J = np.zeros_like(Theta0)
for i in range(Theta0.shape[0]):
    for j in range(Theta0.shape[1]):
        prediction = Theta0[i, j] + Theta1[i, j] * X
        error = prediction - y
        J[i, j] = np.sum(error**2) / (2 * len(X))

contour = plt.contour(Theta0, Theta1, J, np.logspace(-2, 3, 30), cmap='viridis')
plt.colorbar(contour, label='Cost Function (J)')
plt.scatter(theta_1[0], theta_1[1], c='red', label='θ0, θ1', marker='x')
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.title('Name:Ouyang JH, k=1')
plt.show()

yk_500 = theta_500[1] * xk + theta_500[0]
plt.plot(xk, yk_500, label='Line', color='blue')
plt.scatter(X1, y1, label='Data')
plt.xlim(5, 25)
plt.ylim(-5, 25)
plt.xlabel('population')
plt.ylabel('profit')
plt.title('Name:Ouyang JH, k=500')
plt.legend()
plt.show()

contour = plt.contour(Theta0, Theta1, J, np.logspace(-2, 3, 30), cmap='viridis')
plt.colorbar(contour, label='Cost Function (J)')
plt.scatter(theta_500[0], theta_500[1], c='red', label='θ0, θ1', marker='x')
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.title('Name:Ouyang JH, k=500')
plt.show()

yk_1000 = theta_1000[1] * xk + theta_1000[0]
plt.plot(xk, yk_1000, label='Line', color='blue')
plt.scatter(X1, y1, label='Data')
plt.xlim(5, 25)
plt.ylim(-5, 25)
plt.xlabel('population')
plt.ylabel('profit')
plt.title('Name:Ouyang JH, k=1000')
plt.legend()
plt.show()

contour = plt.contour(Theta0, Theta1, J, np.logspace(-2, 3, 30), cmap='viridis')
plt.colorbar(contour, label='Cost Function (J)')
plt.scatter(theta_1000[0], theta_1000[1], c='red', label='θ0, θ1', marker='x')
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.title('Name:Ouyang JH, k=1000')
plt.show()

yk_5000 = theta_5000[1] * xk + theta_5000[0]
plt.plot(xk, yk_5000, label='Line', color='blue')
plt.scatter(X1, y1, label='Data')
plt.xlim(5, 25)
plt.ylim(-5, 25)
plt.xlabel('population')
plt.ylabel('profit')
plt.title('Name:Ouyang JH, k=5000')
plt.legend()
plt.show()

contour = plt.contour(Theta0, Theta1, J, np.logspace(-2, 3, 30), cmap='viridis')
plt.colorbar(contour, label='Cost Function (J)')
plt.scatter(theta_5000[0], theta_5000[1], c='red', label='θ0, θ1', marker='x')
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.title('Name:Ouyang JH, k=5000')
plt.show()

yk_last = theta_last[1] * xk + theta_last[0]
plt.plot(xk, yk_last, label='Line', color='blue')
plt.scatter(X1, y1, label='Data')
plt.xlim(5, 25)
plt.ylim(-5, 25)
plt.xlabel('population')
plt.ylabel('profit')
plt.title(f'Name:Ouyang JH, k={d}')
plt.legend()
plt.show()

contour = plt.contour(Theta0, Theta1, J, np.logspace(-2, 3, 30), cmap='viridis')
plt.colorbar(contour, label='Cost Function (J)')
plt.scatter(theta_last[0], theta_last[1], c='red', label='θ0, θ1', marker='x')
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.title(f'Name:Ouyang JH, k={d}')
plt.show()
-----------------------------------------------------------------------------------------------
X = np.column_stack((np.ones_like(X), X))

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

theta0, theta1 = theta
print("正规方程求解最优参数 (θ0, θ1):", theta0, theta1)

