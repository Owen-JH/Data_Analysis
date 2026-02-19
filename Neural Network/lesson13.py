%matplotlib inline
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
filename = 'hw13data.mat'
data = loadmat(filename)
X = np.array(data['X'])
Y = np.array(data['y'])
Xval = np.array(data['Xval'])
yval = np.array(data['yval'])
Xtest = np.array(data['Xtest'])
Ytest = np.array(data['ytest'])

plt.scatter(X, Y, color='blue', marker='o', label='Points')
plt.title('the training set')
plt.xlabel('Change in water level(X)')
plt.ylabel('Water flowing out of the dam(Y)')
plt.legend()
plt.show()

theta = np.array([[1.0], [1.0]])
ones = np.ones((X.shape[0], 1))
X1 = np.hstack((ones, X))
sum = 0
lambda1 = 1
m = X1.shape[0]
for i in range(m):
    sum += (1/(2*m))*((np.matmul(X1[i, :], theta)-Y[i])**2)
sum += (lambda1/(2*m))*np.sum(theta**2)
print(sum)

sum1 = 0
sum2 = 0
for i in range(m):
    sum1 += (1/m)*(np.matmul(X1[i, :], theta) - Y[i])
    sum2 += (1/m)*((np.matmul(X1[i, :], theta) - Y[i]) * X1[i, 1])
print(sum1,sum2)

for j in range(100000):
    sum1 = 0
    sum2 = 0
    for i in range(m):
        sum1 += np.matmul(X1[i, :], theta) - Y[i]
        sum2 += (np.matmul(X1[i, :], theta) - Y[i]) * X1[i, 1]
    theta[0] = theta[0] - (0.001/m) * sum1
    theta[1] = theta[1] - (0.001/m) * sum2

xk = np.linspace(-50, 40, 1000)
yk_1 = theta[1] * xk + theta[0]
plt.plot(xk, yk_1, label='Line', color='blue')
plt.scatter(X, Y, color='red', marker='o', label='Points')
plt.title('the training set')
plt.xlabel('Change in water level(X)')
plt.ylabel('Water flowing out of the dam(Y)')
plt.legend()
plt.show()

sum = 0
theta1 = np.array([[1.0], [1.0]])
for k in range(m):
    for j in range(100):
        sum1 = 0
        sum2 = 0
        for i in range(k+1):
            sum1 += np.matmul(X1[i, :], theta1) - Y[i]
            sum2 += (np.matmul(X1[i, :], theta1) - Y[i]) * X1[i, 1]
        theta1[0] = theta1[0] - (0.001 / m) * sum1
        theta1[1] = theta1[1] - (0.001 / m) * sum2
    for n in range(k+1):
        sum += (1 / (2 * (k+1))) * ((np.matmul(X1[n, :], theta) - Y[n]) ** 2)









