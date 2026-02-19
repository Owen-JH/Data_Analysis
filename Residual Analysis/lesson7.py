%matplotlib inline
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
filename1 = 'mathmetician.xls'
data1 = pd.read_excel(filename1)
n = data1.shape[0]
p = data1.shape[1]
ones = np.ones((n, 1))
X = data1[['x1', 'x2', 'x3']].values
X = np.hstack((ones, X))
y = data1['y'].values.reshape(n, 1)

XTX_inv = np.linalg.inv(np.matmul(X.T, X))
beta_hat = np.matmul(np.matmul(XTX_inv, X.T), y)
np.set_printoptions(precision=4)

y_bar = y.mean()
y_hat = np.matmul(X, beta_hat)
SSE = sum((y-y_hat)**2)

transfer = []
r0 = []
for i in range(n):
    transfer.append(np.matmul(np.matmul(X[i].T, XTX_inv), X[i]))
    r0.append((y[i]-y_hat[i])/((SSE/(n-p)*(1-transfer[i])))**0.5)
r = np.around(r0, decimals=4)
print(r)

counter1 = 0
counter2 = 0
counter3 = 0
for i in r:
    if i>=-1 and i<=1:
        counter1 = counter1+1
    if i>=-1.5 and i<=1.5:
        counter2 = counter2+1
    if i>=-2 and i<=2:
        counter3 = counter3+1
print('counter1 = ', counter1)
print('counter2 = ', counter2)
print('counter3 = ', counter3)
print('区间[-1,1]:', counter1/n)
print('区间[-1.5,1.5]:', counter2/n)
print('区间[-2,2]:', counter3/n)
-------------------------------------------------------------------------------------
sorted_r = sorted(r)
q = []
for i in range(n+1):
    if i == 0:
        continue
    else:
      q.append(stats.norm.ppf((i - 0.375) / (n + 0.25)))

q_percentiles = np.percentile(q, np.arange(1, 101))
r_percentiles = np.percentile(r, np.arange(1, 101))

plt.figure(figsize=(8, 8))
plt.scatter(q_percentiles, r_percentiles)
plt.plot([-2.5, 2.5], [-2.5, 2.5], linestyle='--', color='red')
plt.xlabel("q")
plt.ylabel("r")
plt.title("Normal QQ Plot for q and r")
plt.show()
--------------------------------------------------------------------------------------
epsilon_hat = y_hat-y
plt.scatter(y_hat, epsilon_hat)
plt.xlabel('y_hat')
plt.ylabel('epsilon_hat')
plt.show()

plt.scatter(X[:, 1], epsilon_hat)
plt.xlabel('x1')
plt.ylabel('epsilon_hat')
plt.show()

plt.scatter(X[:, 2], epsilon_hat)
plt.xlabel('x2')
plt.ylabel('epsilon_hat')
plt.show()

plt.scatter(X[:, 3], epsilon_hat)
plt.xlabel('x3')
plt.ylabel('epsilon_hat')
plt.show()
----------------------------------------------------------------------------------
filename2 = 'liver.xls'
data2 = pd.read_excel(filename2, header=None)
n = data2.shape[0]
p = data2.shape[1]
ones = np.ones((n, 1))
X = data2.iloc[:, 0:4].values
X = np.hstack((ones, X))
y = data2.iloc[:, 4].values.reshape(n, 1)

XTX_inv = np.linalg.inv(np.matmul(X.T, X))
beta_hat = np.matmul(np.matmul(XTX_inv, X.T), y)
np.set_printoptions(precision=4)

y_bar = y.mean()
y_hat = np.matmul(X, beta_hat)
SST = sum((y-y_bar)**2)
SSE = sum((y-y_hat)**2)

transfer = []
r0 = []
for i in range(n):
    transfer.append(np.matmul(np.matmul(X[i].T, XTX_inv), X[i]))
    r0.append((y[i]-y_hat[i])/((SSE/(n-p)*(1-transfer[i])))**0.5)
r = np.around(r0, decimals=4)
print(r)

counter1 = 0
counter2 = 0
counter3 = 0
for i in r:
    if i>=-1 and i<=1:
        counter1 = counter1+1
    if i>=-1.5 and i<=1.5:
        counter2 = counter2+1
    if i>=-2 and i<=2:
        counter3 = counter3+1
print('counter1 = ', counter1)
print('counter2 = ', counter2)
print('counter3 = ', counter3)
print('区间[-1,1]:', counter1/n)
print('区间[-1.5,1.5]:', counter2/n)
print('区间[-2,2]:', counter3/n)

sorted_r = sorted(r)
q = []
for i in range(n+1):
    if i == 0:
        continue
    else:
      q.append(stats.norm.ppf((i - 0.375) / (n + 0.25)))

q_percentiles = np.percentile(q, np.arange(1, 101))
r_percentiles = np.percentile(r, np.arange(1, 101))

plt.figure(figsize=(8, 8))
plt.scatter(q_percentiles, r_percentiles)
plt.plot([-2.5, 2.5], [-2.5, 2.5], linestyle='--', color='red')
plt.xlabel("q")
plt.ylabel("r")
plt.title("Normal QQ Plot for q and r")
plt.show()

epsilon_hat = y_hat-y
plt.scatter(y_hat, epsilon_hat)
plt.xlabel('y_hat')
plt.ylabel('epsilon_hat')
plt.show()

plt.scatter(X[:, 1], epsilon_hat)
plt.xlabel('x1')
plt.ylabel('epsilon_hat')
plt.show()

plt.scatter(X[:, 2], epsilon_hat)
plt.xlabel('x2')
plt.ylabel('epsilon_hat')
plt.show()

plt.scatter(X[:, 3], epsilon_hat)
plt.xlabel('x3')
plt.ylabel('epsilon_hat')
plt.show()

plt.scatter(X[:, 4], epsilon_hat)
plt.xlabel('x4')
plt.ylabel('epsilon_hat')
plt.show()
-------------------------------------------------------------------------------------
yboxcox, maxlog = stats.boxcox(y)
XTX_inv = np.linalg.inv(np.matmul(X.T, X))
yboxcox_beta_hat = np.matmul(np.matmul(XTX_inv, X.T), yboxcox)
np.set_printoptions(precision=4)

yboxcox_bar = yboxcox.mean()
yboxcox_hat = np.matmul(X, yboxcox_beta_hat)
yboxcox_SSE = sum((yboxcox-yboxcox_hat)**2)

transfer = []
r0 = []
for i in range(n):
    transfer.append(np.matmul(np.matmul(X[i].T, XTX_inv), X[i]))
    r0.append((yboxcox[i]-yboxcox_hat[i])/((yboxcox_SSE/(n-p)*(1-transfer[i])))**0.5)
r = np.around(r0, decimals=4)
print(r)

counter1 = 0
counter2 = 0
counter3 = 0
for i in r:
    if i>=-1 and i<=1:
        counter1 = counter1+1
    if i>=-1.5 and i<=1.5:
        counter2 = counter2+1
    if i>=-2 and i<=2:
        counter3 = counter3+1
print('counter1 = ', counter1)
print('counter2 = ', counter2)
print('counter3 = ', counter3)
print('区间[-1,1]:', counter1/n)
print('区间[-1.5,1.5]:', counter2/n)
print('区间[-2,2]:', counter3/n)

sorted_r = sorted(r)
q = []
for i in range(n+1):
    if i == 0:
        continue
    else:
      q.append(stats.norm.ppf((i - 0.375) / (n + 0.25)))

q_percentiles = np.percentile(q, np.arange(1, 101))
r_percentiles = np.percentile(r, np.arange(1, 101))

plt.figure(figsize=(8, 8))
plt.scatter(q_percentiles, r_percentiles)
plt.plot([-2.5, 2.5], [-2.5, 2.5], linestyle='--', color='red')
plt.xlabel("q")
plt.ylabel("r")
plt.title("Normal QQ Plot for q and r")
plt.show()

yboxcox_epsilon_hat = yboxcox_hat-yboxcox
plt.scatter(yboxcox_hat, yboxcox_epsilon_hat)
plt.xlabel('yboxcox_hat')
plt.ylabel('yboxcox_epsilon_hat')
plt.show()

plt.scatter(X[:, 1], yboxcox_epsilon_hat)
plt.xlabel('x1')
plt.ylabel('yboxcox_epsilon_hat')
plt.show()

plt.scatter(X[:, 2], yboxcox_epsilon_hat)
plt.xlabel('x2')
plt.ylabel('yboxcox_epsilon_hat')
plt.show()

plt.scatter(X[:, 3], yboxcox_epsilon_hat)
plt.xlabel('x3')
plt.ylabel('yboxcox_epsilon_hat')
plt.show()

plt.scatter(X[:, 4], yboxcox_epsilon_hat)
plt.xlabel('x4')
plt.ylabel('yboxcox_epsilon_hat')
plt.show()
------------------------------------------------------------------------------------
lambd = 0.07
yboxcox_book = stats.boxcox(y, lambd)
yboxcox_book_beta_hat = np.matmul(np.matmul(XTX_inv, X.T), yboxcox_book)
yboxcox_book_bar = yboxcox_book.mean()
yboxcox_book_hat = np.matmul(X, yboxcox_book_beta_hat)
yboxcox_book_SSE = sum((yboxcox_book-yboxcox_book_hat)**2)
plt.scatter(lambd, yboxcox_book_SSE)
plt.xlabel('λ')
plt.ylabel('SSE')
plt.show()
-------------------------------------------------------------------------------------
filename3 = 'liver.xls'
data3 = pd.read_excel(filename3, header=None)
n = data3.shape[0]
p = data3.shape[1]
ones = np.ones((n, 1))
X = data3.iloc[:, 0:4].values
X = np.hstack((ones, X))
y = data3.iloc[:, 4].values.reshape(n, 1)
XTX_inv = np.linalg.inv(np.matmul(X.T, X))
beta_hat = np.matmul(np.matmul(XTX_inv, X.T), y)
np.set_printoptions(precision=4)

y_bar = y.mean()
y_hat = np.matmul(X, beta_hat)
SST = sum((y - y_bar) ** 2)

m = list(range(1, X.shape[1]))
for combination_length in range(1, X.shape[1] + 1):
    all_combinations = list(itertools.combinations(m, combination_length))
    print(f"Combinations of length {combination_length}:")
    for combo in all_combinations:
        selected_columns = X[:, combo]
        XTX_inv = np.linalg.inv(np.matmul(selected_columns.T, selected_columns))
        beta_hat = np.matmul(np.matmul(XTX_inv, selected_columns.T), y)
        y_hat = np.matmul(selected_columns, beta_hat)
        SSE = sum((y - y_hat) ** 2)
        S = SSE/SST
        q = combination_length+1
        K = (n-1)/(n-q)
        R = 1-S*K
        print('R:', R)

        MSE = SSE/(n-q)
        C = SSE/MSE-(n-q*2)
        print('C:', C)

        d = y-y_hat
        PRESS = sum(d**2)
        print('PRESS:', PRESS)











