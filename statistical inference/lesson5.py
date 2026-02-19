import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
filename1 = r'C:\大学\数据分析\lesson4\mathmetician.xls'
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
SST = sum((y-y_bar)**2)
SSE = sum((y-y_hat)**2)
SSR = SST-SSE
R2 = SSR/SST
print('复相关系数R2= ', R2)

MSR = SSR/(p-1)
MSE = SSE/(n-p)
F0 = MSR/MSE
df1 = p-1
df2 = n-p
p0 = 1-stats.f.cdf(F0, df1, df2, loc=0, scale=1)
print('SSR= ', SSR)
print('SSE= ', SSE)
print('SST= ', SST)
print('MSR= ', MSR)
print('MSE= ', MSE)
print('F0= ', F0)
print('p0= ', p0)

s = np.sqrt(MSE*np.diag(XTX_inv)).reshape(p, 1)
t0 = np.divide(beta_hat, s)
p0_t = 2*(1-stats.t.cdf(np.abs(t0), n-p))
print('标准差估计s=', s)
print('统计量tk的观测值t0=', t0)
print('检验p值p0_t=', p0_t)

alpha = 0.05
q = 1-alpha/2
df = n-p
t_percentile = stats.t.ppf(q, df, loc=0, scale=1)
beta_hat_left = beta_hat - np.multiply(t_percentile, s)
beta_hat_right = beta_hat + np.multiply(t_percentile, s)
print('β的左侧置信边界beta_hat_left=', beta_hat_left)
print('β的右侧置信边界beta_hat_right=', beta_hat_right)

x0 = np.array([1, 5.1, 20, 7.2])
y0_hat = np.matmul(x0, beta_hat)
temp = np.sqrt(MSE*(1+np.matmul(np.matmul(x0.T, XTX_inv), x0)))
y0_hat_left = y0_hat - t_percentile*temp
y0_hat_right = y0_hat + t_percentile*temp
print('y0_hat=', y0_hat)
print('y0的置信度为95%的置信区间为', y0_hat_left, y0_hat_right)
----------------------------------------------------------------------------------------------
filename2 = r'C:\大学\数据分析\lesson4\cosmetics.xls'
data2 = pd.read_excel(filename2)
n = data2.shape[0]
p = data2.shape[1]
ones = np.ones((n, 1))
X = data2[['x1', 'x2']].values
X = np.hstack((ones, X))

y = data2['y'].values.reshape(n, 1)

XTX_inv = np.linalg.inv(np.matmul(X.T, X))
beta_hat = np.matmul(np.matmul(XTX_inv, X.T), y)
np.set_printoptions(precision=4)
print(beta_hat)

y_bar = y.mean()
y_hat = np.matmul(X, beta_hat)
SST = sum((y-y_bar)**2)
SSE = sum((y-y_hat)**2)
SSR = SST-SSE
R2 = SSR/SST
print('复相关系数R2= ', R2)

MSR = SSR/(p-1)
MSE = SSE/(n-p)
F0 = MSR/MSE
df1 = p-1
df2 = n-p
p0 = 1-stats.f.cdf(F0, df1, df2, loc=0, scale=1)
print('方差分析表中数据如下:')
print('SSR= ', SSR)
print('SSE= ', SSE)
print('SST= ', SST)
print('MSR= ', MSR)
print('MSE= ', MSE)
print('F0= ', F0)
print('p0= ', p0)
------------------------------------------------------------------------------------------------------
s = np.sqrt(MSE*np.diag(XTX_inv)).reshape(p, 1)
alpha = 0.05
q = 1-alpha/2
df = n-p
t_percentile = stats.t.ppf(q, df, loc=0, scale=1)
beta_hat_left = beta_hat - np.multiply(t_percentile, s)
beta_hat_right = beta_hat + np.multiply(t_percentile, s)
print('β置信度为95%的左侧置信边界beta_hat_left=', beta_hat_left)
print('β置信度为95%的右侧置信边界beta_hat_right=', beta_hat_right)
------------------------------------------------------------------------------------------------------
X1 = data2['x1'].values
X2 = data2['x2'].values
Y = data2['y'].values
X1 = sm.add_constant(np.column_stack((X1, X2)))
model = sm.OLS(Y, X1).fit()
print(model.summary())

print("检验X1对Y的影响是否显著:")
print("T-statistic:", model.tvalues[1])
print("P-value:", model.pvalues[1])
if model.pvalues[1] < 0.05:
    print("X1对Y的影响显著")
else:
    print("X1对Y的影响显著")

print("\n检验X2对Y的影响是否显著:")
print("T-statistic:", model.tvalues[2])
print("P-value:", model.pvalues[2])
if model.pvalues[2] < 0.05:
    print("X2对Y的影响显著")
else:
    print("X2对Y的影响不显著")

X_interaction = np.column_stack((X1[:, 1], X2, X1[:, 1] * X2))
X_interaction = sm.add_constant(X_interaction)
model_interaction = sm.OLS(Y, X_interaction).fit()
print(model_interaction.summary())

print("\n检验X1和X2的交互作用对Y的影响是否显著:")
print("F-statistic:", model_interaction.fvalue)
print("P-value:", model_interaction.f_pvalue)
if model_interaction.f_pvalue < 0.05:
    print("X1和X2的交互作用对Y的影响显著")
else:
    print("X1和X2的交互作用对Y的影响不显著")
------------------------------------------------------------------------------------------------------
x1 = np.array([1, 220, 2500])
y1_hat = np.matmul(x1, beta_hat)
temp = np.sqrt(MSE*(1+np.matmul(np.matmul(x1.T, XTX_inv), x1)))
y1_hat_left = y1_hat - t_percentile*temp
y1_hat_right = y1_hat + t_percentile*temp
print('y0_hat=', y1_hat)
print('y0的置信度为95%的置信区间为', y1_hat_left, y1_hat_right)










