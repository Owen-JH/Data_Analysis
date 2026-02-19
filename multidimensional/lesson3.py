import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from itertools import groupby
filename1 = r'C:\大学\数据分析\lesson1\SerumProteinComponents.xls'
data1 = pd.read_excel(filename1, header=None)
x1 = data1.values.flatten()
W1, p_value1 = stats.shapiro(x1)
xbar = np.mean(x1)
s = np.std(x1)
num_bins = 10
observed_counts, bin_edges = np.histogram(x1, bins=num_bins)
bin_widths = np.diff(bin_edges)
expected_freq = np.array([stats.norm.cdf(bin_edges[i + 1], loc=xbar, scale=s) -
                          stats.norm.cdf(bin_edges[i], loc=xbar, scale=s) for i in range(num_bins)])
expected_freq *= len(x1)
chi_statistic, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_freq)
print("卡方统计量:", chi_statistic)
print("p 值:", p_value)

alpha = 0.05
if p_value < alpha:
    print("拒绝 H0，数据不符合正态分布。")
else:
    print("无法拒绝 H0，数据可能符合正态分布。")
stats.anderson(x1, dist='norm')
stats.kstest(x1, 'norm')
------------------------------------------------------------------------------------------------------
filename2 = r'C:\大学\数据分析\lesson3\test_score.xls'
data2 = pd.read_excel(filename2, header=None)
x2 = data2.values.flatten()
W2, p_value2 = stats.shapiro(x2)
print(W2, p_value2)

alpha = 0.05
if p_value2 < alpha:
    print("拒绝 H0，数据不符合正态分布。")
else:
    print("无法拒绝 H0，数据可能符合正态分布。")

shape, loc, scale = stats.weibull_min.fit(x2, floc=0)
ks_statistic, ks_p_value = stats.kstest(x2, 'weibull_min', args=(shape, loc, scale))
print("Weibull 分布参数：")
print("Shape:", shape)
print("Location:", loc)
print("Scale:", scale)
print("K-S 检验统计量:", ks_statistic)
print("K-S 检验 p 值:", ks_p_value)

alpha = 0.05
if ks_p_value < alpha:
    print("拒绝 H0，数据不符合 Weibull 分布。")
else:
    print("无法拒绝 H0，数据可能符合 Weibull 分布。")
------------------------------------------------------------------------------------------------------------
filename3 = r'C:\大学\数据分析\lesson3\height.xls'
data3 = pd.read_excel(filename3, header=None)
x3 = data3.values.flatten()
xbar3 = np.mean(x3)
s23 = np.var(x3)
s3 = np.std(x3)
CV3 = s3/xbar3
skewness3 = stats.skew(x3)
kurt3 = stats.kurtosis(x3)
Q13 = np.quantile(x3, 0.25, interpolation='midpoint')
Q33 = np.quantile(x3, 0.75, interpolation='midpoint')
R13 = Q33-Q13
arithmetic_mean3 = np.mean(x3)
geometric_mean3 = np.exp(np.mean(np.log(x3)))
harmonic_mean3 = len(x3) / np.sum(1.0 / np.array(x3))
print(f'xbar3={xbar3:4.2f}')
print(f's23={s23:4.2f}')
print(f's3={s3:4.2f}')
print(f'CV3={CV3:4.2f}')
print(f'skewness3={skewness3:4.2f}')
print(f'kurt3={kurt3:4.2f}')
print(f'Q13={Q13:4.2f}')
print(f'Q33={Q33:4.2f}')
print(f'R13={R13:4.2f}')
print("算术均值:", arithmetic_mean3)
print("几何均值:", geometric_mean3)
print("调和均值:", harmonic_mean3)
plt.rcParams['font.family'] = 'SimHei'
plt.hist(x3, bins=5, edgecolor='k')
plt.xlabel('身高')
plt.ylabel('人数')
plt.title('60名11岁学生身高直方图-欧阳嘉华')
plt.show()
for k, g in groupby(sorted(x3), key=lambda x: int(x)//10):
    lst = map(str, [int(_)%10 for _ in list(g)])
    print('%2d | %s'%(k, ' '.join(lst)))
W3, p_value3 = stats.shapiro(x3)
print(W3, p_value3)
alpha = 0.05
if p_value3 < alpha:
    print("拒绝 H0，数据不符合正态分布。")
else:
    print("无法拒绝 HO，数据可能符合正态分布。")
-------------------------------------------------------------------------------------------------------
filename4 = r'C:\大学\数据分析\lesson3\finance.xls'
data4 = pd.read_excel(filename4, header=None)
for column_name in data4.columns:
    if pd.api.types.is_numeric_dtype(data4[column_name]):  # 检查列是否为数字类型
        column_data = data4[column_name]
        mean_value = column_data.mean()
        variance_value = column_data.var()
        std_value = column_data.std()
        CV_value = std_value/mean_value
        skewness_value = column_data.skew()
        kurtosis_value = column_data.kurtosis()
        median_value = column_data.median()
        q3_value = np.percentile(column_data, 75, interpolation='midpoint')
        q1_value = np.percentile(column_data, 25, interpolation='midpoint')
        iqr_value = q3_value - q1_value

        print(f"列名: {column_name}")
        print(f"均值: {mean_value}")
        print(f"方差: {variance_value}")
        print(f"标准差: {std_value}")
        print(f"变异系数: {CV_value}")
        print(f"偏度: {skewness_value}")
        print(f"峰度: {kurtosis_value}")
        print(f"中位数: {median_value}")
        print(f"上四分位数: {q3_value}")
        print(f"下四分位数: {q1_value}")
        print(f"四分位极差: {iqr_value}")

        plt.rcParams['font.family'] = 'SimHei'
        plt.hist(column_data, bins=5, edgecolor='k')
        plt.xlabel('预算收入')
        plt.ylabel('城市数量')
        plt.show()

        x = np.sort(column_data)
        y = np.arange(1, len(column_data) + 1) / len(column_data)
        plt.step(x, y, where='post')
        plt.xlabel('预算收入')
        plt.ylabel('累积分布函数值')
        plt.title('经验分布函数图')
        plt.grid(True)
        plt.show()

column1 = data4[0]
column2 = data4[1]
correlation_coefficient1, p_value1 = stats.pearsonr(column1, column2)
correlation_coefficient2, p_value2 = stats.spearmanr(column1, column2)

print("Pearson相关系数:", correlation_coefficient1)
print("p-value:", p_value1)
print("Spearman相关系数:", correlation_coefficient2)
print("p-value:", p_value2)
--------------------------------------------------------------------------------------------------------
filename5 = r'C:\大学\数据分析\lesson3\blood.xls'
data5 = pd.read_excel(filename5, header=None)
xbar5 = data5.mean(axis=0)
cov = data5.cov()

print("均值向量μ:\n", xbar5)
print("协方差矩阵∑:\n", cov)

M5 = data5.median(axis=0)
Pearson = data5.corr(method='pearson')
Spearman = data5.corr(method='spearman')
print("Pearson相关矩阵R:\n", Pearson)
print("Spearman相关矩阵Q:\n", Spearman)

pearson_p = np.zeros((data5.shape[1], data5.shape[1]))
for i in range(data5.shape[1]):
    for j in range(i+1, data5.shape[1]):
        rij = stats.pearsonr(data5.iloc[:, i], data5.iloc[:, j])
        pearson_p[i, j] = rij[1]
print("Pearson相关性检验:\n", pearson_p)
spearman_p = np.zeros((data5.shape[1], data5.shape[1]))
for i in range(data5.shape[1]):
    for j in range(i+1, data5.shape[1]):
        rij = stats.spearmanr(data5.iloc[:, i], data5.iloc[:, j])
        spearman_p[i, j] = rij[1]
print("Spearman相关性检验:\n",spearman_p)






