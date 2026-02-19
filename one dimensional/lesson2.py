import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.stats import skew, kurtosis
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.api as stm
filename1 = 'SerumProteinComponents.xls'
data1 = pd.read_excel(filename1, header=None)
x = data1.values.flatten()
xbar = np.mean(x)
M = np.median(x)
Q1 = np.quantile(x, 0.25, interpolation='midpoint')
Q3 = np.quantile(x, 0.75, interpolation='midpoint')
s2 = np.var(x)
s = np.std(x)
CV = s/xbar
R = np.ptp(x)
R1 = Q3-Q1
print(f's2={s2:4.2f}')
print(f's={s:4.2f}')
print(f'CV={CV:4.2f}')
print(f'R={R:4.2f}')
print(f'R1={R1:4.2f}')
for k in x:
    if k>=Q3+1.5*R1 or k<=Q1-1.5*R1:
        print(f'error={k:4.2f}')
#--------------------------------------------------------------------------------------
plt.rcParams['font.family'] = 'SimHei'
name_count = {}
filename2 = 'drinkbrands.xlsx'
data2 = pd.read_excel(filename2, header=None)
for col in data2.columns:
    for name in data2[col]:
        if name in name_count:
            name_count[name] += 1
        else:
            name_count[name] = 1

for name, count in name_count.items():
    print(f"{name}: {count}")

plt.figure(figsize=(8, 6))
plt.bar(name_count.keys(), name_count.values())
plt.xlabel('饮料品牌')
plt.ylabel('频数')
plt.title('饮料销售情况条形图')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(name_count.values(), bins=5, edgecolor='k')
plt.xlabel('饮料品牌')
plt.ylabel('频数')
plt.title('饮料销售情况柱状图')
plt.show()

plt.figure(figsize=(8, 6))
plt.pie(name_count.values(), labels=name_count.keys(), autopct='%1.1f%%', startangle=140)
plt.title('饮料销售情况饼图')
plt.axis('equal')
plt.show()
#----------------------------------------------------------------------------------------------------
filename3 = 'ComputerSales.xls'
data3 = pd.read_excel(filename3, header=None)
x1 = data3.values.flatten()
plt.rcParams['font.family'] = 'SimHei'
plt.hist(x1, bins=5, edgecolor='k')
plt.xlabel('销量')
plt.ylabel('频数')
plt.title('电脑公司销售情况直方图')
plt.show()
for k, g in groupby(sorted(x1), key=lambda x: int(x)//10):
    lst = map(str, [int(_)%10 for _ in list(g)])
    print('%2d | %s'%(k, ' '.join(lst)))
#------------------------------------------------------------------------------------
filename4 = 'eg1d3data.xls'
data4 = pd.read_excel(filename4, header=None)
result = pd.DataFrame(columns=['列名', '均值', '中位数', '上四分位数', '下四分位数', '方差', '标准差', '极差', '四分位极差', '偏度', '峰度'])

for column_name in data4.columns[1:6]:
    column_data = data4[column_name]
    mean_value = column_data.mean()
    median_value = column_data.median()
    q3_value = np.percentile(column_data, 75, interpolation='midpoint')
    q1_value = np.percentile(column_data, 25, interpolation='midpoint')
    variance_value = column_data.var()
    std_value = column_data.std()
    range_value = column_data.max() - column_data.min()
    iqr_value = q3_value - q1_value
    skewness_value = column_data.skew()
    kurtosis_value = column_data.kurtosis()

    result = pd.concat([result, pd.DataFrame({
        '列名': [column_name],
        '均值': [mean_value],
        '中位数': [median_value],
        '上四分位数': [q3_value],
        '下四分位数': [q1_value],
        '方差': [variance_value],
        '标准差': [std_value],
        '极差': [range_value],
        '四分位极差': [iqr_value],
        '偏度': [skewness_value],
        '峰度': [kurtosis_value]
    })], ignore_index=True)

result.to_excel('数字特征结果.xlsx', index=False)
#-----------------------------------------------------------------------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
filename5 = 'SerumProteinComponents.xls'
data5 = pd.read_excel(filename5, header=None)
x5 = data5.values.flatten()
mu, std = norm.fit(x5)
y = np.linspace(min(x5), max(x5), 100)
pdf = norm.pdf(y, mu, std)
plt.hist(x5, bins=10, density=True, alpha=0.6, color='g', label='Histogram')
plt.plot(y, pdf, 'r-', linewidth=2, label='Normal Distribution Fit')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Histogram with Normal Distribution Fit')
plt.legend()
plt.show()

ecdf = ECDF(x5)
plt.figure(figsize=(8, 6))
plt.step(ecdf.x, ecdf.y, label='Empirical CDF', color='blue')
plt.plot(y, pdf, 'r-', linewidth=2, label='Normal Distribution Fit')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Empirical CDF with Normal Distribution Fit')
plt.legend()
plt.show()

stm.qqplot(x5, line='s')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values')
plt.title('Probability Plot')
plt.show()



















