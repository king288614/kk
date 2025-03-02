import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 数据载入
dir_head ='chp3_data/' #将这个数据文件夹放在此py文件的同一目录下（即左边），即可直接引用
list_name = {os.path.splitext(fileName)[0]:os.path.splitext(fileName)[1]
             for fileName in os.listdir(dir_head)}
dataset = {} #将chp3_data中的所有数据放在dataset中并下载，之后可直接调用
for fileName in list_name.keys():
    if list_name[fileName] == '.csv':
        dataset[fileName] = pd.read_csv(dir_head+fileName+'.csv')
        dataset[fileName].set_index(dataset[fileName].columns[0], inplace = True)
        print(fileName+' is loaded.')

# 索引调整（使所有数据的行列索引一致，并且时间列不影响后面的数据处理）
for fileName in dataset.keys():
    if fileName == 'stk_company_info'or'securities_info':
        continue
    dataset[fileName].index = pd.to_datetime(dataset[fileName].index, format= '%Y-%m-%d')

# 计算复权价格及收益率，adj_就是复权价格
pricename = ['close','open','high','low']
for fileName in pricename:
    dataset['adj_'+fileName] = dataset[fileName]*dataset['adj_factor']

#计算收益率即'pct_chg'
dataset['pct_chg'] = np.log(dataset['adj_close']).diff()
y = dataset['pct_chg']
#用每列的中位数填充缺失值
y = y.fillna(y.median())
print(y)

#第一个因子：mv
a1 = dataset['total_mv']
#用每列的中位数填充缺失值
a1 = a1.fillna(a1.median())
#标准化
a1 = (a1 - a1.mean()) / a1.std()

#第二个因子：bp
a2 = 1/dataset['pb']
#用每列的中位数填充缺失值
a2 = a2.fillna(a2.median())
#标准化
a2 = (a2 - a2.mean()) / a2.std()

#第三个因子：alpha_037
def alpha_037():
    ##(-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
    ret = y
    temp = dataset['open'].rolling(window=5).sum() * ret.rolling(window=5).sum()
    part1 = temp.rank(axis=1, pct=True)
    part2 = temp.shift(10)
    result = -part1 - part2
    return result
a3 = alpha_037()
#用每列的中位数填充缺失值
a3 = a3.fillna(a3.median())
#标准化
a3 = (a3 - a3.mean()) / a3.std()
#用每列的中位数填充缺失值
a3 = a3.fillna(a3.median())
# 用每行的中位数填充该行的缺失值
a3 = a3.apply(lambda row: row.fillna(row.median()), axis=1)

#第四个因子：alpha_150
def alpha_150(close,high,low,vol):
    #(CLOSE + HIGH + LOW)/3 * VOLUME
    alpha = ((close + high + low) / 3 * vol)
    return alpha
a4 = alpha_150(dataset['close'],dataset['high'],dataset['low'],dataset['vol'])
#用每列的中位数填充缺失值
a4 = a4.fillna(a4.median())
#标准化
a4 = (a4 - a4.mean()) / a4.std()

#第五个因子：alpha_060
def alpha_060(close, low, high, vol, window=20):
    # 初始化一个空的 Series 用于存储计算结果
    result = pd.DataFrame(index=close.index,columns=close.columns)
    # 对于每一行数据，滚动计算窗口大小为 window 的 alpha_060
    for i in range(window, len(close)):
        # 选择滚动窗口内的过去 window 天的数据
        part1 = (close.iloc[i - window:i, :] - low.iloc[i - window:i, :]) - (
                    high.iloc[i - window:i, :] - close.iloc[i - window:i, :])
        part2 = high.iloc[i - window:i, :] - low.iloc[i - window:i, :]
        # 计算每个窗口内的值
        window_result = vol.iloc[i - window:i, :] * part1 / part2
        result.iloc[i] = window_result.sum()
    return result
a5 = alpha_060(dataset['close'], dataset['low'], dataset['high'], dataset['vol'], window=20)
#用每列的中位数填充缺失值
a5 = a5.fillna(a5.median())
#标准化
a5 = (a5 - a5.mean()) / a5.std()

#第六个因子：alpha_043
def alpha_043(close,vol):
    #SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
    delay1 = close.shift()
    condition1 = (close > delay1)
    condition2 = (close < delay1)
    temp1 = vol[condition1].fillna(0)
    temp2 = -vol[condition2].fillna(0)
    result = temp1 + temp2
    result = result.rolling(window=6).sum()
    return result
a6 = alpha_043(dataset['close'],dataset['vol'])
#用每列的中位数填充缺失值
a6 = a6.fillna(a6.median())
#标准化
a6 = (a6 - a6.mean()) / a6.std()

#第七个因子：alpha_046
def alpha_046(close):
    #(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    part1 = close.rolling(window=3).mean()
    part2 = close.rolling(window=6).mean()
    part3 = close.rolling(window=12).mean()
    part4 = close.rolling(window=24).mean()
    result = (part1 + part2 + part3 + part4) * 0.25 / close
    return result
a7 = alpha_046(dataset['close'])
#用每列的中位数填充缺失值
a7 = a7.fillna(a7.median())
#标准化
a7 = (a7 - a7.mean()) / a7.std()
#用每列的中位数填充缺失值
a7 = a7.fillna(a7.median())
# 用每行的中位数填充该行的缺失值
a7 = a7.apply(lambda row: row.fillna(row.median()), axis=1)

#第八个因子：alpha_132
def alpha_132(amount):
    #MEAN(AMOUNT, 20)
    alpha = amount.rolling(window=20, min_periods=1).mean()  # 使用rolling窗口计算均值
    return alpha
a8 = alpha_132(dataset['amount'])
#用每列的中位数填充缺失值
a8 = a8.fillna(a8.median())
#标准化
a8 = (a8 - a8.mean()) / a8.std()

#第九个因子：alpha_136
def alpha_136(y,vol,open):
    # ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
    ret = y  # 计算收益率
    data1 = -ret.diff(3).rank(axis=1, pct=True)  # 对差分进行排名并取负数
    # 计算 OPEN 和 VOLUME 的滚动相关性
    data2 = open.rolling(window=10, min_periods=1).corr(vol)  # 使用滚动窗口计算相关性
    # 计算 alpha
    alpha = data1 * data2
    return alpha
a9 = alpha_136(y,dataset['vol'],dataset['open'])
#用每列的中位数填充缺失值
a9 = a9.fillna(a9.median())
#标准化
a9 = (a9 - a9.mean()) / a9.std()
#用每行的中位数填充缺失值
a9 = a9.apply(lambda row: row.fillna(row.median()), axis=1)

#第十个因子：alpha_139
def alpha_139(open,vol):
    # (-1 * CORR(OPEN, VOLUME, 10))
    alpha = open.rolling(window=10, min_periods=1).corr(vol)
    return alpha
a10 = alpha_139(dataset['open'],dataset['vol'])
#用每列的中位数填充缺失值
a10 = a10.fillna(a10.median())
#标准化
a10 = (a10 - a10.mean()) / a10.std()
#用每行的中位数填充缺失值
a10 = a10.apply(lambda row: row.fillna(row.median()), axis=1)


##因子分析##

# 计算IC值
def a_corr(alpha,pct_chg):
  mean1 = alpha.mean(axis=1)
  mean2 = pct_chg.mean(axis=1)
# 计算每行的中心化数据
  centered1 = alpha.sub(mean1, axis=0)
  centered2 = pct_chg.sub(mean2, axis=0)
# 计算每行的方差
  var1 = centered1.pow(2).mean(axis=1)
  var2 = centered2.pow(2).mean(axis=1)
# 计算每行的标准差
  std1 = np.sqrt(var1)
  std2 = np.sqrt(var2)
# 计算每对行的协方差
  cov = centered1.mul(centered2.shift(-1)).mean(axis=1)
# 计算每对行的相关系数
  IC_value  = cov / (std1 * std2)
  return IC_value
IC1 = a_corr(a1, y)
IC2 = a_corr(a2, y)
IC3 = a_corr(a3, y)
IC4 = a_corr(a4, y)
IC5 = a_corr(a5, y)
IC6 = a_corr(a6, y)
IC7 = a_corr(a7, y)
IC8 = a_corr(a8, y)
IC9 = a_corr(a9, y)
IC10 = a_corr(a10, y)
#计算 IC 值序列的均值
IC1_mean = IC1.mean()
print(f"IC1值序列的均值: {IC1_mean}")
IC2_mean = IC2.mean()
print(f"IC2值序列的均值: {IC2_mean}")
IC3_mean = IC3.mean()
print(f"IC3值序列的均值: {IC3_mean}")
IC4_mean = IC4.mean()
print(f"IC4值序列的均值: {IC4_mean}")
IC5_mean = IC5.mean()
print(f"IC5值序列的均值: {IC5_mean}")
IC6_mean = IC6.mean()
print(f"IC6值序列的均值: {IC6_mean}")
IC7_mean = IC7.mean()
print(f"IC7值序列的均值: {IC7_mean}")
IC8_mean = IC8.mean()
print(f"IC8值序列的均值: {IC8_mean}")
IC9_mean = IC9.mean()
print(f"IC9值序列的均值: {IC9_mean}")
IC10_mean = IC10.mean()
print(f"IC10值序列的均值: {IC10_mean}")
#计算 IC 值序列的标准差
IC1_std = IC1.std()
print(f"IC1值序列的标准差: {IC1_std}")
IC2_std = IC2.std()
print(f"IC2值序列的标准差: {IC2_std}")
IC3_std = IC3.std()
print(f"IC3值序列的标准差: {IC3_std}")
IC4_std = IC4.std()
print(f"IC4值序列的标准差: {IC4_std}")
IC5_std = IC5.std()
print(f"IC5值序列的标准差: {IC5_std}")
IC6_std = IC6.std()
print(f"IC6值序列的标准差: {IC6_std}")
IC7_std = IC7.std()
print(f"IC7值序列的标准差: {IC7_std}")
IC8_std = IC8.std()
print(f"IC8值序列的标准差: {IC8_std}")
IC9_std = IC9.std()
print(f"IC9值序列的标准差: {IC9_std}")
IC10_std = IC10.std()
print(f"IC10值序列的标准差: {IC10_std}")
#计算 IC 值的累积值
IC1_cumsum = IC1.cumsum()
IC2_cumsum = IC2.cumsum()
IC3_cumsum = IC3.cumsum()
IC4_cumsum = IC4.cumsum()
IC5_cumsum = IC5.cumsum()
IC6_cumsum = IC6.cumsum()
IC7_cumsum = IC7.cumsum()
IC8_cumsum = IC8.cumsum()
IC9_cumsum = IC9.cumsum()
IC10_cumsum = IC10.cumsum()
#绘制 IC 值的累积曲线
plt.figure(figsize=(10, 6))
plt.plot(IC1_cumsum, label='IC1', color='blue')
plt.axhline(0, color='red', linestyle='--', label='0')  # 添加零水平线
plt.title('IC1-Cumulative Curve')
plt.xlabel('time')
plt.ylabel('IC1')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(IC2_cumsum, label='IC2', color='blue')
plt.axhline(0, color='red', linestyle='--', label='0')  # 添加零水平线
plt.title('IC2-Cumulative Curve')
plt.xlabel('time')
plt.ylabel('IC2')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(IC3_cumsum, label='IC3', color='blue')
plt.axhline(0, color='red', linestyle='--', label='0')  # 添加零水平线
plt.title('IC3-Cumulative Curve')
plt.xlabel('time')
plt.ylabel('IC3')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(IC4_cumsum, label='IC4', color='blue')
plt.axhline(0, color='red', linestyle='--', label='0')  # 添加零水平线
plt.title('IC4-Cumulative Curve')
plt.xlabel('time')
plt.ylabel('IC4')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(IC5_cumsum, label='IC5', color='blue')
plt.axhline(0, color='red', linestyle='--', label='0')  # 添加零水平线
plt.title('IC5-Cumulative Curve')
plt.xlabel('time')
plt.ylabel('IC5')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(IC6_cumsum, label='IC6', color='blue')
plt.axhline(0, color='red', linestyle='--', label='0')  # 添加零水平线
plt.title('IC6-Cumulative Curve')
plt.xlabel('time')
plt.ylabel('IC6')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(IC7_cumsum, label='IC7', color='blue')
plt.axhline(0, color='red', linestyle='--', label='0')  # 添加零水平线
plt.title('IC7-Cumulative Curve')
plt.xlabel('time')
plt.ylabel('IC7')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(IC8_cumsum, label='IC8', color='blue')
plt.axhline(0, color='red', linestyle='--', label='0')  # 添加零水平线
plt.title('IC8-Cumulative Curve')
plt.xlabel('time')
plt.ylabel('IC8')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(IC9_cumsum, label='IC9', color='blue')
plt.axhline(0, color='red', linestyle='--', label='0')  # 添加零水平线
plt.title('IC9-Cumulative Curve')
plt.xlabel('time')
plt.ylabel('IC9')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(IC10_cumsum, label='IC10', color='blue')
plt.axhline(0, color='red', linestyle='--', label='0')  # 添加零水平线
plt.title('IC10-Cumulative Curve')
plt.xlabel('time')
plt.ylabel('IC10')
plt.legend()
plt.grid(True)
plt.show()
#计算 IC 值大于零的占比
IC1_positive_ratio = (IC1 > 0).mean()
print(f"IC1值大于零的占比: {IC1_positive_ratio:.2%}")
IC2_positive_ratio = (IC2 > 0).mean()
print(f"IC2值大于零的占比: {IC2_positive_ratio:.2%}")
IC3_positive_ratio = (IC3 > 0).mean()
print(f"IC3值大于零的占比: {IC3_positive_ratio:.2%}")
IC4_positive_ratio = (IC4 > 0).mean()
print(f"IC4值大于零的占比: {IC4_positive_ratio:.2%}")
IC5_positive_ratio = (IC5 > 0).mean()
print(f"IC5值大于零的占比: {IC5_positive_ratio:.2%}")
IC6_positive_ratio = (IC6 > 0).mean()
print(f"IC6值大于零的占比: {IC6_positive_ratio:.2%}")
IC7_positive_ratio = (IC7 > 0).mean()
print(f"IC7值大于零的占比: {IC7_positive_ratio:.2%}")
IC8_positive_ratio = (IC8 > 0).mean()
print(f"IC8值大于零的占比: {IC8_positive_ratio:.2%}")
IC9_positive_ratio = (IC9 > 0).mean()
print(f"IC9值大于零的占比: {IC9_positive_ratio:.2%}")
IC10_positive_ratio = (IC10 > 0).mean()
print(f"IC10值大于零的占比: {IC10_positive_ratio:.2%}")

#计算三因子模型中的benchmark
# 计算日收益率：后一天减去前一天的数据除以前一天的数据
x7 = np.log(dataset['benchmark']).diff()
# 用0填充空值
x7 = x7.fillna(0)
# 使用 np.tile 来将单列数据复制为5093列
x7 = np.tile(x7, (1, 5093))
# 将 numpy 数组转化为 DataFrame
x7 = pd.DataFrame(x7)
#将x7的行列索引改成与y一致
x7.columns = y.columns
x7.index = y.index


#Barra多因子模型

#对自变量和因变量分别选择前80%行的数据作为训练集，剩下的20%行作为验证集
# 选择前80%的数据
y = y.iloc[:472]
x0 = a1.iloc[:472]
x1 = a2.iloc[:472]
x2 = a3.iloc[:472]
x3 = a5.iloc[:472]
x4 = a7.iloc[:472]
x5 = a9.iloc[:472]
x6 = a10.iloc[:472]
x71 = x7.iloc[:472]
#使用 statsmodels 进行最小二乘回归分析：
from matplotlib.ticker import MaxNLocator
# 计算样本数量，假设所有DataFrame的行数相同
n_samples = len(x1)  # 这里应该是总行数
# 对每一行数据进行回归分析
for t in range(n_samples - 1):  # 减 1 是因为要留出一行作为目标变量，预测下一期收益率
    # 提取特征矩阵X，使用第t行
    X = pd.concat([x1.iloc[t], x2.iloc[t], x3.iloc[t], x4.iloc[t], x5.iloc[t], x6.iloc[t]],
                  axis=1)
    X = sm.add_constant(X)  # 添加常数项（截距）
    # 提取目标变量y，使用第t+1行
    Y = y.iloc[t + 1, :]
    # 确保X和y的形状正确
    X = X.values  # 将X转换为NumPy数组
    Y = Y.values  # 将y转换为NumPy数组
    # 使用 statsmodels 的 OLS 方法
    model = sm.OLS(Y, X)  # OLS：普通最小二乘法
    results = model.fit()  # 拟合模型
    residuals = results.resid  # 获取残差
    # 获取回归系数和截距并添加到列表中
    coefficients = results.params[1:]
    intercepts = results.params[0]
# 打印回归系数和截距
print("barra模型的平均回归系数:", coefficients)
print("barra模型的平均截距:", intercepts)
# 打印回归结果的摘要
print(results.summary())
#提取剩下的20%数据即测试集
x11 = a2.iloc[472:]
x21 = a3.iloc[472:]
x31 = a5.iloc[472:]
x41 = a7.iloc[472:]
x51 = a9.iloc[472:]
x61 = a10.iloc[472:]
#计算预期收益率
y_test = intercepts + x11 * coefficients[0] + x21 * coefficients[1] + x31 * coefficients[2] + x41 * \
         coefficients[3] + x51 * coefficients[4] + x61 * coefficients[5]

# 回测图（Barra多因子模型）
#将剩下20%的测试集即一百多行中的30个最高收益率选出来，并算出每行平均值放在最后一列，用这一列的值画回测图
def calculate_avg_pcg(row):
    # 获取每行最大的30个值
    top_30 = row.nlargest(30)
    # 计算这些值的平均值
    return top_30.mean()
# 应用这个函数到每一行，并创建新的列
y_test['avg_pcg'] = y_test.apply(calculate_avg_pcg, axis=1)
avg_pcg_np = y_test['avg_pcg'].values
b= 10000
for i in range(len(avg_pcg_np)):
    b = b*(1+avg_pcg_np[i])
b = format(b,'.2f')
print('Barra模型的最终收益:',b)
# 假设 y_test 是你的 DataFrame，并且已经包含了 'avg_pcg' 列
# 计算累积乘积
y_test['cumulative_product'] = (1 + y_test['avg_pcg']).cumprod() - 1
# 第一个图形：累计曲线图
fig1, ax1 = plt.subplots(figsize=(10, 6))  # 创建一个图形和一个子图
ax1.plot(y_test.index, y_test['cumulative_product'], marker='o', linestyle='-', color='b', label='Cumulative Returns')
ax1.set_title('Barra Cumulative Returns')  # 设置图标题
ax1.set_xlabel('Index')  # 设置x轴标签
ax1.set_ylabel('Cumulative Returns')  # 设置y轴标签
ax1.legend()  # 显示图例
ax1.grid(True)  # 显示网格
# 设置x轴主刻度间隔为20，并将标签旋转90度
ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))
ax1.tick_params(axis='x', rotation=90)
# 第二个图形：收益率图像
fig2, ax2 = plt.subplots(figsize=(10, 5))  # 创建一个图形和一个子图
ax2.plot(y_test.index, y_test['avg_pcg'], marker='x', linestyle='--', color='r', label='Daily Returns')
ax2.set_title('Barra Daily Returns')  # 设置图标题
ax2.set_xlabel('Index')  # 设置x轴标签
ax2.set_ylabel('Daily Returns')  # 设置y轴标签
ax2.legend()  # 显示图例
ax2.grid(True)  # 显示网格
# 设置x轴主刻度间隔为20，并将标签旋转90度
ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))
ax2.tick_params(axis='x', rotation=90)
plt.tight_layout()  # 调整图布局
# plt.show()  # 显示图形

# 假设检验（Barra多因子模型）
# 1. 线性关系：通过散点图和残差图检查是否存在线性关系。
#散点图（绘制每个自变量与因变量的散点图）
# 绘制散点图
plt.scatter(x1, y)
plt.title('Scatter Plot of x1 vs Y')  # 标题
plt.xlabel('X')  # x轴标签
plt.ylabel('Y')  # y轴标签
plt.show()
plt.scatter(x2, y)
plt.title('Scatter Plot of x2 vs Y')  # 标题
plt.xlabel('X')  # x轴标签
plt.ylabel('Y')  # y轴标签
plt.show()
plt.scatter(x3, y)
plt.title('Scatter Plot of x3 vs Y')  # 标题
plt.xlabel('X')  # x轴标签
plt.ylabel('Y')  # y轴标签
plt.show()
plt.scatter(x4, y)
plt.title('Scatter Plot of x4 vs Y')  # 标题
plt.xlabel('X')  # x轴标签
plt.ylabel('Y')  # y轴标签
plt.show()
plt.scatter(x5, y)
plt.title('Scatter Plot of x5 vs Y')  # 标题
plt.xlabel('X')  # x轴标签
plt.ylabel('Y')  # y轴标签
plt.show()
plt.scatter(x6, y)
plt.title('Scatter Plot of x6 vs Y')  # 标题
plt.xlabel('X')  # x轴标签
plt.ylabel('Y')  # y轴标签
plt.show()
# 残差图（拟合值 vs 残差）
plt.figure(figsize=(6, 4))
# noinspection LanguageDetectionInspection
plt.scatter(results.fittedvalues, residuals) #上面回归模型时已获取残差residuals
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()
# 2. 独立性：通过 Durbin-Watson 检验检查误差项是否自相关
dw_stat = durbin_watson(residuals)
print(f'Durbin-Watson Statistic: {dw_stat}')
# 3. 同方差性：通过残差图和 Breusch-Pagan 检验检查是否存在异方差性
bp_test = het_breuschpagan(residuals, X)
print(f'Breusch-Pagan Test Statistic: {bp_test[0]}')
print(f'Breusch-Pagan Test p-value: {bp_test[1]}')
# 4. 正态性：通过 Shapiro-Wilk 检验和 Q-Q 图检查误差项是否符合正态分布
#Shapiro-Wilk 检验
stat, p_value = stats.shapiro(residuals)
print(f'Shapiro-Wilk Test p-value: {p_value}')
# Q-Q 图
stats.probplot(residuals, dist="norm", plot=plt)
# plt.show()
# 5. 多重共线性：通过计算 VIF 值判断自变量是否存在多重共线性
# 计算各自变量的VIF值
vif = variance_inflation_factor(X, 1)  # 1 代表计算 X1 的 VIF
print(f"x1的VIF: {vif}")
vif = variance_inflation_factor(X, 2)  # 2 代表计算 X2 的 VIF
print(f"x2的VIF: {vif}")
vif = variance_inflation_factor(X, 3)
print(f"x3的VIF: {vif}")
vif = variance_inflation_factor(X, 4)
print(f"x4的VIF: {vif}")
vif = variance_inflation_factor(X, 5)
print(f"x5的VIF: {vif}")
vif = variance_inflation_factor(X, 6)
print(f"x6的VIF: {vif}")


#简化版的Fama-French三因子模型

#将y换成y-x7
y = y-x71 #y,x71都是前472行
print(y)
# 计算样本数量，假设所有DataFrame的行数相同
n_samples = len(x1)  # 这里应该是总行数
# 对每一行数据进行回归分析
for t in range(n_samples - 1):  # 减1是因为要留出一行作为目标变量
    # 提取特征矩阵X，使用第t行
    X = pd.concat([x0.iloc[t], x1.iloc[t]],
                  axis=1)
    X = sm.add_constant(X)  # 添加常数项（截距）
    # 提取目标变量y，使用第t+1行
    Y = y.iloc[t + 1, :]
    # 确保X和y的形状正确
    X = X.values  # 将X转换为NumPy数组
    Y = Y.values  # 将y转换为NumPy数组
    # 使用 statsmodels 的 OLS 方法
    model = sm.OLS(Y, X)  # OLS：普通最小二乘法
    results = model.fit()  # 拟合模型
    # 获取回归系数和截距并添加到列表中
    coefficients = results.params[1:]
    intercepts = results.params[0]
# 打印回归系数和截距
print("简化版F-F三因子模型的平均回归系数:", coefficients)
print("简化版F-F三因子模型的平均截距:", intercepts)
# 打印回归结果的摘要
print(results.summary())
#提取剩下的20%数据即测试集
x11 = a1.iloc[472:]
x21 = a2.iloc[472:]
#计算预期收益率
y1_test = intercepts + x11 * coefficients[0] + x21 * coefficients[1]
#把benchmark加回到收益率（因为前面做简化版三因子模型时减掉了benchmark）
y1_test.index = pd.to_datetime(y1_test.index, format= '%Y-%m-%d')
x72 = x7.iloc[472:].reindex(y1_test.index)  # 对齐索引 #x72是后一百多行
y11_test = y1_test + x72
print(y1_test)
print(x72)
print(y11_test)

#回测图（简化版的F-F三因子模型）
#将剩下20%的测试集即一百多行中的30个最高收益率选出来，并算出每行平均值放在最后一列，用这一列的值画回测图
def calculate_avg_pcg(row):
    # 获取每行最大的30个值
    top_30 = row.nlargest(30)
    # 计算这些值的平均值
    return top_30.mean()
# 应用这个函数到每一行，并创建新的列
y11_test['avg_pcg'] = y11_test.apply(calculate_avg_pcg, axis=1)
print(y11_test['avg_pcg'])
avg_pcg_np = y11_test['avg_pcg'].values
b1 = 10000
for i in range(len(avg_pcg_np)):
    b1 = b1*(1+avg_pcg_np[i])
b1 = format(b1, '.2f')
print('三因子模型的最终收益:',b1)
# 假设 y11_test 是你的 DataFrame，并且已经包含了 'avg_pcg' 列
# 计算累积乘积
y11_test['cumulative_product'] = (1 + y11_test['avg_pcg']).cumprod() - 1
# 第一个图形：累计曲线图
fig, ax1 = plt.subplots(figsize=(10, 5))  # 创建一个图形和一个子图
ax1.plot(y11_test.index, y11_test['cumulative_product'], marker='o', linestyle='-', color='b', label='Cumulative Returns')
ax1.set_title('F-F Cumulative Returns')  # 设置图标题
ax1.set_xlabel('Index')  # 设置x轴标签
ax1.set_ylabel('Cumulative Returns')  # 设置y轴标签
ax1.legend()  # 显示图例
ax1.grid(True)  # 显示网格
# 设置x轴主刻度间隔为20，并将标签旋转90度
ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))
ax1.tick_params(axis='x', rotation=90)
plt.tight_layout()  # 调整图布局
# plt.show()  # 显示图形
# 第二个图形：收益率图像
fig2, ax2 = plt.subplots(figsize=(10, 6))  # 创建一个图形和一个子图
ax2.plot(y11_test.index, y11_test['avg_pcg'], marker='x', linestyle='--', color='r', label='Daily Returns')
ax2.set_title('F-F Daily Returns')  # 设置图标题
ax2.set_xlabel('Index')  # 设置x轴标签
ax2.set_ylabel('Daily Returns')  # 设置y轴标签
ax2.legend()  # 显示图例
ax2.grid(True)  # 显示网格
# 设置x轴主刻度间隔为20，并将标签旋转90度
ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))
ax2.tick_params(axis='x', rotation=90)
plt.tight_layout()  # 调整图布局
# plt.show()  # 显示图形


#三种收益率作图进行比较

y_test.columns = y_test.columns.str.strip()  # 去除列名的前后空格
y11_test.columns = y11_test.columns.str.strip()
x7.iloc[472:].columns = x7.iloc[472:].columns.str.strip()
Ba_return = y_test['avg_pcg'] #多因子回归模型收益率
FF_return =y11_test['avg_pcg'] #三因子回归模型收益率
b_return =x7.iloc[472:] #benchmark收益率
#使他们索引一样
common_index = y_test['avg_pcg'].index
# 创建一个图形
plt.figure(figsize=(10, 6))
# 绘制三个因变量的曲线
plt.plot(common_index, Ba_return, color='b', linestyle='-', marker='o')
plt.plot(common_index, FF_return, color='r', linestyle='--', marker='x')
plt.plot(common_index, b_return, color='g', linestyle=':', marker='s')
# 添加标题和标签
plt.title('Comparison of Ba_return, FF_return, and b_return')
plt.xlabel('Index')
plt.ylabel('Returns')
# 添加图例
# plt.legend(loc='best')
# 显示网格
plt.grid(True)
# 显示图形
plt.tight_layout()
# plt.show()


