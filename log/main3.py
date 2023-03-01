
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

train = pd.read_csv(r'C:\Users\902sx\Desktop\graduate\train_set.csv')
test = pd.read_csv(r'C:\Users\902sx\Desktop\graduate\test_set.csv')

train.shape
# out:(25317, 18)
test.shape
# out:(10852, 17)

train.isnull().sum()  # 不存在缺失值
train.duplicated().sum()  # 不存在重复值

train.describe()  # 无异常值

train['y'].value_counts()[1] / train['y'].value_counts().sum()
# out:0.11695698542481336
# 样本存在严重的不均衡问题，正样本数只占11.7%
#
# factor = ['ID','age', 'job', 'marital', 'education', 'default', 'balance',
#               'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'
#              ]
#
# for i in range(len(factor)):
#         for j in range(i, len(factor) - 1):
#             print(
#                 "指标%s与指标%s之间的相关性大小为%f" % (factor[i], factor[j + 1], pearsonr(train[factor[i]], train[factor[j + 1]])[0]))
#
#
factor = ['ID','age', 'job', 'marital', 'education', 'default', 'balance',
              'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'
             ]
from scipy.stats import pearsonr
for i in range(len(factor)):
        for j in range(i, len(factor) - 1):
            print(
                "指标%s与指标%s之间的相关性大小为%f" % (factor[i], factor[j + 1], pearsonr(train[factor[i]], train[factor[j + 1]])[0]))
# print(
#                 "指标%s与指标%s之间的相关性大小为%f" % ('age', 'balance', pearsonr(train['age'], train['balance'])[0]))
# import matplotlib.pyplot as plt
# factor = ['age', 'job', 'marital', 'education', 'default', 'balance',
#               'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
#              ]
#
# for i in range(len(factor)):
#         for j in range(i, len(factor) - 1):
#             plt.figure(figsize=(10, 4), dpi=100)
#             plt.scatter(train[factor[i]], train[factor[j + 1]])
#             plt.xlabel(factor[i])
#             plt.ylabel(factor[j + 1])
#             plt.show()


import pandas as pd
# from scipy.stats import pearsonr
#
# def pearsonr_demo():
#     """
#     相关系数计算
#     :return: None
#     """
#
#     factor = ['age', 'job', 'marital', 'education', 'default', 'balance',
#               'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
#               ]
#
#     for i in range(len(factor)):
#         for j in range(i, len(factor) - 1):
#             print(
#                 "指标%s与指标%s之间的相关性大小为%f" % (factor[i], factor[j + 1], pearsonr(train[factor[i]], train[factor[j + 1]])[0]))
#
#     return None
# pearsonr_demo()