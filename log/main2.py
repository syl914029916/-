import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
# 逻辑回归
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix as cm, precision_score, recall_score, roc_curve, \
    roc_auc_score as AUC

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


# 需要进行数据无量纲化处理的列
standard_scaler_list = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
# 需要转换为0-1二值编码的列
set_01_list = ['default', 'housing', 'loan']
# 需要进行one-hot编码的列
one_hot_list = ['job', 'marital', 'education', 'contact', 'day', 'month', 'poutcome']

# 1.0-1编码
# 训练集
from sklearn.preprocessing import OrdinalEncoder

train_done = train.copy()
encoder = OrdinalEncoder()
encoder.fit(train_done.loc[:, set_01_list])
train_done.loc[:, set_01_list] = encoder.transform(train_done.loc[:, set_01_list])
# 测试集
test_done = test.copy()
test_done.loc[:, set_01_list] = encoder.transform(test_done.loc[:, set_01_list])

# 2.one-hot编码
# 训练集
train_onehot = train[one_hot_list]
for i in one_hot_list:
    a = pd.get_dummies(train_onehot[i], columns=[i], prefix=i)
    train_done = pd.concat([train_done, a], axis=1)

train_done.drop(one_hot_list, axis=1, inplace=True)
# 测试集
test_onehot = test[one_hot_list]
for i in one_hot_list:
    a = pd.get_dummies(test_onehot[i], columns=[i], prefix=i)
    test_done = pd.concat([test_done, a], axis=1)
test_done.drop(one_hot_list, axis=1, inplace=True)

# 3.数据无量纲化
# 训练集
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_done.loc[:, standard_scaler_list])
train_done.loc[:, standard_scaler_list] = scaler.transform(train_done.loc[:, standard_scaler_list])

# 测试集
test_done.loc[:, standard_scaler_list] = scaler.transform(test_done.loc[:, standard_scaler_list])

# 构建训练集
X = train_done.drop(['ID', 'y'], axis=1)
y = train_done['y']
# 测试集处理
test_x = test_done.drop('ID', axis=1)
test_id = test_done['ID']






# 上采样法平衡样本
import imblearn
from imblearn.over_sampling import SMOTE
# 拆分数据集，构建训练、测试数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
sm = SMOTE(random_state=420)
Xtrain_, Ytrain_ = sm.fit_resample(Xtrain, Ytrain)
# 建模
lr_ = LR(solver='liblinear', random_state=420)
# auc_score_ = cross_val_score(lr_, Xtrain_, Ytrain_, cv=10).mean()
# 调参C，因为现在加了很多手工数据，样品是平衡的，这里就用默认的准确率作为模型评分
# score = []
# C = np.arange(0.01, 10.01, 0.1)
# for i in C:
#     lr_ = LR(solver='liblinear', C=i, random_state=420)
#     score.append(cross_val_score(lr_, Xtrain_, Ytrain_, cv=10).mean())
# print(max(score), C[score.index(max(score))])
# plt.figure(figsize=(20, 5))
# plt.plot(C, score, label='test')
# plt.xticks(C)
# plt.legend()
# plt.show()
lr_ = LR(solver='liblinear', C=0.51, random_state=420)
# 训练
lr_ = lr_.fit(Xtrain_, Ytrain_)
Ypred_train_ = lr_.predict(Xtrain_)
Ypred_test_ = lr_.predict(Xtest)


# 混淆矩阵
print("训练集混淆矩阵")
print(cm(Ytrain_, Ypred_train_, labels=[1, 0]))

'''
array([[13406,  2234],
       [ 2289, 13351]], dtype=int64)
'''
print("训练集精确度和召回率")
print(precision_score(Ytrain_, Ypred_train_))

print(recall_score(Ytrain_, Ypred_train_))
# 混淆矩阵-测试集
print("测试集混淆矩阵")
print(cm(Ytest, Ypred_test_, labels=[1, 0]))
'''
array([[ 703,  177],
       [1018, 5698]], dtype=int64)
'''
print("测试集精确度和召回率")
print(precision_score(Ytest, Ypred_test_))

print(recall_score(Ytest, Ypred_test_))
# AUC面积
print("训练集auc面积")
print(AUC(Ytrain_, lr_.predict_proba(Xtrain_)[:, 1]))
'''
0.9246362816504339
'''
print("测试集集auc面积")
print(AUC(Ytest, lr_.predict_proba(Xtest)[:, 1]))
'''
0.9031982646597
'''
#
# lr = LR(solver='liblinear', C=0.11, random_state=420)
# lr = lr.fit(X, y)
# ytest_pred2 = lr.predict_proba(test_x)[:, 1]
# result2 = pd.DataFrame({'ID': test_id, 'pred': ytest_pred2})
# result2.to_csv(r'C:\Users\902sx\Desktop\graduate\result_lr2.csv', index=False)