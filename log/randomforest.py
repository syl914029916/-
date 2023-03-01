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

from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix as cm, precision_score, recall_score, roc_curve, \
    roc_auc_score as AUC

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0)

# 拆分数据集，构建训练、测试数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

# # 调参
# score1 = []
# param = np.arange(1, 1000, 100)
# for i in param:
#     rfc = RandomForestClassifier(n_estimators=i, n_jobs=-1, random_state=90)
#     score = cross_val_score(rfc, Xtrain, Ytrain, cv=10, scoring='roc_auc').mean()
#     score1.append(score)
# print(max(score1), param[(score1.index(max(score1)))])
# plt.figure(figsize=[20, 5])
# plt.plot(param, score1, 'o-')
# plt.show()
# # 可以继续细化，最优参数：n_estimators=960

# 调参max_depth
# score1 = []
# param = np.arange(1, 20, 1)
# for i in param:
#     rfc = RandomForestClassifier(n_estimators=960, max_depth=i, n_jobs=-1, random_state=90)
#     score = cross_val_score(rfc, Xtrain, Ytrain, cv=5, scoring='roc_auc').mean()
#     score1.append(score)
# print(max(score1), param[score1.index(max(score1))])
# plt.figure(figsize=[20, 5])
# plt.plot(param, score1, 'o-')
# plt.show()
# 最优参数，max_depth=18

# 调参max_features
# score1 = []
# param = np.arange(5, 40, 5)
# for i in param:
#     rfc = RandomForestClassifier(n_estimators=960, max_depth=18, max_features=i, n_jobs=-1, random_state=90)
#     score = cross_val_score(rfc, Xtrain, Ytrain, cv=5, scoring='roc_auc').mean()
#     score1.append(score)
# print(max(score1), param[score1.index(max(score1))])
# plt.figure(figsize=[20, 5])
# plt.plot(param, score1, 'o-')
# plt.show()
# 最优参数,max_features=15

# 网格搜索
from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier(n_estimators=960, max_depth=18, max_features=15, n_jobs=-1, random_state=90)
para_grid = {'min_samples_split': np.arange(2, 11), 'min_samples_leaf': np.arange(2, 11)}
gs = GridSearchCV(rfc, param_grid=para_grid, cv=5, scoring='roc_auc')

gs.fit(Xtrain, Ytrain)

# 最优参数模型
gs_best = gs.best_estimator_

gs.best_score_
'''
0.9314588907764207
'''

rfc = gs_best.fit(Xtrain, Ytrain)
Ypred = rfc.predict(Xtrain)
Ypred_test = rfc.predict(Xtest)
# 结果看起来在测试集上跟训练集上还是差一些,可以再调参试试。或者用过采样法试试
print(cm(Ytrain, Ypred, labels=[1, 0]))
'''
array([[ 1614,   467],
       [    8, 15632]], dtype=int64)
'''
print(cm(Ytest, Ypred_test, labels=[1, 0]))
'''
array([[ 384,  496],
       [ 213, 6503]], dtype=int64)
'''

rfc1 = gs_best.fit(Xtrain, Ytrain)


def get_rocauc(X, y, clf):
    from sklearn.metrics import roc_curve
    FPR, recall, thresholds = roc_curve(y, clf.predict_proba(X)[:, 1], pos_label=1)
    area = AUC(y, clf.predict_proba(X)[:, 1])

    maxindex = (recall - FPR).tolist().index(max(recall - FPR))
    threshold = thresholds[maxindex]

    plt.figure()
    plt.plot(FPR, recall, color='red', label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.scatter(FPR[maxindex], recall[maxindex], c='black', s=30)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.show()
    return threshold


# 获取最佳阀值
threshold = get_rocauc(Xtrain, Ytrain, rfc1)


def get_ypred(X, clf, threshold):
    y_pred = []
    for i in clf.predict_proba(X)[:, 1]:
        if i > threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


# 根据阀值调整分类结果
ytrain_pred = get_ypred(Xtrain, rfc1, threshold)
ytest_pred = get_ypred(Xtest, rfc1, threshold)

# 混淆矩阵
print(cm(Ytrain, ytrain_pred, labels=[1, 0]))
'''
array([[ 2011,    70],
       [  918, 14722]], dtype=int64)
'''

print(cm(Ytest, ytest_pred, labels=[1, 0]))
'''
array([[ 855,   25],
       [ 445, 6271]], dtype=int64)
'''

# rfc_done = gs_best.fit(X, y)
# y_pred = rfc_done.predict_proba(test_x)[:, 1]
# result2 = pd.DataFrame({'ID': test_id, 'pred': y_pred})
# result2.to_csv(r'E:\date\kesci\result_gs.csv', index=False)
