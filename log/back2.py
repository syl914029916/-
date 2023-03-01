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
X = train_done.drop(['ID', 'job', 'y'], axis=1)
# X = Xt.drop('age', axis=1)
y = train_done['y']
# 测试集处理
test_x = test_done.drop(['ID', 'job'], axis=1)
test_id = test_done['ID']

print(X)
print(test_x)

# 逻辑回归
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix as cm, precision_score, recall_score, roc_curve, \
    roc_auc_score as AUC

# 拆分数据集，构建训练、测试数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

# 调参C
score = []
C = np.arange(0.01, 10.01, 0.1)
for i in C:
    lr = LR(solver='liblinear', C=i, random_state=420)
    score.append(cross_val_score(lr, Xtrain, Ytrain, cv=10, scoring='roc_auc').mean())
print(max(score), C[score.index(max(score))])
plt.figure(figsize=(20, 5))
plt.plot(C, score, label='test')
plt.xticks(C)
plt.legend()
plt.show()
# 可以继续细化调参范围，我这边获得的最佳参数C=0.11

# 训练数据
lr = LR(solver='liblinear', C=0.11, random_state=420)
lr = lr.fit(Xtrain, Ytrain)
# 模型跑出的训练数据结果
Ytrain_pred = lr.predict(Xtrain)
# 模型跑出的测试数据结果
Ytest_pred = lr.predict(Xtest)

# 混淆矩阵
print(cm(Ytrain, Ytrain_pred, labels=[1, 0]))
'''
array([[  741,  1340],
       [  342, 15298]], dtype=int64)
'''
# 从上面结果可以看出，训练数据集中正样本大部分都被分错了
print(cm(Ytest, Ytest_pred, labels=[1, 0]))
'''
array([[ 317,  563],
       [ 174, 6542]], dtype=int64)
'''
# 测试数据集也是这样

# AUC面积
print(AUC(Ytrain, lr.predict_proba(Xtrain)[:, 1]))
'''
0.9141338145270016
'''
print(AUC(Ytest, lr.predict_proba(Xtest)[:, 1]))
'''
0.9032848963127402
'''


# AUC面积看起来还挺高，只能说负样本占比太大了。


# 画roc-auc曲线
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


threshold = get_rocauc(Xtrain, Ytrain, lr)
print(threshold)
'''
0.11465704617442256
'''


def get_ypred(X, clf, threshold):
    y_pred = []
    for i in clf.predict_proba(X)[:, 1]:
        if i > threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


ytrain_pred = get_ypred(Xtrain, lr, threshold)
# 混淆矩阵
print(cm(Ytrain, ytrain_pred, labels=[1, 0]))
'''
array([[ 1783,   298],
       [ 2651, 12989]], dtype=int64)
'''
# 识别出了更多的正类

# 精准率低了很多，但是recall比例有更大的提升
print(precision_score(Ytrain, ytrain_pred))
'''
0.40211998195760035
'''

print(recall_score(Ytrain, ytrain_pred))
'''
0.8567996155694377
'''
# 测试集上的recall表现也不错
print(cm(Ytest, get_ypred(Xtest, lr, threshold), labels=[1, 0]))
'''
array([[ 727,  153],
       [1189, 5527]], dtype=int64)
'''

# lr = LR(solver='liblinear', C=0.11, random_state=420)
# lr = lr.fit(X, y)
# ytest_pred2 = lr.predict_proba(test_x)[:, 1]
# result2 = pd.DataFrame({'ID': test_id, 'pred': ytest_pred2})
# result2.to_csv(r'C:\Users\902sx\Desktop\graduate\result_lr.csv', index=False)
