import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 1000)

pd.set_option('display.width', 1000)

pd.set_option('display.max_colwidth', 1000)
# %matplotlib inline

train = pd.read_csv(r'C:\Users\902sx\Desktop\graduate\train_set.csv')
test = pd.read_csv(r'C:\Users\902sx\Desktop\graduate\test_set.csv')
customer_buy = pd.read_csv(r'C:\Users\902sx\Desktop\graduate\buy.csv')
customer_not_buy = pd.read_csv(r'C:\Users\902sx\Desktop\graduate\notbuy.csv')

train.shape
# out:(25317, 18)
test.shape
# out:(10852, 17)
# print(train.isnull().sum())
# print(train.duplicated().sum())
# print(train.describe())
# print(train['y'].value_counts()[1] / train['y'].value_counts().sum())
# out:0.11695698542481336
# 样本存在严重的不均衡问题，正样本数只占11.7%


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# data={
#     'Buy':(customer_buy.iloc[:,0].size,'#7199cf'),
#     'NotBuy':(customer_not_buy.iloc[:,0].size,'#ffff10'),
# }
# fig=plt.figure(figsize=(6,6))
# results=data.keys()
# values=[x[0] for x in data.values()]
# colors=[x[1] for x in data.values()]
#
# ax1=fig.add_subplot(111)
# labels=['{}:{}'.format(result,value) for result,value in zip(results,values)]
# ax1.pie(values, autopct='%1.0f%%',labels=labels, colors=colors, shadow=True,textprops={'fontsize':20, 'color':'k'})
# plt.show()
#
# data={'primary', 'secondary', 'tertiary', 'unknown'}
# x=np.arange(4)
# data1=customer_buy.groupby(by='education').size()/customer_buy['ID'].count()
# data2=customer_not_buy.groupby(by='education').size()/customer_not_buy['ID'].count()
# print("buy")
# print(customer_buy.groupby(by='education').size() / customer_buy['ID'].count())
# print("not buy")
# print(customer_not_buy.groupby(by='education').size() / customer_not_buy['ID'].count())
# y1=data1
# y2=data2
# bar_width = 0.3
# label =data1.index.values
# plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
# plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
# plt.legend()
# plt.xticks(x+bar_width/2, label)
# plt.xlabel("教育水平")
# plt.ylabel('样本占比')
# plt.title("教育程度")
# plt.show()

# data1 = pd.DataFrame(customer_buy.groupby(by='job').size() / customer_buy['ID'].count())
# data1.rename(columns = {0: 'Per'}, inplace=True)
# print(data1.sort_values(by='Per', ascending=False))
#
# data2 = pd.DataFrame(customer_not_buy.groupby(by='job').size() / customer_not_buy['ID'].count())
# data2.rename(columns = {0: 'Per'}, inplace=True)
# print(data2.sort_values(by='Per', ascending=False))



# x=np.arange(3)
# data1=customer_buy.groupby(by='marital').size()/customer_buy['ID'].count()
# data2=customer_not_buy.groupby(by='marital').size()/customer_not_buy['ID'].count()
# print("buy")
# print(customer_buy.groupby(by='marital').size() / customer_buy['ID'].count())
# print("not buy")
# print(customer_not_buy.groupby(by='marital').size() / customer_not_buy['ID'].count())
# y1=data1
# y2=data2
# bar_width = 0.3
# label =data1.index.values
# plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
# plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
# plt.legend()
# plt.xticks(x+bar_width/2, label)
# plt.xlabel("婚姻状态")
# plt.ylabel("样本占比")
# plt.title("婚姻状况")
# plt.show()

# print("buy")
# print(customer_buy.groupby(by='default').size() / customer_buy['ID'].count())
# print("not buy")
# print(customer_not_buy.groupby(by='default').size() / customer_not_buy['ID'].count())


# x=np.arange(2)
# data1=customer_buy.groupby(by='loan').size()/customer_buy['ID'].count()
# data2=customer_not_buy.groupby(by='loan').size()/customer_not_buy['ID'].count()
# print("buy")
# print(customer_buy.groupby(by='loan').size() / customer_buy['ID'].count())
# print("not buy")
# print(customer_not_buy.groupby(by='loan').size() / customer_not_buy['ID'].count())
# y1=data1
# y2=data2
# bar_width = 0.3
# label =data1.index.values
# plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
# plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
# plt.legend()
# plt.xticks(x+bar_width/2, label)
# plt.title("个人贷款")
# plt.show()

# x=np.arange(2)
# data1=customer_buy.groupby(by='housing').size()/customer_buy['ID'].count()
# data2=customer_not_buy.groupby(by='housing').size()/customer_not_buy['ID'].count()
# print("buy")
# print(customer_buy.groupby(by='housing').size() / customer_buy['ID'].count())
# print("not buy")
# print(customer_not_buy.groupby(by='housing').size() / customer_not_buy['ID'].count())
# y1=data1
# y2=data2
# bar_width = 0.3
# label =data1.index.values
# plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
# plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
# plt.legend()
# plt.xticks(x+bar_width/2, label)
# plt.title("住房贷款")
# plt.show()

customer_buy.loc[:,'age_cut']=pd.qcut(customer_buy['age'],4).astype(str)
customer_not_buy.loc[:,'age_cut']=pd.qcut(customer_not_buy['age'],4).astype(str)
x=np.arange(4)
data1=customer_buy.groupby(by='age_cut').size()/customer_buy['ID'].count()
data2=customer_not_buy.groupby(by='age_cut').size()/customer_not_buy['ID'].count()
print("buy")
print(customer_buy.groupby(by='age_cut').size() / customer_buy['ID'].count())
print("not buy")
print(customer_not_buy.groupby(by='age_cut').size() / customer_not_buy['ID'].count())
y1=data1
y2=data2
bar_width = 0.3
label =data1.index.values
plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
plt.legend()
plt.xticks(x+bar_width/2, label)
plt.xlabel("年龄分布")
plt.ylabel("样本占比")
plt.title("客户年龄")
plt.show()

# customer_buy.loc[:,'balance_cut']=pd.cut(customer_buy['balance'],[-8020,0,5000,10000,50000,100000,150000]).astype(str)
# customer_not_buy.loc[:,'balance_cut']=pd.cut(customer_not_buy['balance'],[-8020,0,5000,10000,50000,100000,150000]).astype(str)
# # print(customer_buy['balance_cut'])
# x=np.arange(6)
# data1=customer_buy.groupby(by='balance_cut').size()/customer_buy['ID'].count()
# data2=customer_not_buy.groupby(by='balance_cut').size()/customer_not_buy['ID'].count()
# # print("buy")
# # print(customer_buy.groupby(by='balance_cut').size() / customer_buy['ID'].count())
# # print("not buy")
# # print(customer_not_buy.groupby(by='balance_cut').size() / customer_not_buy['ID'].count())
# y1=data1
# series = pd.Series([0.0],index = ['(100000,150000]'])
# y1=y1.append(series)
# y2=data2
# bar_width = 0.3
# label =y1.index.values
# plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
# plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
# plt.legend()
# plt.xticks(x+bar_width/2, label)
# plt.show()

# x=np.arange(3)
# data1=customer_buy.groupby(by='contact').size()/customer_buy['ID'].count()
# data2=customer_not_buy.groupby(by='contact').size()/customer_not_buy['ID'].count()
# print("buy")
# print(customer_buy.groupby(by='contact').size() / customer_buy['ID'].count())
# print("not buy")
# print(customer_not_buy.groupby(by='contact').size() / customer_not_buy['ID'].count())
# y1=data1
# y2=data2
# bar_width = 0.3
# label =data1.index.values
# plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
# plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
# plt.legend()
# plt.xticks(x+bar_width/2, label)
# plt.title("联系方式")
# plt.show()
#
# x=np.arange(4)
# data1=customer_buy.groupby(by='poutcome').size()/customer_buy['ID'].count()
# data2=customer_not_buy.groupby(by='poutcome').size()/customer_not_buy['ID'].count()
# print("buy")
# print(customer_buy.groupby(by='poutcome').size() / customer_buy['ID'].count())
# print("not buy")
# print(customer_not_buy.groupby(by='poutcome').size() / customer_not_buy['ID'].count())
# y1=data1
# y2=data2
# bar_width = 0.3
# label =data1.index.values
# plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
# plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
# plt.legend()
# plt.xticks(x+bar_width/2, label)
# plt.title("上一次联系结果")
# plt.show()

# cont_list = ['duration']
# customer_buy[cont_list].boxplot()

# customer_buy.loc[:,'campaign_cut']=pd.cut(customer_buy['campaign'],[-1,0,1,2,3,100]).astype(str)
# customer_not_buy.loc[:,'campaign_cut']=pd.cut(customer_not_buy['campaign'],[-1,0,1,2,3,100]).astype(str)
# # print(customer_buy['balance_cut'])
# x=np.arange(4)
# data1=customer_buy.groupby(by='campaign_cut').size()/customer_buy['ID'].count()
# data2=customer_not_buy.groupby(by='campaign_cut').size()/customer_not_buy['ID'].count()
# print("buy")
# print(customer_buy.groupby(by='campaign_cut').size() / customer_buy['ID'].count())
# print("not buy")
# print(customer_not_buy.groupby(by='campaign_cut').size() / customer_not_buy['ID'].count())
# y1=data1
# # series = pd.Series([0.0],index = ['(100000,150000]'])
# # y1=y1.append(series)
# y2=data2
# bar_width = 0.3
# label =y1.index.values
# plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
# plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
# plt.legend()
# plt.xticks(x+bar_width/2, label)
# plt.show()
#
# customer_buy.loc[:,'previous_cut']=pd.cut(customer_buy['previous'],[-1,0,1,2,3,100]).astype(str)
# customer_not_buy.loc[:,'previous_cut']=pd.cut(customer_not_buy['previous'],[-1,0,1,2,3,100]).astype(str)
# # print(customer_buy['balance_cut'])
# x=np.arange(5)
# data1=customer_buy.groupby(by='previous_cut').size()/customer_buy['ID'].count()
# data2=customer_not_buy.groupby(by='previous_cut').size()/customer_not_buy['ID'].count()
# print("buy")
# print(customer_buy.groupby(by='previous_cut').size() / customer_buy['ID'].count())
# print("not buy")
# print(customer_not_buy.groupby(by='previous_cut').size() / customer_not_buy['ID'].count())
# y1=data1
# # series = pd.Series([0.0],index = ['(100000,150000]'])
# # y1=y1.append(series)
# y2=data2
# bar_width = 0.3
# label =y1.index.values
# plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
# plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
# plt.legend()
# plt.xticks(x+bar_width/2, label)
# plt.show()


# customer_buy.loc[:,'pdays_cut']=pd.cut(customer_buy['pdays'],[-2,0,30,90,180,365,730,900]).astype(str)
# customer_not_buy.loc[:,'pdays_cut']=pd.cut(customer_not_buy['pdays'],[-2,0,30,90,180,365,730,900]).astype(str)
# # print(customer_buy['balance_cut'])
# x=np.arange(7)
# data1=customer_buy.groupby(by='pdays_cut').size()/customer_buy['ID'].count()
# data2=customer_not_buy.groupby(by='pdays_cut').size()/customer_not_buy['ID'].count()
# # print("buy")
# # print(customer_buy.groupby(by='pdays_cut').size() / customer_buy['ID'].count())
# # print("not buy")
# # print(customer_not_buy.groupby(by='pdays_cut').size() / customer_not_buy['ID'].count())
# y1=data1
# # series = pd.Series([0.0],index = ['(100000,150000]'])
# # y1=y1.append(series)
# y2=data2
# bar_width = 0.3
# label =y1.index.values
# plt.bar(x,y1,bar_width,color = '#7199cf', label = 'Buy')
# plt.bar(x+bar_width,y2,bar_width,color = '#ffff10', label = 'Not Buy')
# plt.legend()
# plt.xticks(x+bar_width/2, label)
# plt.show()
