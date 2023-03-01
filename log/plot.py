import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from pylab import *
import csv
import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei']

# f = open(r'C:\Users\902sx\Desktop\graduate\train_set.csv')
# L = list(csv.reader(f))
b = 0
c = 0
d = 0
e = 0
f = 0
g = 0
h = 0
j = 0
k = 0
m = 0
n = 0

file = "C:/Users/902sx/Desktop/graduate/train_set.csv"
file1 = "C:/Users/902sx/Desktop/graduate/test_set.csv"
data = pd.read_csv(file, encoding='utf-8')
data1 = pd.read_csv(file1, encoding='utf-8')

# for i in range(len(L)):
#     if i == 0:
#         continue
#     data['job'].loc(i)


# for i in range(len(L)):
#     if i == 0:
#         continue
#     age = int(L[i][1])
#     # print(age)
#     if 10 <= age < 20:
#         b = b + 1
#     if 20 <= age < 30:
#         c = c + 1
#     if 30 <= age < 40:
#         d = d + 1
#     if 40 <= age < 50:
#         e = e + 1
#     if 50 <= age < 60:
#         f = f + 1
#     if 60 <= age < 70:
#         g = g + 1
#     if 70 <= age < 80:
#         h = h + 1
#     if 80 <= age < 90:
#         j = j + 1
#
# x = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
# y = [b, c, d, e, f, g, h, j]
# plt.bar(x, y)  # 竖的条形图
# # plt.grid(True)
# # plt.barh(x,y)#横的条形图，注意x,y坐标
# plt.title("年龄分布")
# plt.xlabel("年龄分段")
# plt.ylabel("人数")
# plt.show()

# print(a)

for i in range(0, 25317):
    # if i == 0:
    #     continue
    job = data['job'][i]
    # print(age)
    if job == "admin.":
        data['job'][i] = 5
    if job == "blue-collar":
        data['job'].loc[i] = 6
    if job == "entrepreneur":
        data['job'].loc[i] = 8
    if job == "management":
        data['job'].loc[i] = 8
    if job == "retired":
        data['job'].loc[i] = 7
    if job == "services":
        data['job'].loc[i] = 6
    if job == "student":
        data['job'].loc[i] = 5
    if job == "technician":
        data['job'].loc[i] = 9
    if job == "unemployed":
        data['job'].loc[i] = 9
    if job == "unknown":
        data['job'].loc[i] = 9
    if job == "self-employed":
        data['job'].loc[i] = 8
    if job == "housemaid":
        data['job'].loc[i] = 6

data.to_csv(file, index=0, encoding='utf-8')

for i in range(0, 10852):
    # if i == 0:
    #     continue
    job = data1['job'][i]
    # print(age)
    if job == "admin.":
        data1['job'][i] = 5
    if job == "blue-collar":
        data1['job'].loc[i] = 6
    if job == "entrepreneur":
        data1['job'].loc[i] = 8
    if job == "management":
        data1['job'].loc[i] = 8
    if job == "retired":
        data1['job'].loc[i] = 7
    if job == "services":
        data1['job'].loc[i] = 6
    if job == "student":
        data1['job'].loc[i] = 5
    if job == "technician":
        data1['job'].loc[i] = 9
    if job == "unemployed":
        data1['job'].loc[i] = 9
    if job == "unknown":
        data1['job'].loc[i] = 9
    if job == "self-employed":
        data1['job'].loc[i] = 8
    if job == "housemaid":
        data1['job'].loc[i] = 6

data1.to_csv(file1, index=0, encoding='utf-8')

# labels = (
# "admin.", "blue-collar", "entrepreneur", "management", "retired", "services", "student", "technician", "unemployed",
# "self-employed", "housemaid")
# colors = ("red", "aqua", "khaki", "yellow", "gray", "coral", "silver", "orange", "violet", "pink", "gold")
# sum = b + c + d + e + f + g + h + j + k + m + n
# fracs = [100 * b / sum, 100 * c / sum, 100 * d / sum, 100 * e / sum, 100 * f / sum, 100 * g / sum, 100 * h / sum,
#          100 * j / sum, 100 * k / sum, 100 * m / sum, 100 * n / sum]
#
# plt.pie(fracs, labels=labels, colors=colors, autopct='%1.0f%%')
# plt.title("客户职业")
# plt.show()


# for i in range(len(L)):
#     if i == 0:
#         continue
#     marital = L[i][3]
#     # print(age)
#     if marital == "married":
#         b = b + 1
#     if marital == "divorced":
#         c = c + 1
#     if marital == "single":
#         d = d + 1
#
# labels = ("married", "divorced", "single")
# colors = ("pink", "aqua", "orchid")
# sum = b + c + d
# fracs = [100 * b / sum, 100 * c / sum, 100 * d / sum]
#
# plt.pie(fracs, labels=labels, colors=colors, autopct='%1.0f%%')
# plt.title("婚姻状况")
# plt.show()

# for i in range(len(L)):
#     if i == 0:
#         continue
#     education = L[i][4]
#     # print(age)
#     if education == "primary":
#         b = b + 1
#     if education == "secondary":
#         c = c + 1
#     if education == "tertiary":
#         d = d + 1
#     if education == "unknown":
#         e = e + 1
#
# dataX = ["primary", "secondary", "unknown", "tertiary"]
# dataY = [b, c, e, d]
# plt.plot(dataX, dataY)
# plt.title("教育程度")
# plt.xlabel("x轴")
# plt.ylabel("y轴")
# plt.show()


# for i in range(len(L)):
#     if i == 0:
#         continue
#     default = L[i][5]
#     # print(age)
#     if default == "yes":
#         b = b + 1
#     if default == "no":
#         c = c + 1
#
# for i in range(len(L)):
#     if i == 0:
#         continue
#     housing = L[i][7]
#     # print(age)
#     if housing == "yes":
#         d = d + 1
#     if housing == "no":
#         e = e + 1
#
# for i in range(len(L)):
#     if i == 0:
#         continue
#     loan = L[i][8]
#     # print(age)
#     if loan == "yes":
#         f = f + 1
#     if loan == "no":
#         g = g + 1
#
# fig = plt.figure()
#
# ax1 = fig.add_subplot(221)
# sum = b + c
# labels = ("yes", "no")
# colors = ("yellow", "aqua")
# plt.title("是否有违约记录")
# fracs = [100 * b / sum, 100 * c / sum]
# ax1.pie(fracs, labels=labels, colors=colors, autopct='%1.0f%%')
#
#
# ax2 = fig.add_subplot(222)
# sum = d + e
# labels = ("yes", "no")
# colors = ("yellow", "aqua")
# plt.title("是否有住房贷款")
# fracs = [100 * d / sum, 100 * e / sum]
# ax2.pie(fracs, labels=labels, colors=colors, autopct='%1.0f%%')
#
# ax3 = fig.add_subplot(223)
# sum = f + g
# labels = ("yes", "no")
# colors = ("yellow", "aqua")
# plt.title("是否有个人贷款")
# fracs = [100 * f / sum, 100 * g / sum]
# ax3.pie(fracs, labels=labels, colors=colors, autopct='%1.0f%%')
# #
# # ax4 = fig.add_subplot(224)
# # ax4.plot(x, np.log(x))
#
# # plt.grid(True)
# # plt.barh(x,y)#横的条形图，注意x,y坐标
# plt.show()

# for i in range(len(L)):
#     if i == 0:
#         continue
#     contact = L[i][9]
#     # print(age)
#     if contact == "cellular":
#         b = b + 1
#     if contact == "telephone":
#         c = c + 1
#     if contact == "unknown":
#         d = d + 1
#
#
# dataX = ["telephone", "unknown", "cellular"]
# dataY = [ c, d, b]
# plt.bar(dataX, dataY)
# plt.title("与客户联系的沟通方式")
# plt.xlabel("x轴")
# plt.ylabel("y轴")
# plt.show()

# for i in range(len(L)):
#     if i == 0:
#         continue
#     balance = int(L[i][6])
#     # print(age)
#     if balance < 0:
#         b = b + 1
#     if 0 < balance < 500:
#         c = c + 1
#     if 500 < balance < 1000:
#         d = d+ 1
#     if 1000 < balance < 2000:
#         e = e + 1
#     if 2000 < balance < 3000:
#         f = f + 1
#     if 3000 < balance < 5000:
#         g = g + 1
#     if 5000 < balance:
#         h = h + 1
# x = ['有负债', '0~500', '500~1000', '1000~2000', '2000~3000', '3000~5000', '>5000']
# y = [b, c, d, e, f, g, h]
# plt.bar(x, y)  # 竖的条形图
# # plt.grid(True)
# # plt.barh(x,y)#横的条形图，注意x,y坐标
# plt.title("平均余额")
# plt.xlabel("金额分布")
# plt.ylabel("人数")
# plt.show()

# for i in range(len(L)):
#     if i == 0:
#         continue
#     duration = int(L[i][12])
#     # print(age)
#     if 0 < duration < 100:
#         b = b + 1
#     if 100 < duration < 200:
#         c = c + 1
#     if 200 < duration < 300:
#         d = d+ 1
#     if 300 < duration < 400:
#         e = e + 1
#     if 400 < duration < 500:
#         f = f + 1
#     if 500 < duration < 1000:
#         g = g + 1
#     if 1000 < duration:
#         h = h + 1
# x = ['0~100', '100~200', '200~300', '300~400', '400~500', '500~1000', '>1000']
# y = [b, c, d, e, f, g, h]
# plt.bar(x, y)  # 竖的条形图
# # plt.grid(True)
# # plt.barh(x,y)#横的条形图，注意x,y坐标
# plt.title("通话时长")
# plt.xlabel("时长分布")
# plt.ylabel("统计人数")
# plt.show()

# for i in range(len(L)):
#     if i == 0:
#         continue
#     campaign = int(L[i][13])
#     # print(age)
#     if campaign == 1 :
#         b = b + 1
#     if campaign == 2:
#         c = c + 1
#     if campaign == 3:
#         d = d+ 1
#     if campaign == 4:
#         e = e + 1
#     if campaign == 5:
#         f = f + 1
#     if campaign == 6:
#         g = g + 1
#     if campaign == 7:
#         h = h + 1
#     if campaign == 8:
#         j = j + 1
#     if 8 < campaign :
#         k = k + 1
# x = ['1', '2', '3', '4', '5', '6', '7', '8', '> 8']
# y = [b, c, d, e, f, g, h, j, k]
# plt.bar(x, y)  # 竖的条形图
# # plt.grid(True)
# # plt.barh(x,y)#横的条形图，注意x,y坐标
# plt.title("本次活动交流次数")
# plt.xlabel("交流次数")
# plt.ylabel("统计人数")
# plt.show()

# for i in range(len(L)):
#     if i == 0:
#         continue
#     campaign = int(L[i][15])
#     # print(age)
#     if campaign == 0:
#         b = b + 1
#     if campaign == 1:
#         c = c + 1
#     if campaign == 2:
#         d = d+ 1
#     if campaign == 3:
#         e = e + 1
#     if campaign == 4:
#         f = f + 1
#     if campaign == 5:
#         g = g + 1
#     if campaign == 6:
#         h = h + 1
#     if campaign == 7:
#         j = j + 1
#     if 7 < campaign :
#         k = k + 1
# x = ['0', '1', '2', '3', '4', '5', '6', '7', '> 7']
# y = [b, c, d, e, f, g, h, j, k]
# plt.bar(x, y)  # 竖的条形图
# # plt.grid(True)
# # plt.barh(x,y)#横的条形图，注意x,y坐标
# plt.title("本次活动之前交流次数")
# plt.xlabel("交流次数")
# plt.ylabel("统计人数")
# plt.show()

# for i in range(len(L)):
#     if i == 0:
#         continue
#     pdays = int(L[i][14])
#     # print(age)
#     if pdays == -1:
#         b = b + 1
#     if 0 < pdays < 100:
#         c = c + 1
#     if 100 < pdays < 200:
#         d = d + 1
#     if 200 < pdays < 300:
#         e = e + 1
#     if 300 < pdays :
#         f = f + 1
# dataX = ["今天刚交流", "0-100天", "100-200天", "200-300天", ">300天"]
# dataY = [b, c, d, e, f]
# plt.plot(dataX, dataY)
# plt.title("距离上次交流间隔时间")
# plt.xlabel("间隔时间")
# plt.ylabel("统计人数")
# plt.show()

# for i in range(len(L)):
#     if i == 0:
#         continue
#     poutcome = L[i][16]
#     # print(age)
#     if poutcome == "unknown":
#         b = b + 1
#     if poutcome == "other":
#         c = c + 1
#     if poutcome == "success":
#         d = d + 1
#     if poutcome == "failure":
#         e = e + 1
#
# labels = ("unknown", "other", "success", "failure")
# colors = ("pink", "aqua", "orchid", "grey")
# sum = b + c + d + e
# fracs = [100 * b / sum, 100 * c / sum, 100 * d / sum, 100 * e / sum]
#
# plt.pie(fracs, labels=labels, colors=colors, autopct='%1.0f%%')
# plt.title("上次活动结果")
# plt.show()

# count_1 = []
# count_2 = []
# count_3 = []
# count_4 = []
# count_5 = []
# count_6 = []
# count_7 = []
# count_8 = []
# count_9 = []
# count_10 = []
# count_11 = []
# count_12 = []
# for i in range(31):
#     count_1.append(0)
#     count_2.append(0)
#     count_3.append(0)
#     count_4.append(0)
#     count_5.append(0)
#     count_6.append(0)
#     count_7.append(0)
#     count_8.append(0)
#     count_9.append(0)
#     count_10.append(0)
#     count_11.append(0)
#     count_12.append(0)

# print(count_8)

# for i in range(len(L)):
#     if i == 0:
#         continue
#     # day = L[i][10]
#     month = L[i][11]
#     # print(age)
#     if month == "jan":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_1[day] = count_1[day] + 1
#     if month == "feb":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_2[day] = count_2[day] + 1
#     if month == "mar":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_3[day] = count_3[day] + 1
#     if month == "apr":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_4[day] = count_4[day] + 1
#     if month == "may":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_5[day] = count_5[day] + 1
#     if month == "jun":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_6[day] = count_6[day] + 1
#     if month == "jul":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_7[day] = count_7[day] + 1
#     if month == "aug":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_8[day] = count_8[day] + 1
#     if month == "sep":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_9[day] = count_9[day] + 1
#     if month == "oct":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_10[day] = count_10[day] + 1
#     if month == "nov":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_11[day] = count_11[day] + 1
#     if month == "dec":
#         day = int(L[i][10]) - 1
#         # print(day)
#         count_12[day] = count_12[day] + 1

# print(count_1)
# print(count_2)
# print(count_3)
# print(count_4)
# print(count_5)
# print(count_6)
# print(count_7)
# print(count_8)
# print(count_9)
# print(count_10)
# print(count_11)
# print(count_12)

#
# x = np.arange(1, 32)
# # m = np.arange(1, 32)
# # print(m2)
# y1 = count_1
# y2 = count_2
# y3 = count_3
# y4 = count_4
# y5 = count_5
# y6 = count_6
# y7 = count_7
# y8 = count_8
# y9 = count_9
# y10 = count_10
# y11 = count_11
# y12 = count_12
# plt.figure(figsize=(10, 8))
#
# plt.title("上次联系时间各月统计")
# plt.xlabel("具体日期")
# plt.ylabel("统计人数")
# plt.plot(x, y1, label='一月')
# plt.plot(x, y2, c='green', label='二月')
# plt.plot(x, y3, c='red', label='三月')
# plt. plot(x, y4, c='blue', label='四月')
# plt. plot(x, y5, c='violet', label='五月')
# plt. plot(x, y6, c='pink', label='六月')
# plt. plot(x, y7, c='gold', label='七月')
# plt. plot(x, y8, c='aqua', label='八月')
# plt. plot(x, y9, c='grey', label='九月')
# plt. plot(x, y10, c='black', label='十月')
# plt. plot(x, y11, c='silver', label='十一月')
# plt. plot(x, y12, c='orchid', label='十二月')
#
# plt.legend()
# plt.show()
