import pandas as pd
from matplotlib.font_manager import FontProperties
from pylab import *
file = "C:/Users/902sx/Desktop/graduate/train_set.csv"
file1 = "C:/Users/902sx/Desktop/graduate/test_set.csv"
data = pd.read_csv(file, encoding='utf-8')
data1 = pd.read_csv(file1, encoding='utf-8')
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
import pandas as pd
from matplotlib.font_manager import FontProperties
from pylab import *
file = "C:/Users/902sx/Desktop/graduate/train_set.csv"
file1 = "C:/Users/902sx/Desktop/graduate/test_set.csv"
data = pd.read_csv(file, encoding='utf-8')
data1 = pd.read_csv(file1, encoding='utf-8')
for i in range(0, 25317):
    # if i == 0:
    #     continue
    marital = data['marital'][i]
    # print(age)
    if marital == "married":
        data['marital'][i] = 5
    if marital == "divorced":
        data['marital'].loc[i] = 6
    if marital == "single":
        data['marital'].loc[i] = 8


data.to_csv(file, index=0, encoding='utf-8')

for i in range(0, 10852):
    # if i == 0:
    #     continue
    marital = data1['marital'][i]
    # print(age)
    if marital == "married":
        data1['marital'][i] = 5
    if marital == "divorced":
        data1['marital'].loc[i] = 6
    if marital == "single":
        data1['marital'].loc[i] = 8

data1.to_csv(file1, index=0, encoding='utf-8')
import pandas as pd
from matplotlib.font_manager import FontProperties
from pylab import *
file = "C:/Users/902sx/Desktop/graduate/train_set.csv"
file1 = "C:/Users/902sx/Desktop/graduate/test_set.csv"
data = pd.read_csv(file, encoding='utf-8')
data1 = pd.read_csv(file1, encoding='utf-8')
for i in range(0, 25317):
    # if i == 0:
    #     continue
    education = data['education'][i]
    # print(age)
    if education == "tertiary":
        data['education'][i] = 8
    if education == "primary":
        data['education'].loc[i] = 6
    if education == "secondary":
        data['education'].loc[i] = 7
    if education == "unknown":
        data['education'].loc[i] = 5


data.to_csv(file, index=0, encoding='utf-8')

for i in range(0, 10852):
    # if i == 0:
    #     continue
    education = data1['education'][i]
    # print(age)
    if education == "tertiary":
        data1['education'][i] = 8
    if education == "primary":
        data1['education'].loc[i] = 6
    if education == "secondary":
        data1['education'].loc[i] = 7
    if education == "unknown":
        data1['education'].loc[i] = 5

data1.to_csv(file1, index=0, encoding='utf-8')
import pandas as pd
from matplotlib.font_manager import FontProperties
from pylab import *
file = "C:/Users/902sx/Desktop/graduate/train_set.csv"
file1 = "C:/Users/902sx/Desktop/graduate/test_set.csv"
data = pd.read_csv(file, encoding='utf-8')
data1 = pd.read_csv(file1, encoding='utf-8')
for i in range(0, 25317):
    # if i == 0:
    #     continue
    contact = data['contact'][i]
    # print(age)
    if contact == "cellular":
        data['contact'][i] = 7
    if contact == "telephone":
        data['contact'].loc[i] = 8
    if contact == "unknown":
        data['contact'].loc[i] = 9


data.to_csv(file, index=0, encoding='utf-8')

for i in range(0, 10852):
    # if i == 0:
    #     continue
    contact = data1['contact'][i]
    # print(age)
    if contact == "cellular":
        data1['contact'][i] = 7
    if contact == "telephone":
        data1['contact'].loc[i] = 8
    if contact == "unknown":
        data1['contact'].loc[i] = 9

data1.to_csv(file1, index=0, encoding='utf-8')
import pandas as pd
from matplotlib.font_manager import FontProperties
from pylab import *
file = "C:/Users/902sx/Desktop/graduate/train_set.csv"
file1 = "C:/Users/902sx/Desktop/graduate/test_set.csv"
data = pd.read_csv(file, encoding='utf-8')
data1 = pd.read_csv(file1, encoding='utf-8')
for i in range(0, 25317):
    # if i == 0:
    #     continue
    month = data['month'][i]
    # print(age)
    if month == "sep":
        data['month'][i] = 9
    if month == "oct":
        data['month'].loc[i] = 9
    if month == "nov":
        data['month'].loc[i] = 8
    if month == "jan":
        data['month'].loc[i] = 7
    if month == "feb":
        data['month'].loc[i] = 7
    if month == "mar":
        data['month'].loc[i] = 6
    if month == "apr":
        data['month'].loc[i] = 6
    if month == "may":
        data['month'].loc[i] = 6
    if month == "jun":
        data['month'].loc[i] = 5
    if month == "jul":
        data['month'].loc[i] = 5
    if month == "aug":
        data['month'].loc[i] = 5
    if month == "dec":
        data['month'].loc[i] = 5

data.to_csv(file, index=0, encoding='utf-8')

for i in range(0, 10852):
    # if i == 0:
    #     continue
    month = data1['month'][i]
    # print(age)
    if month == "sep":
        data1['month'][i] = 9
    if month == "oct":
        data1['month'].loc[i] = 9
    if month == "nov":
        data1['month'].loc[i] = 8
    if month == "jan":
        data1['month'].loc[i] = 7
    if month == "feb":
        data1['month'].loc[i] = 7
    if month == "mar":
        data1['month'].loc[i] = 6
    if month == "apr":
        data1['month'].loc[i] = 6
    if month == "may":
        data1['month'].loc[i] = 6
    if month == "jun":
        data1['month'].loc[i] = 5
    if month == "jul":
        data1['month'].loc[i] = 5
    if month == "aug":
        data1['month'].loc[i] = 5
    if month == "dec":
        data1['month'].loc[i] = 5

data1.to_csv(file1, index=0, encoding='utf-8')
import pandas as pd
from matplotlib.font_manager import FontProperties
from pylab import *
file = "C:/Users/902sx/Desktop/graduate/train_set.csv"
file1 = "C:/Users/902sx/Desktop/graduate/test_set.csv"
data = pd.read_csv(file, encoding='utf-8')
data1 = pd.read_csv(file1, encoding='utf-8')
for i in range(0, 25317):
    # if i == 0:
    #     continue
    poutcome = data['poutcome'][i]
    # print(age)
    if poutcome == "failure":
        data['poutcome'][i] = 6
    if poutcome == "success":
        data['poutcome'].loc[i] = 8
    if poutcome == "other":
        data['poutcome'].loc[i] = 7
    if poutcome == "unknown":
        data['poutcome'].loc[i] = 8


data.to_csv(file, index=0, encoding='utf-8')

for i in range(0, 10852):
    # if i == 0:
    #     continue
    poutcome = data1['poutcome'][i]
    # print(age)
    if poutcome == "failure":
        data1['poutcome'][i] = 6
    if poutcome == "success":
        data1['poutcome'].loc[i] = 8
    if poutcome == "other":
        data1['poutcome'].loc[i] = 7
    if poutcome == "unknown":
        data1['poutcome'].loc[i] = 8

data1.to_csv(file1, index=0, encoding='utf-8')