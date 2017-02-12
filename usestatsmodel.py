#!/usr/bin/python
# -*- coding:utf-8 -*-



##使用CSV读取数据
path = '10.Advertising.csv'  # 设置文件路径,因为在一个包内优先会检查本地的
data = pd.read_csv(path) #header=None
x = data[['TV', 'Radio']] #df.values[:, :-1]
y = data['Sales'] #df.values[:, -1]

##使用自带的数据
from sklearn.datasets import load_iris
iris = load_iris() #导入IRIS数据集
iris.data #特征矩阵
iris.target #目标向量


##从ftp上读取数据
url='http://stats191.stanford.edu/data/salary.table'
salary_table=pd.read_table(url)

import pandas as pd
url='https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima=pd.read_csv(url,header=None,names=col_names)
feature_cols=['pregnant','insulin','bmi','age']
X=pima[feature_cols]
y=pima.label




##使用pandas处理文本格式的时间  时间序列处理 将字符合并 并转成dti
eq.columns=['Date','Time','Latitude','Longitude','Depth','Magnitude','EventType','Place'] #修改列名为英文
eq['Time']=eq['Time'].map(lambda x: x[0:8])
eq['Datetime']=eq['Date']+' '+eq['Time']  #字符型构造成datetime64的形式
eq.Datetime=pd.to_datetime(eq.Datetime) #转换
eq=eq.set_index('Datetime') #将时间设置成index
days=eq.index.to_period('D').value_counts()# 对日期频数统计





##使用pandas进行数据探索 统计描述
import  pandas as pd
tips.dtypes  #查看数据类型
tips.describe()

import seaborn as sns
sns.pairplot(tips, hue="smoker") #色调由day 衡量(类别型变量) 展示的是数值型的相互关系图

##类别数据
s1 = pd.Series(["a","b","c","a"], dtype="category")
s2= pd.Series(["a","b","c","a"])
s_cat = s2.astype("category", categories=["b","c","d"], ordered=True)
tips['sex']=tips['sex'].astype('category') #s3
tips['sex'].cat.ordered #类别型变量是否有属性
tips['sex']=tips['sex'].cat.set_categories(['Male','Female'], ordered=True) # 开始是按字符串字母顺序排序,现在用cat定义了顺序
tips['sex'].cat.ordered
tips[['sex','smoker','day','time']].describe() #类别型的数据描述 计数
tips['sex'].value_counts() #类别数据的统计
tips['raio']=tips['tip']/tips['total_bill']
tips.groupby(['sex','time']).mean().raio# 每次都是对整体先做在取出黎要的



#使用statsmodels 也可以做逻辑回归,而且结果更加计量化可接受
import statsmodels.api as sm
import statsmodels.formula.api as smf
model = sm.logit("Survived ~ Pclass_1 + Pclass_2 + Pclass_1 + Sex_0 + Sex_1 + Age_scaled + SibSp + Parch + Fare_scaled", data=input_df)
result = model.fit() # 内部使用极大似然估计，所以会有结果返回，表示成功收敛
print(result.summary())

from  statsmodels.formula.api import ols,rlm,glm#TSA
formula='S~C(E)+X:C(M)-1'
#  -表示移除 -1移除参数项 :a * b * c - a:b:c 只有a:b+b:c+a:c
# a * b 表示  a + b + a:b,
# a / b 表示 a + a:b ;a / (b + c) 表示 a + a:b + a:c
# (a + b):(c + d) 表示 a:c + a:d + b:c + b:d ; (a:b):(a:c) is the same as a:b:c.
# I(a + b) 类似转义 算的是a+b
# C(a) 类型变量
lm=ols(formula,salary_table).fit()
print lm.summary()
lm.predict({'X':[12,12],'M':[1,1],'E':[1,2]})





#使用statsmodels 进行TSA 时间序列分析
import statsmodels as sm
model = sm.tsa.AR(df_march.temp) # AR 模型就是自回归模型
result = model.fit(72) #认为过去 72 小时都会影响现在时间点的温度


#方差分析
from statsmodels.stats.api import anova_lm
print anova_lm(lm)