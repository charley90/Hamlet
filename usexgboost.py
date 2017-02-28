# /usr/bin/python
# -*- encoding:utf-8 -*-



### 数据清洗

## 读取文件
import pandas as pd
loans_2007 = pd.read_csv('LoanStats3a.csv', skiprows=1)


## 手动  多次使用的数据查看
print(loans.info()) #最终的要求为所有特征 1不为空 2数据格式 最终的查看
print(loans_2007.shape[1])#查询有多少cloumn
print(loans_2007.iloc[0]) #查询cloumn的名称
print(loans_2007.isnull().sum()) #汇总看看每列空值的情况 根据重要性和缺失性进行处理
print(loans.select_dtypes(include=["object"]).iloc[0]) #查询object类型(混合格式)的情况
print(loans_2007['loan_status'].value_counts()) #查询cat类别,object类别字段的详细情况


## 自动 删除缺失数据太多的列
half_count = len(loans_2007) / 2  #设置阈值
loans_2007 = loans_2007.dropna(thresh=half_count, axis=1)
loans_2007 = loans_2007.drop(['desc', 'url'],axis=1)



## 自动 去掉单调重复列
orig_columns = loans_2007.columns
drop_columns = []
for col in orig_columns:
    col_series = loans_2007[col].dropna().unique()  #不为nan的值
    if len(col_series) == 1:
        drop_columns.append(col) #将单调的列添加到删除的列中
loans_2007 = loans_2007.drop(drop_columns, axis=1)



## 手动 部分特征剔除
loans_2007 = loans_2007.drop(["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d"], axis=1) #无关特征剔除
loans_2007 = loans_2007.drop(["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp"], axis=1) #结果特征剔除



## 半自动  处理缺失值
# 删除
loans_2007 = loans_2007.dropna(axis=0) #删除掉缺失的行
loans_2007 = loans_2007.drop("pub_rec_bankruptcies", axis=1) #删除掉缺失值依然太多的列

# 泰坦尼克预测 补齐船票价格缺失值
    if len(data.Fare[data.Fare.isnull()]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = data[data.Pclass == f + 1]['Fare'].dropna().median() #按照船舱等级计算每个等级的均值
        for f in range(0, 3):  # loop 0 to 2
            data.loc[(data.Fare.isnull()) & (data.Pclass == f + 1), 'Fare'] = fare[f] #用均值补全缺失值

# 年龄：使用随机森林预测年龄缺失值
from sklearn.ensemble import RandomForestRegressor
if is_train: #如果是训练
    print '随机森林预测缺失年龄：--start--'
    data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    age_exist = data_for_age.loc[(data.Age.notnull())]  # 年龄不缺失的数据
    age_null = data_for_age.loc[(data.Age.isnull())]  #年龄数据缺失的
    # print age_exist
    x = age_exist.values[:, 1:]
    y = age_exist.values[:, 0]
    rfr = RandomForestRegressor(n_estimators=1000)
    rfr.fit(x, y)
    age_hat = rfr.predict(age_null.values[:, 1:])
    # print age_hat
    data.loc[(data.Age.isnull()), 'Age'] = age_hat
    print '随机森林预测缺失年龄：--over--'
else: #测试集上面
    print '随机森林预测缺失年龄2：--start--'
    data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    age_exist = data_for_age.loc[(data.Age.notnull())]  # 年龄不缺失的数据
    age_null = data_for_age.loc[(data.Age.isnull())]
    # print age_exist
    x = age_exist.values[:, 1:]
    y = age_exist.values[:, 0]
    rfr = RandomForestRegressor(n_estimators=1000)
    rfr.fit(x, y)
    age_hat = rfr.predict(age_null.values[:, 1:])
    # print age_hat
    data.loc[(data.Age.isnull()), 'Age'] = age_hat
    print '随机森林预测缺失年龄2：--over--'





## 半自动 处理数据格式
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}
loans_2007 = loans_2007.replace(mapping_dict)  #复合格式字典替换

loans["int_rate"] = loans["int_rate"].str.rstrip("%").astype("float")  #字符转化
loans["revol_util"] = loans["revol_util"].str.rstrip("%").astype("float")
#时间序列转换
  ##主要将时间序列转换成距离今天的值,时期数据转换成数值



## 半自动 对于cat类别的标签替换
loans_2007 = loans_2007[(loans_2007['loan_status'] == "Fully Paid") | (loans_2007['loan_status'] == "Charged Off")] #有些异常的标签不显示了
status_replace = {     #设置替换字典
    "loan_status" : {
        "Fully Paid": 1,
        "Charged Off": 0,
    }
}
loans_2007 = loans_2007.replace(status_replace)  #进行cat类数据替换

## 半自动 哑变量处理
cat_columns = ["home_ownership", "verification_status", "emp_length", "purpose", "term"]
dummy_df = pd.get_dummies(loans_2007[cat_columns]) #生成哑变量
loans_2007 = pd.concat([loans_2007, dummy_df], axis=1) #列向拼接哑变量
loans_2007 = loans_2007.drop(cat_columns, axis=1)
loans_2007 = loans_2007.drop("pymnt_plan", axis=1)

embarked_data = pd.get_dummies(data.Embarked)
embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
data = pd.concat([data, embarked_data], axis=1)

############

### 特征工程

###########

import xgboost as xgb
import numpy as np


## 数据集
from sklearn.model_selection import train_test_split   # cross_validation
#x, y = np.split(data, (4,), axis=1)
cols = loans.columns
train_cols = cols.drop("loan_status")
features = loans[train_cols]
target = loans["loan_status"]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)
data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)


#稀疏矩阵
def read_data(path):  #将紧密型格式转换为稀疏矩阵
    y = []
    row = []
    col = []
    values = []
    r = 0       # 首行
    for d in open(path):
        d = d.strip().split()      # 以空格分开
        y.append(int(d[0]))
        d = d[1:]
        for c in d:
            key, value = c.split(':')
            row.append(r)
            col.append(int(key))
            values.append(float(value))
        r += 1
    x = scipy.sparse.csr_matrix((values, (row, col))).toarray()
    y = np.array(y)
    return x, y


## 模型
# 定义f: theta * x
#自定义损失函数的梯度和二阶导 因为下面使用的是逻辑回归来做二分类,所以用逻辑回归的导师
def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h


def error_rate(y_hat, y):  #定义误差率
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print acc
    print tip + '正确率：\t', float(acc.sum()) / a.size





param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}  # logitraw
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3} #多分类问题,需要编码为0开头
# md:树最大深度,eta:防止拟合的惩罚因子 eat越小越能防止过拟合,silent:是否输出,objective:二分类逻辑回归,目标成本函数
# param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
watchlist = [(data_test, 'eval'), (data_train, 'train')]
# 因为想输出在测试集上的误差率的情况,这边就看了
n_round = 3  # 3棵决策树
# bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate) #添加了损失函数的梯度,和错误的评估方法

#
# 这里要重点讲一下 Xgboost 的调参。通常认为对它性能影响较大的参数有：
#
# eta：每次迭代完成后更新权重时的步长。越小训练越慢。
# num_round：总共迭代的次数。
# subsample：训练每棵树时用来训练的数据占全部的比例。用于防止 Overfitting。
# colsample_bytree：训练每棵树时用来训练的特征的比例，类似 RandomForestClassifier 的 max_features。
# max_depth：每棵树的最大深度限制。与 Random Forest 不同，Gradient Boosting 如果不对深度加以限制，最终是会 Overfit 的。
# early_stopping_rounds：用于控制在 Out Of Sample 的验证集上连续多少个迭代的分数都没有提高后就提前终止训练。用于防止 Overfitting。
# 一般的调参步骤是：
# 1将训练数据的一部分划出来作为验证集。
# 2先将 eta 设得比较高（比如 0.1），num_round 设为 300 ~ 500。
# 3用 Grid Search 对其他参数进行搜索
# 4逐步将 eta 降低，找到最佳值。
# 5以验证集为 watchlist，用找到的最佳参数组合重新在训练集上训练。注意观察算法的输出，看每次迭代后在验证集上分数的变化情况，从而得到最佳的 early_stopping_rounds。



X_dtrain, X_deval, y_dtrain, y_deval = cross_validation.train_test_split(X_train, y_train, random_state=1026, test_size=0.3)
dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
deval = xgb.DMatrix(X_deval, y_deval)
watchlist = [(deval, 'eval')]
params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'eta': 0.05,
    'max_depth': 7,
    'seed': 2016,
    'silent': 0,
    'eval_metric': 'rmse'
}
clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
pred = clf.predict(xgb.DMatrix(df_test))



## 评估
from sklearn.metrics import confusion_matrix #Confusion Matrix 混淆矩阵
confusion=confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,1]





























