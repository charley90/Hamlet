#!/usr/bin/python
# -*- coding:utf-8 -*-




import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


import sklearn.datasets as ds
#from matplotlib import rcParams
#rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [u'STHeitiSC-Light']})
mpl.rcParams['font.sans-serif'] = [u'Hei'] #绘图中的中文问题
mpl.rcParams['axes.unicode_minus'] = False


##屏蔽掉警告的操作
import warnings
warnings.filterwarnings("ignore") # 屏蔽掉讨厌的警告





##生成样本

N = 400
centers = 4
data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)# 生成聚类中心为centers,特征维度为2维的样本400个,随机数种子2
data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1, 2.5, 0.5, 2), random_state=2)# 指定了方差不相同
data3 = np.vstack((data[y == 0][:], data[y == 1][:50], data[y == 2][:20], data[y == 3][:5]))
y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5) # 构造了不平衡样本

m = np.array(((1, 1), (1, 3)))
data_r = data.dot(m) #构造旋转后的样本

centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]] #指定聚类中心
data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)


from scipy.stats import multivariate_normal #构造数据分布
np.random.seed(0)
mu1_fact = (0, 0, 0)
# cov_fact = np.diag(2) #对角线
# cov_fact = np.array([(1,2,3),(2,3,4),(3,4,5)])
cov_fact = np.identity(3)
data1 = np.random.multivariate_normal(mu1_fact, cov_fact, 400)  # 数据分布
mu2_fact = (2, 2, 1)
cov_fact = np.identity(3)  # I
data2 = np.random.multivariate_normal(mu2_fact, cov_fact, 100)
data = np.vstack((data1, data2))  # 样本堆叠
y = np.array([True] * 400 + [False] * 100)  # 标签生成一个是400个,一个是100个





##使用CSV读取数据
def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

path = '10.Advertising.csv'  # 设置文件路径,因为在一个包内优先会检查本地的
data = pd.read_csv(path, dtype=float, delimiter=',', converters={4: iris_type}) #header=None
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





##数据集处理
#将数据按0.8分为训练集合测试集,其中random_state为随机数种子便于重新测试验证
from sklearn.model_selection import train_test_split  #训练集和测试集样本分离
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

#多类别标签编码
from sklearn import preprocessing
enc = preprocessing.LabelEncoder()
enc.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
y = enc.transform(y)
print enc.classes_

#使用pandas 转换哑变量  转换哑变量会变成多列和标签编码方式是不同的
#哑变量处理 最重要的原因,数据可乘性   0.5*2(红色)=1(绿色) 就非常不合适了,分成多列,一般用于类型数据非序列编码类型
#onehot 编码 类型数据的多项化(PolynomialFeatures).特征为啥多的原因. 稀疏数据的存储 记做64:1 第64个特征为1
#3个特征用两个!!!这样子做可以避免列之间的相关性. 性别数据只有0,1 似乎也可以不分0*n=0吧
input_df.Sex = input_df.Sex.map({"male": 1, "female": 0}) # 首先做个映射 类似于上面的标签编码
dummies_Sex = pd.get_dummies(input_df['Sex'], prefix= 'Sex') #将性别列转换成哑变量
dummies_Pclass = pd.get_dummies(input_df['Pclass'], prefix= 'Pclass')

input_df = pd.concat([input_df, dummies_Sex, dummies_Pclass], axis=1) #添加哑变量
input_df.drop(['Pclass','Sex'], axis=1, inplace=True) #删除掉原来的列
input_df.head()
#使用sklearn 处理哑变量问题
from sklearn.preprocessing import OneHotEncoder
OneHotEncoder().fit_transform(iris.target.reshape((-1,1)))



#处理缺失值
#重要的数据可以建模预测,如泰坦尼克号生存预测中的年龄,可否用姓名中的信息等建立模型预测可能比用整个船舱的平均年龄估计要好
#重要性高的,也可以通过业务逻辑,或经验预估
from sklearn.preprocessing import Imputer
Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))
# imp = Imputer(missing_values=0, strategy='mean', axis=0)
# imp. t_transform(X_train)

#对于某些特征进行处理
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler() # scaling，其实就是将一些变化幅度较大的特征化到[-1,1]之内。
age_scale_param = scaler.fit(input_df['Age']) #仅仅是对指定的特征做处理
input_df['Age_scaled'] = scaler.fit_transform(input_df['Age'], age_scale_param)

from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(). fit(X_train)
# standardized_X = scaler.transform(X_train)# transform方法是对所有的特征都进行了除了
StandardScaler().fit_transform(iris.data)  #两步可以合并成一步写
#1）对数据先fit，再transform，好处是我可以拿到数据变换(比如scaling/幅度变换/标准化)的参数，这样你可以在测试集上也一样做相同的数据变换处理
#2）fit_trainsform，一次性完成数据的变换(比如scaling/幅度变换/标准化)，比较快。
# 但是如果在训练集和测试集上用fit_trainsform，可能执行的是两套变换标准(因为训练集和测试集幅度不一样)


from sklearn.preprocessing import Normalizer #其通过求z-score的方法，将样本的特征值转换到同一量纲下
scaler = Normalizer(). fit(X_train)
normalized_X = scaler.transform(X_train)

from sklearn.preprocessing import Binarizer #定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
binarizer = Binarizer(threshold=0.0). fit(X)
binary_X = binarizer.transform(X)


#多项式转换
from sklearn.preprocessing import PolynomialFeatures
PolynomialFeatures().fit_transform(iris.data) #参数degree为度，默认值为2

#自定义转换函数为对数函数的数据变换
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
FunctionTransformer(log1p).fit_transform(iris.data)#第一个参数是单变元函数


#只需要对特征矩阵的某些列进行转换，而不是所有列
#还有一种方法就将特征拆分成多个部分,然后再使用pd拼接的方法  pd.merge(left, right, on='key') # 按列合并 join
from sklearn.pipeline import FeatureUnion, _fit_one_transformer, _fit_transform_one, _transform_one
from sklearn.externals.joblib import Parallel, delayed
from scipy import sparse
import numpy as np

#部分并行处理，继承FeatureUnion
class FeatureUnionExt(FeatureUnion):
    #相比FeatureUnion，多了idx_list参数，其表示每个并行工作需要读取的特征矩阵的列
    def __init__(self, transformer_list, idx_list, n_jobs=1, transformer_weights=None):
        self.idx_list = idx_list
        FeatureUnion.__init__(self, transformer_list=map(lambda trans:(trans[0], trans[1]), transformer_list), n_jobs=n_jobs,
                              transformer_weights=transformer_weights)

    #由于只部分读取特征矩阵，方法fit需要重构
    def fit(self, X, y=None):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        transformers = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入fit方法
            delayed(_fit_one_transformer)(trans, X[:,idx], y)
            for name, trans, idx in transformer_idx_list)
        self._update_transformer_list(transformers)
        return self

    #由于只部分读取特征矩阵，方法fit_transform需要重构
    def fit_transform(self, X, y=None, **fit_params):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        result = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入fit_transform方法
            delayed(_fit_transform_one)(trans, name, X[:,idx], y,
                                        self.transformer_weights, **fit_params)
            for name, trans, idx in transformer_idx_list)

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    #由于只部分读取特征矩阵，方法transform需要重构
    def transform(self, X):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        Xs = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入transform方法
            delayed(_transform_one)(trans, name, X[:,idx], self.transformer_weights)
            for name, trans, idx in transformer_idx_list)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

#我们对特征矩阵的第1列（花的颜色）进行定性特征编码，对第2、3、4列进行对数函数转换，对第5列进行定量特征二值化处理。使用FeatureUnionExt类进行部分并行处理的代码如下：
from numpy import log1p
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer

#新建将部分特征矩阵进行定性特征编码的对象
step2_1 = ('OneHotEncoder', OneHotEncoder(sparse=False))
#新建将部分特征矩阵进行对数函数转换的对象
step2_2 = ('ToLog', FunctionTransformer(log1p))
#新建将部分特征矩阵进行二值化类的对象
step2_3 = ('ToBinary', Binarizer())
#新建部分并行处理对象
#参数transformer_list为需要并行处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
#参数idx_list为相应的需要读取的特征矩阵的列
step2 = ('FeatureUnionExt', FeatureUnionExt(transformer_list=[step2_1, step2_2, step2_3], idx_list=[[0], [1, 2, 3], [4]]))




#流水线上除最后一个工作以外，其他都要执行fit_transform方法，且上一个工作输出作为下一个工作的输入。最后一个工作必须实现fit方法，输入为上一个工作的输出；
# 但是不限定一定有transform方法，因为流水线的最后一个工作可能是训练！
from numpy import log1p
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline  #using a list of (key, value) pairs 使用键值对形式

#新建计算缺失值的对象
step1 = ('Imputer', Imputer())
#新建将部分特征矩阵进行定性特征编码的对象
step2_1 = ('OneHotEncoder', OneHotEncoder(sparse=False))
#新建将部分特征矩阵进行对数函数转换的对象
step2_2 = ('ToLog', FunctionTransformer(log1p))
#新建将部分特征矩阵进行二值化类的对象
step2_3 = ('ToBinary', Binarizer())
#新建部分并行处理对象，返回值为每个并行工作的输出的合并
step2 = ('FeatureUnionExt', FeatureUnionExt(transformer_list=[step2_1, step2_2, step2_3], idx_list=[[0], [1, 2, 3], [4]]))
#新建无量纲化对象
step3 = ('MinMaxScaler', MinMaxScaler())
#新建卡方校验选择特征的对象
step4 = ('SelectKBest', SelectKBest(chi2, k=3))
#新建PCA降维的对象
step5 = ('PCA', PCA(n_components=2))
#新建逻辑回归的对象，其为待训练的模型作为流水线的最后一步
step6 = ('LogisticRegression', LogisticRegression(penalty='l2'))
#新建流水线处理对象
#参数steps为需要流水线处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
pipeline = Pipeline(steps=[step1, step2, step3, step4, step5, step6])
pipeline.steps[0] #查询pipeline的的步骤
pipeline.set_params(clf__C=10) #重新设置pipeline 的参数值




# 使用网格调藏
from klearn.model_selection import GridSearchCV

#新建网格搜索对象
#第一参数为待训练的模型
 #param_grid为待调参数组成的网格，字典格式，键为参数名称（格式“对象名称__子对象名称__参数名称”），值为可取的参数值列表
 grid_search = GridSearchCV(pipeline, param_grid={'FeatureUnionExt__ToBinary__threshold':[1.0, 2.0, 3.0, 4.0],...
     'LogisticRegression__C':[0.1, 0.2, 0.4, 0.8]})
#训练以及调参
grid_search.fit(iris.data, iris.target)




## 特征选择
#通过相关系数 选择K个最好的特征，返回选择特征后的数据
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#参数k为选择的特征个数

#使用包裹
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数



##建模  模型就是那些普世的通用性的方法
#线性回归
from sklearn.linear_model import LinearRegression     #线性回归模型
linreg = LinearRegression()
linreg.fit(x_train, y_train)
#lasso,ridge 线性回归
from sklearn.linear_model import Lasso, Ridge         #线性模型引入lasso,ridge
model = Lasso() #S1 先建立整个模型
lasso_model.fit(x_train, y_train) #S3使用模型进行fit
#model = linear_model.LassoCV() # 可以去尝试不同的参数值 实际用时用 LassoCV 自动找出最好的 alpha
#model.alpha_ # 自动找出最好的 alpha
ridge = RidgeCV(alphas=np.logspace(-3, 2, 10), fit_intercept=False) #带有CV自带有参数选择
lr = LogisticRegression(penalty='l2') #带有l2正则项的逻辑回归
penalty = {
    0: 5,
    1: 1
}
lr = LogisticRegression(class_weight=penalty) #解决类别不平衡问题 降采样



#同时进行多个模型测试
models = [Pipeline([
    ('poly', PolynomialFeatures()),
    ('linear', LinearRegression(fit_intercept=False))]),
    Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]),
    Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]),
    Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', ElasticNetCV(alphas=np.logspace(-3, 2, 50), l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                fit_intercept=False))])
]
for t in range(4):
    model = models[t]
    for i, d in enumerate(d_pool):
        model.set_params(poly__degree=d)  #重新设置参数
        model.fit(x, y.ravel())
        lin = model.get_params('linear')['linear']
        output = u'%s：%d阶，系数为：' % (titles[t], d)
        if hasattr(lin, 'alpha_'):
            idx = output.find(u'系数')
            output = output[:idx] + (u'alpha=%.6f，' % lin.alpha_) + output[idx:]
        if hasattr(lin, 'l1_ratio_'):  # 根据交叉验证结果，从输入l1_ratio(list)中选择的最优l1_ratio_(float)
            idx = output.find(u'系数')
            output = output[:idx] + (u'l1_ratio=%.6f，' % lin.l1_ratio_) + output[idx:]
        print output, lin.coef_.ravel()
        x_hat = np.linspace(x.min(), x.max(), num=100)
        x_hat.shape = -1, 1
        y_hat = model.predict(x_hat)
        s = model.score(x, y)
        r2, corr_coef = xss(y, model.predict(x))
        # print 'R2和相关系数：', r2, corr_coef
        # print 'R2：', s, '\n'
        z = N - 1 if (d == 2) else 0
        label = u'%d阶，$R^2$=%.3f' % (d, s)
        if hasattr(lin, 'l1_ratio_'):
            label += u'，L1 ratio=%.2f' % lin.l1_ratio_
        plt.plot(x_hat, y_hat, color=clrs[i], lw=line_width[i], alpha=0.75, label=label, zorder=z)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title(titles[t], fontsize=18)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)



#使用管道方法流程化处理 逻辑回归
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

lr = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])  #先进行标准化在进行逻辑回归处理
lr.fit(x, y.ravel()) #y.ravel()是将列向量转置成为行向量

#决策树方法
from sklearn.tree import DecisionTreeClassifier #引入决策树的方法
#决策树容易过拟合的原因是深度太深或太浅,本质来说都是leaf节点的样本数目不够所以需要对 分支进行规范
# min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
# min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则完成分支；否则，不进行分支
model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
#使用的ID3模式  还可以为gini  mse 使用mse的时候为回归决策树
#决策树的深度和特征关, 为2^n n为特征列  两个特征的时候可以可视化的展示
model = model.fit(x_train, y_train)
y_test_hat = model.predict(x_test)      # 测试数据


#决策树的保存
from sklearn import tree
import pydotplus  #绘图模块

# 保存
# dot -Tpng my.dot -o my.png
# 1、输出
with open('iris.dot', 'w') as f:  #需要安装graphviz
    tree.export_graphviz(model, out_file=f)
# 2、给定文件名
# tree.export_graphviz(model, out_file='iris.dot')
# 3、输出为pdf格式
dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_E, class_names=iris_class,
                                filled=True, rounded=True, special_characters=True) #out_file=None 抑制文件输出,结果在下面用
graph = pydotplus.graph_from_dot_data(dot_data) #graph 为中间件
graph.write_pdf('iris.pdf')
f = open('iris.png', 'wb')
f.write(graph.create_png())
f.close()

#随机森林
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=3)
rfc = RandomForestClassifier(100, criterion='gini', min_samples_split=2,
                             min_impurity_split=1e-10, bootstrap=True, oob_score=True) #OOB是边跑边测的准确率
rf_clf = clf.fit(x, y.ravel())

from sklearn.ensemble import BaggingRegressor #bagging方法是集成学习的一部分
dtr = DecisionTreeRegressor(max_depth=5) #使用集成的方法对于弱分类器有较好的效果,强分类器提升不大
BaggingRegressor(dtr, n_estimators=100, max_samples=0.3) #一个决策树,样本又放回抽样率为0.3.这么小为就有了不错的随机性,可以跳出局部最优


##GBDT
from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = GradientBoostingRegressor(**params)



## SVM
from sklearn import svm

clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr',class_weight={-1: 1, 1: 10})
# c为松弛系数 越大拟合越好,泛化越差,kernel为核函数,gamma为因子 越大样本越独立,多输出问题:ovr为onevsrest,ovo为onevsone,class_weight解决类别不平衡问题
clf.fit(x_train, y_train.ravel())

clf.decision_function(grid_test)    # 样本到决策面的距离
clf.n_support_ #支撑向量的数目
clf.dual_coef_ #支撑向量的系数
clf.support_ #支撑向量

params = {'C': np.logspace(0, 3, 7), 'gamma': np.logspace(-5, 0, 11)}
# model = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=params, cv=3)
svr.best_estimator_.support_  #最有估计的支撑向量序号方便标出


svr_rbf = svm.SVR(kernel='rbf', gamma=0.2, C=100)
svr_linear = svm.SVR(kernel='linear', C=100)
svr_poly = svm.SVR(kernel='poly', degree=3, C=100)



##Kmeans
from sklearn.cluster import KMeans
cls = KMeans(n_clusters=4, init='k-means++')
#关键的问题一个是类别数目的选择可能用canopy,还有初始化的规则.另外使用谱聚类开始做下维度优化
y_hat = cls.fit_predict(data)
#在正态分布数据,类圆,均方差上面表现较好 ,对于异常点没有什么好办法


##meanshift和K-means都属于中心迭代的方法,不过这种是指定了一个半径,然后不断向密度中心移动的办法
m = euclidean_distances(data, squared=True)
bw = np.median(m) #同样实用均值的办法来构造开始的点
for i, mul in enumerate(np.linspace(0.1, 0.4, 4)):
    band_width = mul * bw  #模型的均值距离以此为表征
    model = MeanShift(bin_seeding=True, bandwidth=band_width)
    ms = model.fit(data)
    centers = ms.cluster_centers_
    y_hat = ms.labels_
    n_clusters = np.unique(y_hat).size
    print '带宽：', mul, band_width, '聚类簇的个数为：', n_clusters

    plt.subplot(2, 2, i + 1)
    plt.title(u'带宽：%.2f，聚类簇的个数为：%d' % (band_width, n_clusters))
    clrs = []
    for c in np.linspace(16711680, 255, n_clusters):
        clrs.append('#%06x' % c)
    # clrs = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
    print clrs
    for k, clr in enumerate(clrs):
        cur = (y_hat == k)
        plt.scatter(data[cur, 0], data[cur, 1], c=clr, edgecolors='none')
    plt.scatter(centers[:, 0], centers[:, 1], s=150, c=clrs, marker='*', edgecolors='k')
    plt.grid(True)
plt.tight_layout(2)
plt.suptitle(u'MeanShift聚类', fontsize=20)
plt.subplots_adjust(top=0.92)
plt.show()



##SC 谱聚类 谱聚类开始用随机游走的思想圈定特征处理后的样本集,然后在处理后的样本集上做K-means 能够除了较特殊的
from sklearn.cluster import spectral_clustering
n_clusters = 3
m = euclidean_distances(data, squared=True)
sigma = np.median(m)
for i, s in enumerate(np.logspace(-2, 0, 6)):
    print s
    af = np.exp(-m ** 2 / (s ** 2)) + 1e-6  #RBF做映射,高斯相似度,后面的1e-6是为了避免0值出现
    y_hat = spectral_clustering(af, n_clusters=n_clusters, assign_labels='kmeans', random_state=1)



##DBSCAN  密度聚类的典型方法,比较适用于地图等不均匀,不规则图形,要求内部致密性比较好
from sklearn.cluster import DBSCAN
params = ((0.5, 3), (0.5, 5), (0.5, 10), (1., 3), (1., 10), (1., 20))
for i in range(6):
    eps, min_samples = params[i]  # eps 为条件密集可达的半径 min_samples 为在这个密度类要求的最小的点的个数
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(data)
    y_hat = model.labels_#序列的标签
    y_unique = np.unique(y_hat)
    n_clusters = y_unique.size - (1 if -1 in y_hat else 0)  # -1表示的是噪音点
    print y_unique, '聚类簇的个数为：', n_clusters


## AP 吸引子模型
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import euclidean_distances

m = euclidean_distances(data, squared=True)  # 计算点与点之间的欧拉距离
preference = -np.median(m)  # 用聚类的中位数数作为吸引的参数值 比较平均

plt.figure(figsize=(12, 9), facecolor='w')
for i, mul in enumerate(np.linspace(1, 4, 9)):
    print mul
    p = mul * preference  # 吸引子这种超参数应该如何调整,这个例子给了一个方法,先用某个均值,然后在用一个附件的系数来调节.非常重要
    model = AffinityPropagation(affinity='euclidean', preference=p)
    af = model.fit(data)
    center_indices = af.cluster_centers_indices_  # 显示类别的数目
    n_clusters = len(center_indices)
    print ('p = %.1f' % mul), p, '聚类簇的个数为：', n_clusters
    y_hat = af.labels_  # 显示序列类别的标签

    plt.subplot(3, 3, i + 1)
    plt.title(u'Preference：%.2f，簇个数：%d' % (p, n_clusters))
    clrs = []
    for c in np.linspace(16711680, 255, n_clusters):  # 聚类是开始类别不知道,构造调色盘
        clrs.append('#%06x' % c)
    # clrs = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
    for k, clr in enumerate(clrs):
        cur = (y_hat == k)
        plt.scatter(data[cur, 0], data[cur, 1], c=clr, edgecolors='none')
        center = data[center_indices[k]]
        for x in data[cur]:
            plt.plot([x[0], center[0]], [x[1], center[1]], color=clr, zorder=1)
    plt.scatter(data[center_indices, 0], data[center_indices, 1], s=100, c=clrs, marker='*', edgecolors='k',
                zorder=2)  # 聚类中心用星型表示
    plt.grid(True)
plt.tight_layout()
plt.suptitle(u'AP聚类', fontsize=20)
plt.subplots_adjust(top=0.92)
plt.show()




## GMM模型也是无监督模型 使用的是EM算法
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
gmm= GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
# n_components 类别数据
# covariance_type 协方差类型 full是全部都估计,参数最多;diag是对对角线估计;tied是类别协方差矩阵相关;sperical是只有横和纵的
# max_iter  最大迭代轮数
# n_init 因为没有初始化估计先验分布,多迭代很多次,然后自己选择比较好的
# 使用GMM绘制图形如果有外面一块是里面的类别的没什么奇怪的一个分布矮胖罢了
dpgmm = BayesianGaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000, n_init=5,
                                weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=10)
# 狄利克雷过程高斯混合模型,能够过滤聚类中心的个数,即使给错了聚类个数也能很好的拟合

gmm.fit(data)
print '类别概率:\t', gmm.weights_[0]
print '均值:\n', gmm.means_, '\n'
print '方差:\n', gmm.covariances_, '\n'
mu1, mu2 = gmm.means_
sigma1, sigma2 = gmm.covariances_  # 这边可以看看方差和给出的是否一致

#指定类别顺序
order = pairwise_distances_argmin(m, gmm.means_, axis=1, metric='euclidean')
#就是算各个类别的均值,然后有小到大排个序?

bic[i] = gmm.bic(x) #自带BIC指标

y_hat = gmm.predict(x)
y_test_hat = gmm.predict(x_test)
# 调节顺序
change = (gmm.means_[0][0] > gmm.means_[1][0])
if change:
    z = y_hat == 0
    y_hat[z] = 1
    y_hat[~z] = 0
    z = y_test_hat == 0
    y_test_hat[z] = 1
    y_test_hat[~z] = 0

##调参数

#调参数的思想
m = euclidean_distances(data, squared=True) #计算点与点之间的欧拉距离
preference = -np.median(m) #用聚类的中位数数作为吸引的参数值 比较平均

for i, mul in enumerate(np.linspace(1, 4, 9)):
    print mul
    p = mul * preference #吸引子这种超参数应该如何调整,这个例子给了一个方法,先用某个均值,然后在用一个附件的系数来调节.非常重要



from sklearn.model_selection import GridSearchCV  #CV集调参数
alpha_can = np.logspace(-3, 2, 10) #调参数的范围 这种方法选择的参数近似于[0.01,0.03.0.1,0.3,1,3.10,30,100]
lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5),scoring='accuracy' #S2 选择最优参数建立成精模型
# #CV集为5 alpha为选择的范围返回最优的参数模型
print '超参数：\n', lasso_model.best_params_   #输出选择的最优参数

##一个调参数案例
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target
k_range=range(1,31)
weight_options=['uniform','distance']
param_grid=dict(n_neighbors=k_range,weights=weight_options) #建立计数器迭代网格 参数词n_neighbors =变换的参数

knn=KNeighborsClassifier()
grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(X,y)

# 在有权重参数的时候 不容易画图
grid.grid_scores_
grid_mean_scores=[result.mean_validation_score for result in grid.grid_scores_]
print grid_mean_scores

plt.plot(k_range,grid_mean_scores)
## 网格能够返回最优的参数和模型
print grid.best_score_
print grid.best_params_
print grid.best_estimator_
grid.predict([3,5,4,2]) # 网格默认使用最优的模型来模拟参数



##预测
#线性回归
y_hat = linreg.predict(np.array(x_test))
#分类回归
y_hat = lr.predict(x)
y_hat_prob = lr.predict_proba(x)


##评估
#线性回归
mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
print lasso_model.score(x_test, y_test) #R2

from sklearn.metrics import mean_absolute_error #Mean Absolute Error
y_true = [3, -0.5, 2]
mean_absolute_error(y_true, y_pred)

from sklearn.metrics import mean_squared_error #Mean Squared Error
mean_squared_error(y_test, y_pred)

from sklearn.metrics import r2_score #R2
r2_score(y_true, y_pred)


#分类回归
print u'准确度：%.2f%%' % (100*np.mean(y_hat == y.ravel()))
print(metrics.classification_report(y_test, y_test_pred)) # 报告查准率,召回率和f1
print metrics.confusion_matrix(y_test, y_test_pred) #混淆矩阵

from sklearn.metrics import accuracy_score #Accuracy Score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classi cation_report #Classfication Report
print(classi cation_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix #Confusion Matrix 混淆矩阵
confusion=confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,1]

print metrics.precision_score(y_test,y_pred) # TP/(TP+FP)
print metrics.recall_score(y_test,y_pred)  # TP/(TP+FN)
print metrics.f1_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support, classification_report

print 'Accuracy：\t', accuracy_score(y_true, y_hat)
print 'Precision:\t', precision
print 'Recall:  \t', recall
print 'f1 score: \t', f1_score(y_true, y_hat)
print precision_recall_fscore_support(y_true, y_hat, beta=1)
print classification_report(y_true, y_hat)


# 评估不同模型在训练样本比例不同的表现
# 值得学习的是非使用gridsearh进行多个模型的同时进行
train_size_vec = np.linspace(0.1, 0.9, 30) # 尝试不同的训练集样本大小比例
classifiers = [tree.DecisionTreeClassifier,  #定义分类器
               neighbors.KNeighborsClassifier,
               svm.SVC,
               ensemble.RandomForestClassifier
               ]
cm_diags = np.zeros((3, len(train_size_vec), len(classifiers)), dtype=float) # 用来放结评估果

for n, train_size in enumerate(train_size_vec): #循环训练集样本的大小比例
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(iris.data, iris.target, train_size=train_size)

    for m, Classifier in enumerate(classifiers):  #循环模型
        classifier = Classifier()
        classifier.fit(X_train, y_train)
        y_test_pred = classifier.predict(X_test)
        cm_diags[:, n, m] = metrics.confusion_matrix(y_test, y_test_pred).diagonal()
        cm_diags[:, n, m] /= np.bincount(y_test)

# 聚类效果 都是在有标签模型下面做的
from sklearn import metrics

h = metrics.homogeneity_score(y, y_hat) #同一性(Homogeneity)
c = metrics.completeness_score(y, y_hat) #完整性(Completeness)
v = metrics.v_measure_score(y, y_hat) #V-Measure

ari = metrics.adjusted_rand_score(y, y_hat) #ARI 指数

# 聚类效果在无标签模型下面
silhouette=metrics.silhouette_score(X,kmeans_model.lables_,metric='euclidean')
#轮廓系数, 因为是无标签模型,所以使用原始的特征和聚类后的标签,使用欧拉距离作为衡量

