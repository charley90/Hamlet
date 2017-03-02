#!/usr/bin/python
# -*- coding:utf-8 -*-




##字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# from matplotlib.font_manager import FontProperties
# font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
# fontproperties=font_set


## 坐标轴拓展使用 让显示的数据不在最边上
def expand(a, b):
    d = (b - a) * 0.05
    return a-d, b+d


#自定义显示函数  显示的是:训练集的残差，测试集的残差，各个系数的大小
fig, ax = plot_residuals_and_coeff(resid_train, resid_test, model.coef_);
def plot_residuals_and_coeff(resid_train, resid_test, coeff):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].bar(np.arange(len(resid_train)), resid_train) # 各个样本对应的残差
    axes[0].set_xlabel("sample number")
    axes[0].set_ylabel("residual")
    axes[0].set_title("training data")
    axes[1].bar(np.arange(len(resid_test)), resid_test) # 各个样本对应的残差
    axes[1].set_xlabel("sample number")
    axes[1].set_ylabel("residual")
    axes[1].set_title("testing data")
    axes[2].bar(np.arange(len(coeff)), coeff) # 各个变量的系数
    axes[2].set_xlabel("coefficient number")
    axes[2].set_ylabel("coefficient")
    fig.tight_layout()
    return fig, axes



N, M = 50, 50  # 横纵各采样多少个值
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
print x_show.shape



# # 无意义，只是为了凑另外两个维度 另外一种写法
# # 打开该注释前，确保注释掉x = x[:, :2]
# x3 = np.ones(x1.size) * np.average(x[:, 2])
# x4 = np.ones(x1.size) * np.average(x[:, 3])
# x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  #浅色背景表示分块区域
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])  #深色表示样本点
y_show_hat = model.predict(x_show)  # 预测值
print y_show_hat.shape
print y_show_hat
y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
print y_show_hat
plt.figure(facecolor='w')  #开始背景色为白色
plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示  用浅色背景色填充分割区域
print y_test
print y_test.ravel()
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', s=120, cmap=cm_dark, marker='*')  # 用星号深色表示测试集
plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据 用深色填充
plt.xlabel(iris_feature[0], fontsize=15)
plt.ylabel(iris_feature[1], fontsize=15)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(True)
plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
plt.show()







# 过拟合：错误率 错误率作图
depth = np.arange(1, 15)  #设置树的深度
err_list = []  #构建数组存放错误率
for d in depth:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
    clf = clf.fit(x_train, y_train)
    y_test_hat = clf.predict(x_test)  # 测试数据
    result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
    err = 1 - np.mean(result)
    err_list.append(err)
    # print d, ' 准确度: %.2f%%' % (100 * err)
    print d, ' 错误率: %.2f%%' % (100 * err)
plt.figure(facecolor='w')
plt.plot(depth, err_list, 'ro-', lw=2)
plt.xlabel(u'决策树深度', fontsize=15)
plt.ylabel(u'错误率', fontsize=15)
plt.title(u'决策树深度与过拟合', fontsize=17)
plt.grid(True)
plt.show()

# 3D作图
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, facecolor='w')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(d[1], d[0], d[2], c='r', s=100*density, marker='o', depthshade=True)
ax.set_xlabel(u'红色分量')
ax.set_ylabel(u'绿色分量')
ax.set_zlabel(u'蓝色分量')
plt.title(u'图像颜色三维频数分布', fontsize=20)

#使用调色盘表征不同的类别

cm = matplotlib.colors.ListedColormap(list('rgbm')) #调色盘设置 聚类有标签可以直接用标签

clrs = plt.cm.Spectral(np.linspace(0, 0.8, n_clusters))

plt.subplot(3, 3, i + 1)
plt.title(u'Preference：%.2f，簇个数：%d' % (p, n_clusters))
clrs = []
for c in np.linspace(16711680, 255, n_clusters):  # 聚类时开始类别数不知道,构造调色盘
    clrs.append('#%06x' % c)
# clrs = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
for k, clr in enumerate(clrs):
    cur = (y_hat == k)
    plt.scatter(data[cur, 0], data[cur, 1], c=clr, edgecolors='none')
    center = data[center_indices[k]]
    for x in data[cur]:
        plt.plot([x[0], center[0]], [x[1], center[1]], color=clr, zorder=1)
plt.scatter(data[center_indices, 0], data[center_indices, 1], s=100, c=clrs, marker='*', edgecolors='k', zorder=2)

cmesh = plt.pcolormesh(x1, x2, grid_hat, cmap=plt.cm.Spectral)
plt.colorbar(cmesh, shrink=0.9)

#聚类中心用星型表示
plt.grid(True)



##打印等高线
p = gmm.predict_proba(grid_test)  # 获取这个预测值得置信概率
p = p[:, 0].reshape(x1.shape)  # 属于第一类的概率,所以yaxis取0
CS = plt.contour(x1, x2, p, levels=(0.2, 0.5, 0.8), colors=list('rgb'), linewidths=2)  # 将置信概率为levels的等高线绘制出来
plt.clabel(CS, fontsize=15, fmt='%.1f', inline=True)  # 在线上打印出数值 ,inline默认为true表示在线上面
ax1_min, ax1_max, ax2_min, ax2_max = plt.axis()

##绘制多bar图
xpos = np.arange(4)
ax = plt.axes()
b1 = ax.bar(xpos - 0.3, err, width=0.3, color='#77E0A0')  # 做两个靠在一起的bar图
b2 = ax.twinx().bar(xpos, bic, width=0.3, color='#FF8080')  # 前面-0.3 这边是0.3 #同时在右轴上使用新的刻度
plt.grid(True)
bic_min, bic_max = expand(bic.min(), bic.max())
plt.ylim((bic_min, bic_max))
plt.xticks(xpos, types)
plt.legend([b1[0], b2[0]], (u'错误率', u'BIC'))
plt.title(u'不同方差类型的误差率和BIC', fontsize=18)
plt.show()


## 绘制分布中心色块图
clrs = list('rgbmy')
for i, (center, cov) in enumerate(zip(centers, covs)): #(centers,convs)是各个类型的(均值,方差)
    value, vector = sp.linalg.eigh(cov) #做特征分解
    width, height = value[0], value[1] #获取长和宽
    v = vector[0] / sp.linalg.norm(vector[0]) #做轴线旋转
    angle = 180 * np.arctan(v[1] / v[0]) / np.pi #计算角度
    e = Ellipse(xy=center, width=width, height=height,
                angle=angle, color=clrs[i], alpha=0.5, clip_box=ax.bbox)
    ax.add_artist(e)