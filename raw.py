import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def  symbol_to_path(symbol,base_dir="data"):
    #通过路径来读取symbol对应的csv文件
    return os.path.join(base_dir,"{}.csv".format(str(symbol)))

def get_data(symbols,dates):
    #读取给定symbols的调节后的收盘价
    df=pd.DataFrame(index=dates)#使用给定的日期来构造空白数据框

    if 'SPY' not in symbols:
        symbols.insert(0,'SPY') #将指数作为第一列插入，指数只要有报价说明都是交易日

    for symbol in symbols:
        df_temp=pd.read_csv(symbol_to_path(symbol),index_col='Date',parse_dates=True,...
                            usecols=['Date','Adj Close'],na_values=['nan'])
        #pandas读csv
        df_temp=df_temp.rename(clumns={'Adj Close':symbol}) #多个symbols 用名称来代替重复的列名
        df=df.join(df_temp) #关联
        if symbol=='SPY' : #如果插完了股指，那么通过股指来去掉不交易的日期
            df=df.dropna(subset=["SYP"])

    df.fillna(method='ffill',inplace="Ture") #先从前填空缺值，避免未来数据对回测的影响
    df.fillna(method='bfill',inplace="Ture") #再从后填空缺值，应对股票未上市

def plot_date(df,title='Stockprices'):
    #自定义画图函数
    ax=df.plot(title=title,fontsize=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

def plot_selected(df,columns,sd,ed):
    #绘制切片图片
    plot_date(df.ix[sd:ed,columns].plot(),title='Selected data')

def normalize_data(df)：
    #将价格跨度大的图片归一化
    return df/df.ix[0，：]





def test_run():
    dates=pd.date_range('2010-01-01','2016-12-31') #时间跨度范围
    symbols=['GOOG','IBM','GLD'] #股票池
    df=get_date(symbols,dates) #直接取到股票池中指定日期的部件


if __name__="__main__":
    test_run()

#分类器的结构化代码
class  LinRegLearner(object):
	"""docstring for  LinRegLearner"""
	def __init__(self, arg):
		super( LinRegLearner, self).__init__()
		self.arg = arg
	def train(X,Y)
		self.m,self.b=favorit_linreg(X,Y)
	def query(X)
		Y=self.m*X+self.b
		return Y






