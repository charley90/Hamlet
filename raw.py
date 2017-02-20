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

## 不同市场之间,可能法定假日不一样,所以需要别的合并方法.
# new = (chinastocks.index | usstocks.index)
# both = pd.DataFrame(index = new, columns = ['600208.XSHG','FXP.US'])
# both['600208.XSHG'] = chinastocks['ClosingPx']
# both['FXP.US'] = usstocks['ClosingPx']
# both


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



##########################################################################################
impoort datetime as dt
impoort matplotlib.pyplot as plt
from matpoltlib import style
import pandas as pd
imort pandas_datareader.data as web

style.use(‘ggplot’)
start=dt.datetime(2000,1,1)
end=dt.datetime(2016.12.31)


df=pd.read_csv(‘tsla.csv’,parse_dates=True,index_col=0)
df=web.DateReader(‘TSLA’,’yahoo’,start,end ) #从雅虎读取数据
print  df.head()

df[‘100ma’]=df[‘Adg Close’].roling(window=100,min_periods=0).mean()#min_periods 小数位 使用TAlib替代这些
df.dropna(implasce=True)#inplace 替换原来的

ax1=plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax1=plt.subplot2grid((6,1),(5,0),rowspan=5,colspan=1,sharex=ax1)
ax1.plot(df.index,df[‘Adj Close’])
ax1.plot(df.index,df[‘100ma’])
ax2.bar(df.index,df[‘volumment’])
plt.show()


#resample 时间框架转换
from matpltlib.finance import candlestick_ohlc
from matplotlib.dates as mdates


df_ohlc=df[‘Adj Close’].resample(’10D’).ohlc() #高开低收转换后还是高开低收
df_volume=df[‘Volume’],resample(’10D’).sum()  #交易量数据加和

df_ohlc.reste_index(inplase=True)
df_ohlc[‘Date’]=df_ohlc[‘Date’].map(mdates.date2num)
candlestick_ohlc(ax1,df_ohlc.values,width2,colorup=‘g’)
ax2.fill_between( df_volume.index.map(mdates.date2num),df_volumes,0))

#构建组合
import bs4 as bs
import pickle
import requests

def save_sp500_tickers():  #获取股票池信息
	resp=requests.get(‘’)
 	soup=bs.BeautifulSoup(resp.txt,’lxml’)
	table=soup.find(‘table’,{‘class’:’wikitable sortable’})
	tickers=[]
	for row in table.findAll(‘tr’)[1:]:
		ticker=row.findAll(‘td’)[0].text
		tickers.append(ticker)
	with open(‘sp300tickers.pickle’,’wb’) as f:
		pickle.dump(tickers,f)
	return tickers

def get_data_from_yahoo(reload_sp500=False): #研究如何将这个数据移动到数据库之中
	if reload_sp500:
		tickers=save_sp500_tickers()
	else:
		with open(‘sp300tickers.pickle’,’wb’) as f:
			tickers=pickle.load(f)
	if not os.path.exists(‘stock_dfs’):
		os.makedirs(‘stock_dfs’)

	start=dt.datetime(200.1,1)
	end=dt.datetime(2016.12.31)

	for ticker in tickers:
		print (ticker)
		if not os.path.exists(‘stock_dfs/{}.csv’fromat(ticker)):
			df=wb.DataReader(ticker,’yahoo’,start,end)
			df.to_csv()
		else:
			print(‘Alread have {}’.format(ticker))


def compile_data():  #拼接股票信息
	with open(‘sp400tickeres.pickle’,’’rb’) as f:
		tickers=pickle.load(f)
	main_df=pd.DataFrame()
	for count ,ticker in enumerate(tickers):
		df=pd.read_csv(‘stock_dfs/{}.csv’format(ticker))
		df.set_index(‘Date’,inplacce=Ture)
		df.rename(columns={‘Adj Close’,ticker},inplace=Ture)
		df.drop([‘Open’,’High’,,,],1,inplace=True) #拼接完的表格中去掉不需要的字段

	 	if main_df.empty:
			main_df=df
		else:
			main_df=main_df.join(df,how=‘outer’)

		if count%10==0: #每十个打印一下
			print(count)

	main_df.to_csv()


def visualize_date(): #可视化数据 热力图
	df=pd.read_csv()
	df_corr=df.corr()
	data=df_corr.values
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	heatmap=ax.pcolor(data,cmap=plt.cm.TdYlGn)
	fig.colorbar(heatmap)
	ax.set_xticks(np.arange(data.shape[0]+0.5,minor=False))	 #左移形成网格
 	ax.set_yticks(np.arange(data.shape[1]+0.5,minor=False))
	ax.inver_yaxis() #去掉没有填充的格子,截断无穷项
	ax.xaxis.tick_top()

	column_lables=df_corr.columns
	row_labels=df_corr.index

	ax.set_xtickelabes(column_labels)
	ax.set_ytickelabes(raw_labels)
	plt.xticks(rotation=90)
	heatmap.set_clim(-1,1) #颜色的对应的值
	plt.tight_layout()
	plt.show()


def process_data_for_labels(ticker): #制作标签
	hm_day=7 # 未来7天内的收益率能否达到要求
	df=pd.read_csv(‘sp500_join_closes.csv’,index_col=0)
	tickers=df.columns.values.tolist()
	df.fillna(0,inplace=True)

	for i in range(i,hm_days+1)
		df[‘{}_{}d’.format(ticker,i)]=(df[ticker].shift(-i)-df[ticker])/df[ticker]
	df.fillna(0,inplace=True)
	return tickers,df

#process_data_for_labels(‘XOM’)

def buy_sell_hold(*args):
	cols=[c for c in args]
	requirement=0.02 #要求的回报率
	for col in cols: #这种写法没有考虑到出现的先后问题,需要改进
		if col>requirement: return 1
		if col<-requirement: return -1
	return 0


from collections import Counter
def extract_featuresets(ticker):  #添加特征
	tickers,df=process_date_for_labels(tickeer)

	df[‘{}_target’.format(ticker)]=list(map(buy_sell,hold,
							df[‘{}_1d’.format(ticker)],
							df[‘{}_2d’.format(ticker)],
							df[‘{}_3d’.format(ticker)],
							df[‘{}_4d’.format(ticker)],
							df[‘{}_5d’.format(ticker)],
							df[‘{}_6d’.format(ticker)],
							df[‘{}_7d’.format(ticker)]
						))
	vals=df[‘{}_targe’.format(ticker)].values.tolist()
	str_vals=[str(i) for i in vals]
	print(‘Data spread:’,Counter(str_vals)) #统计序列中的正负例

	df.fillna(0,inplace=True)
	df=df.replace([np.inf,-np.inf],np.nan)
	df.fropna(inplace=True)

        #随便给的特征
	df_vals=df[[ticker for ticker in tickers]].pct_change()
	df_vals=df_vals.replace([np.inf,-np.inf],0)
	df_vals.fillna(0,inplace=True)

	X=df_vals.valutes
	y=df[‘{}_target’.format(ticker)].values

	return X ,y,df


from sklearn import svm,cross_validation,neighbors
from sklearn.emsemble import VotingClassifier,RandomForestClassifier


def do_ml(ticker):
	X,y,df=def extract_featuresets(ticker)

	X-train,X_test,y_train,y_test=cross_valiadation.train_test_split(X,y,test_size=0.75)

	#clf=neighbors.KNeighborsClassifier()
	clf=VotingClassifier([(‘lsvc’,svm.LinearSVC),
					(‘knn’,neighbors.KNeighorsClassifier())
					(‘rfor’,RandomForestClassifier())
					])


	clf.fit(X_train,y_train)
	confindence=clf.score(X_test,y_test)
	predictions=clf.predict(X_test)
	print(‘Predicted spreda:’ Counter(predictions))

	return confidence





