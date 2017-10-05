import talib
import numpy as np
import pandas as p
from heapq import nsmallest

def initialize(context):
    # 策略参考标准
    set_benchmark('000001.XSHG')
    # 设置手续费，买入时万分之三，卖出时万分之三加千分之一印花税, 每笔交易最低扣5块钱
    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

    #初始化参数
    g.s1 = "000001.XSHG"
    g.OBSERVATION=30
    g.ma = 0
    g.lastma = 0
    g.valid_num_stocks = 8

    #初始化有历史数据的股票池
    g.stocks_hist=[]

#获得某天的量比
def Volume_rate(sec,time_day,context):
	Volume_last_5_day=history(5,'1d','volume',sec).sum()[sec]
	#print '=====Volume_last_5_day=====%s=============='%sec
	#print Volume_last_5_day
	if time_day is context.current_dt.minute:
		volume_last_5_min=history(5,'1m','volume',sec).sum()[sec]
		#print '=====Volume_last_5_min==================='
		#print volume_last_5_min
		if volume_last_5_min == 0 : return 1
		volume_rate_now=(float)(volume_last_5_min/5)/(Volume_last_5_day/(5*4*60))
		#print '=====Volume_rate_now==================='
		#print volume_rate_now
		return volume_rate_now
	Volume_last_1_day=history(1,'1d','volume',sec).sum()[sec]
	Volume_last_6_day=history(6,'1d','volume',sec).sum()[sec]
	#print '=====Volume_last_1_day==================='
	#print Volume_last_1_day
	#print '=====Volume_last_6_day==================='
	#print Volume_last_6_day

	if (Volume_last_6_day-Volume_last_1_day) == 0 : return 1
	Volume_rate_before= (float)(Volume_last_1_day/(4*60))\
				/((float)(Volume_last_6_day-Volume_last_1_day)/(5*4*60))
	#print '=====Volume_rate_before==================='
	#print Volume_rate_before
	return Volume_rate_before


#输入某只股票的hist数据,计算MA，返回一个数
def calMA(data_withHist,num):
    ma=data_withHist[-1*num:].mean()
    return ma


#输入某只股票的hist数据，然后判断是否发出买入卖出信号，输入为dataframe，输出为
#数字1:买入，-1卖出，0表示不变
def cal_signal(data_withHist):
    MA5=calMA(data_withHist,5)
    MA10=calMA(data_withHist,10)
    MA30=calMA(data_withHist,30)
    signal = 0
    if MA5>MA10 and MA10>MA30:
        signal=1
    else:
        signal=-1

    return signal



def before_trading_start(context):
# 选出流通市值小的N只股票
    length =100
    df = get_fundamentals(query(valuation.code, valuation.circulating_market_cap))
    df = df.dropna().sort(columns='circulating_market_cap',ascending=True)
    df = df.head(length)
    g.security = list(df['code'])

#    g.security=get_index_stocks('000001.XSHG')+get_index_stocks('399001.XSHE')

    date=context.current_dt.strftime("%Y-%m-%d")
    g.security= zhangting(g.security,date)

    #得到有历史数据的股票池，并且得到有历史数据
    current_data = get_current_data(g.security)
    g.operate_buy=[]
    g.stocks_hist=[]
    num=len(g.security)
    print num
    for i in range(0,num):
        #得到当前的不停牌股票
        if not current_data[g.security[i]].paused:
            hdict = attribute_history(g.security[i],34, '1d', ('close'),skip_paused=True,df = False)
            temp_hist= hdict['close']
            temp_hist = np.array([x for x in temp_hist if str(x) != 'nan'])


            #成多头形态.MACD出现红柱
            if len(temp_hist)>=30:
                g.stocks_hist.append(g.security[i])
                signal=cal_signal(temp_hist)
                if signal==1:
                    DIF, DEA,hist_macd = talib.MACD(temp_hist, fastperiod=12, slowperiod=26, signalperiod=9)
                    macd=(DIF[-1]-DEA[-1])*2
                    if macd>0 and macd<1:
                        g.operate_buy.append(g.security[i])


    g.operate_buy.append(g.s1)
    set_universe(g.operate_buy)
   # g.security.append(g.s1)
   # set_universe(g.security)


# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context, data):
    #止损
    for stock in list(context.portfolio.positions.keys()):
        his = history(2, '1d', 'close', [stock])
        if ((1-(his[stock][-1]/his[stock][0]))>=0.05):
            if not data[stock].paused == True:
                order_target(stock, 0)
                log.info(str(context.current_dt) + " 止损selling %s" % (stock))
                if stock in g.operate_buy:
                    g.operate_buy.remove(stock)




    buylist = []

    for stock in g.security:
#    for stock in g.operate_buy:
        Volume_rate_now=Volume_rate(stock,context.current_dt.minute,context)
        Volume_rate_day_before=Volume_rate(stock,context.current_dt.day-1,context)
        #if Volume_rate_now>1.5 and Volume_rate_day_before<0.8:
        buylist.append(stock)

    vw_list_temp=buylist
    vw_list=[]
     #去除停盘股票
    for stock in vw_list_temp:
        if not data[stock].paused == True:
            vw_list.append(stock)
    # 去除ST，*ST
    date=context.current_dt.strftime("%Y-%m-%d")
    st=get_extras('is_st', vw_list_temp, start_date=date, end_date=date, df=True)
    st=st.loc[date]
    buylist=list(st[st==False].index)




    buylist = buylist[:g.valid_num_stocks]


    if should_clear(buylist,context,data):
        buylist = []

    have_set = set(context.portfolio.positions.keys())
    hold_set = set(buylist)

    to_buy = hold_set - have_set
    to_sell = have_set - hold_set
    print 'tobuy:%s'%to_buy
    print 'tosell:%s'%to_sell

    for stock in to_sell:
        if not low_enough(stock,data):
            order_target(stock, 0)
            print 'sell:%s'%stock

    if len(to_buy) == 0:
        return

    each = context.portfolio.cash/len(to_buy)
    # if each > context.portfolio.portfolio_value/(context.valid_num_stocks-2):
    #     each = context.portfolio.portfolio_value/(context.valid_num_stocks-2)

    for stock in to_buy:
        #if not high_enough(stock,data):
            order_value(stock, each)
            print 'buy:%s'%stock


def should_clear(stocks,context,bar_dict):
    print '止损判断'
    g.lastma = g.ma
    g.ma = sum(history(g.OBSERVATION,'1d','close')[g.s1][:])/g.OBSERVATION

    #if g.lastma == 0:
        #return True

    return g.ma < g.lastma

def high_enough(stock, bar_dict):
    price = history (2, '1d', 'close')[stock].ix[0]
    pricenow = bar_dict[stock].close
    pct_change = (pricenow - price) / price
    if math.isnan(price):
        return True
    return pct_change>0.09


def low_enough(stock, bar_dict):
    price = history (2, '1d', 'close')[stock].ix[0]
    pricenow = bar_dict[stock].close
    pct_change = (pricenow - price) / price
    if math.isnan(price):
        return True
    return pct_change<-0.09


def zhangting(s_list,date):
    print date
    list_ZT= {}
    for s in s_list:
        #df = get_price(s,start_date=date, fields=['open', 'close', 'high', 'low', 'high_limit', 'money', 'paused'])
        df=attribute_history(s, 1, unit='1d',fields=('close','paused','high_limit'), df=False)
        #获取涨停股票数目
        #df_zt=df[df['high']>=df['high_limit']]

        if df['paused'][-1] == False:
            if df['close'][-1]>=df['high_limit'][-1]:
                list_ZT[s] = df


    return list_ZT.keys()
