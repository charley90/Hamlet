'''
v6
更新说明：
在v5的基础上
更改每日运行时间为  14:30-14:40
更改条件：
（1）今日当前涨幅在-1%--+1%之间（原来为-1%--3%）
增加条件：
（2）13:45—14:30，15分钟K线中的MAVOL5、MAVOL10相差低于20%
（3）13:45—14:30平均成交量均低于MAVOL5

删除条件：15分钟平均成交量位达到14:25之前平均成交量的3倍以上


一、初步筛选

（1）剔除ST股票
（2）剔除当日停牌股票
（3）选择流通市值小于60亿,流通市值大于20亿
（4）选择上市天数大于10天的股票

二、14:23—14:40
（1）14:30—14:40涨幅在-1%--+1%之间
（2）13:45—14:30，15分钟K线中的MAVOL5、MAVOL10相差低于20%
（3）13:45—14:30平均成交量均低于MAVOL5
（4）15分钟K线图MA5、10、20、30,4条均线高度粘合，最高价与最低价之差低于0.2%
（5）13点45分—14点30分，最高价和最低价相差小于1%，并分别与该时间段15分钟K线图MA5、10、20,3条均线的最高价与最低价差值小于0.2%。
（6）13点45分—14点30分的平均价格高于该时间段15分钟K线图MA5、10、20，3条均线的平均价格


'''

# 导入聚宽函数库
import jqdata
import numpy
import talib as tl

# 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    # 定义一个全局变量, 保存要操作的股票
    # 000001(股票:平安银行)
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    #设置待检验股票
    #g.checkstock='600687.XSHG'

# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context, data):
    # 每日开盘清仓
    if context.current_dt.hour == 9 and context.current_dt.minute==30:
        for stock in context.portfolio.positions.keys():
            order_target(stock, 0)
            log.info("卖出 %s",stock)
        return

    if not (context.current_dt.hour == 14 and 30<context.current_dt.minute<41):
        return    ## 只在每天的14:30-14:40进行交易操作

    #获得初筛后的股票池
    primaries_stocks=get_buy_stocks(context)
    #print'primaries_stocks:%s'%primaries_stocks
    buy_stocks=[]

    for i in primaries_stocks:

        #筛选前一分钟股价上涨的
        a=history(2, '1m', 'close', i,  skip_paused=True)
        b=a[i][1]/a[i][0]
        if b<1:
            #stock_check(i)
            #print'stock_check不符合筛选前一分钟股价上涨的'
            continue

        #选择涨幅在-1%--+1%之间的股票
        increase_rate=get_increase_rate(i)
        if not 0.99<increase_rate<1.01:
            #print '剔除: %s  涨幅不符     今日涨幅   :%s'%(i,increase_rate)
            continue

        #剔除换手率低于3%的股票
    #    turnover_rate=get_turnover_rate_txyklrm(context.current_dt, i,5)
    #    #print ' %s    换手率   :%s'%(i,turnover_rate)
    #    if turnover_rate<=0.03:
    #        #print '剔除: %s  换手率低      换手率   :%s'%(i,turnover_rate)
    #        continue

        #选择量比1.2以上的
    #    VolRate=getVolRate(context.current_dt, i,5)
    #    if VolRate<1.2:
    #        #print '剔除: %s  量比太低     今日量比   :%s'%(i,VolRate)
    #        continue

        # 均线粘合判断
        ma_rate=get_ma4s_ratio(context,i,uint='15m',a=5,b=10,c=20,d=30)
        if ma_rate<0.996:
        #    print '剔除: %s  均线不粘合    '%i
            continue

        #15分钟平均成交量位达到14:25之前平均成交量的3倍以上
        vol_now=history(1, '1m', 'volume', i)[i][-1]
        vol_to1425=getVolRate_today(context,i)

        #筛选15分钟平均成交量位达到14:25之前平均成交量的3倍以上的
        #if vol_now/vol_to1425<=3:
        #    print '剔除: %s  未放3倍量                   '%i
        #    continue

        #
        p=is_pinghuan(context,i)
        if p==0:
            continue


        t8=get_res8(context,i)
        if t8==0:
            continue

        #剔除不符合以下两条条件的
        #（2）13:45—14:30，15分钟K线中的MAVOL5、MAVOL10相差低于20%;
        #（3）13:45—14:30平均成交量均低于MAVOL5
        t_vol=is_mavol_ok(context,i)
        if t_vol==0:
            continue

        buy_stocks.append(i)
    print'buy_stocks:%s'%buy_stocks
    print'num of buy_stocks:' ,len(buy_stocks)

    #剔除已持仓股票
    holdstocks=context.portfolio.positions.keys()
    #print '*******',holdstocks
    buy_stocks=[i for i in buy_stocks if i not in holdstocks]

    buy_num=len(buy_stocks)
    if buy_num==0:
        return
    value = context.portfolio.total_value
    order_money=value/buy_num

    for i in buy_stocks:
        order(i, 100)



 #换手率计算-txy-克鲁软猫
def get_turnover_rate_txyklrm(current_dt, stock, backDays) :

    cdt = current_dt

    tStart = timedelta(hours=9, minutes=30)
    tNow = timedelta(hours=cdt.hour, minutes=cdt.minute)

    duration = tNow - tStart
    durationMinutes = duration.seconds / 60

    if cdt.hour > 11 :
        durationMinutes -= 90

    avgVolToday = 0
    if durationMinutes > 0 :
        hData = attribute_history(stock, durationMinutes, unit='1m'
                        , fields=('volume')
                        , skip_paused=True
                        , df=False)
        volumeToday = np.array(hData['volume'], dtype='f8')
        avgVolToday = sum(volumeToday) / durationMinutes

    #查询流通股数
    df=get_fundamentals(query(
        valuation.code,valuation.circulating_cap
        ).filter(valuation.code==stock))

    circulating_cap=df['circulating_cap'][0]*10000
    #print'流通股数： %s'%circulating_cap

    turnover_rate=sum(volumeToday)/circulating_cap
    #print'sum(volumeToday):  %s  circulating_cap:%s  turnover_rate:%s'%(sum(volumeToday), circulating_cap,turnover_rate)

    return turnover_rate


# 量比计算--克鲁软猫
def getVolRate(current_dt, stock, backDays) :

    cdt = current_dt

    tStart = timedelta(hours=9, minutes=30)
    tNow = timedelta(hours=cdt.hour, minutes=cdt.minute)

    duration = tNow - tStart
    durationMinutes = duration.seconds / 60

    if cdt.hour > 11 :
        durationMinutes -= 90

    avgVolToday = 0
    if durationMinutes > 0 :
        hData = attribute_history(stock, durationMinutes, unit='1m'
                        , fields=('volume')
                        , skip_paused=True
                        , df=False)
        volumeToday = np.array(hData['volume'], dtype='f8')
        avgVolToday = sum(volumeToday) / durationMinutes

    hData = attribute_history(stock, backDays, unit='1d'
                    , fields=('volume')
                    , skip_paused=True
                    , df=False)
    volumeHistory = np.array(hData['volume'], dtype='f8')
    avgVolHistory = sum(volumeHistory) / (backDays * 4 * 60)


    volRate = avgVolToday / avgVolHistory

    return volRate

#今日涨幅计算
def get_increase_rate(stock):
    data=get_current_data()
    cy=history(1, '1d', 'close',security_list=stock)
    close_yesterday=cy[stock][-1]

    cp = history(1, '1m', 'close',security_list=stock)     # 取得上一时间点价格（收盘价）
    current_price=cp[stock][-1]
    increase_rate=current_price/close_yesterday
    #print'价格： %s  涨幅： %s'%(current_price,increase_rate)
    return increase_rate


#过滤st 停牌股票
def filter_pause_st(stock_list):
    data_current=get_current_data()
    return[stock for stock in stock_list if not(data_current[stock].paused or data_current[stock].is_st)]

#获取初始查询股票池  选择流通市值小于60亿,流通市值大于20亿，过滤掉上市时间小于10天的
def get_buy_stocks(context):
    q = query(valuation.code,valuation.circulating_market_cap).filter(
        valuation.circulating_market_cap.between(0,100)
    ).order_by(
        valuation.circulating_market_cap.asc()
    )

    stocks = get_fundamentals(q)

    stock_list=list(stocks['code'])
    buy_stocks=[]
    #选择上市天数大于10天的股票
    for i in stock_list:
        if get_listed_days(context,i)>10:
            buy_stocks.append(i)

    #踢出st和停牌股
    final_stocks= filter_pause_st(buy_stocks)

    return final_stocks

# 计算今日(到14:25)每分钟成交量均量
def getVolRate_today(context, stock,end_time=' 14:25:00') :

    today_str=str(context.current_dt)[:10]
    start_str=str(context.current_dt)[:10]+' 09:30:00'
    start_datetime = datetime.datetime.strptime(start_str,'%Y-%m-%d %H:%M:%S')
    end_str=today_str+end_time
    end_datetime = datetime.datetime.strptime(end_str,'%Y-%m-%d %H:%M:%S')

    durationMinutes=(end_datetime-start_datetime).seconds/60


    if end_datetime.hour > 11 :
        durationMinutes -= 90
    #print '今日成交量计算分钟数： %s'%durationMinutes

    avgVolToday = 0
    if durationMinutes > 0 :
        hData = get_price(stock,start_date=start_str, end_date=end_str, frequency='1m',
                            fields=['volume'], skip_paused=True )
        volumeToday = np.array(hData['volume'], dtype='f8')
        avgVolToday = sum(volumeToday) / durationMinutes

    return avgVolToday
#获得与实际运行时一直的ma数
def get_ma_realtime(stock,nowtime,num,freq):
    int_freq=int(freq[:-1])
    min_now=nowtime.minute
    minute_remainder=min_now%int_freq
    end_time=(nowtime-datetime.timedelta(minutes= minute_remainder+1)).strftime("%Y-%m-%d %H:%M:%S")
    end=str(end_time)
    if minute_remainder==0:
        price=get_price(stock, end_date=end, frequency=freq, fields=['close'], skip_paused=True,count=num)
        price_list=list(price['close'])
    else:
        price=get_price(stock, end_date=end, frequency=freq, fields=['close'], skip_paused=True,count=num-1)
        price_now=history(1, '1m', 'close', security_list=stock)
        #print '###price_now  ',price_now,'###################'
        price_list=list(price['close'])
        price_list.append(price_now[stock])

    ma=sum(price_list)/num

    return ma

def get_ma4s_ratio(context,stock,uint='15m',a=5,b=10,c=20,d=30):
    nowtime=context.current_dt
    ma1=get_ma_realtime(stock,nowtime,a,'15m')
    ma2=get_ma_realtime(stock,nowtime,b,'15m')
    ma3=get_ma_realtime(stock,nowtime,c,'15m')
    ma4=get_ma_realtime(stock,nowtime,d,'15m')

    #print'%s  ma5:%s  ma10:%s  ma20:%s  ma30:%s'%(stock,ma1,ma2,ma3,ma4)
    ma_max=max(ma1,ma2,ma3,ma4)
    ma_min=min(ma1,ma2,ma3,ma4)
    ma4s_ratio=ma_min/ma_max

    return ma4s_ratio

def get_listed_days(context,stock):
    today_str=context.current_dt.strftime("%Y-%m-%d")
    today= datetime.datetime.strptime(today_str,'%Y-%m-%d')
    start_date_str=str(get_security_info(stock).start_date)
    start= datetime.datetime.strptime(start_date_str,'%Y-%m-%d')

    listed_days=(today-start).days
    #print'上市时间：',start,'上市天数：',listed_days

    return listed_days

#  条件7判断，不符合以下条件的返回0，符合返回1
# 13点45分—14点30分，最高价和最低价相差小于1%，并分别与该时间段15分钟K线图MA5、10、20、30,
#   4条均线的最高价与最低价差值小于0.2%
def is_pinghuan(context,stock):
    today_str=str(context.current_dt)[:10]
    start_str=str(context.current_dt)[:10]+' 13:45:00'
    #print 'context.current_dt.minute',context.current_dt.minute
    if context.current_dt.minute>30:
        end_str=today_str+' 14:29:00'
    else:
        end_str=str(context.current_dt)

    df= get_price(stock, start_str, end_str, frequency='15m',fields=['high','low'], skip_paused=True)
    rp=(max(df['high'])-min(df['low']))/ max(df['high'])
    if rp>0.01:
        return 0
    nowtime=context.current_dt
    freq='15m'
    ma5 = get_ma_realtime(stock,nowtime, 5,freq)
    ma10= get_ma_realtime(stock,nowtime, 10,freq)
    ma20= get_ma_realtime(stock,nowtime, 20,freq)
    ma30= get_ma_realtime(stock,nowtime, 30,freq)
    Rma=(max(ma5,ma10,ma20,ma30)-min(ma5,ma10,ma20,ma30))/ min(ma5,ma10,ma20,ma30)

    if abs(max(df['high'])- max(ma5,ma10,ma20,ma30))/ max(ma5,ma10,ma20,ma30)>0.002:
        return 0
    if abs(min(ma5,ma10,ma20,ma30)- min(df['low']))/ min(ma5,ma10,ma20,ma30)>0.002:
        return 0
    return 1

    #以下条件判断，符合的返回1，否则返回0
    #13点45分—14点30分的平均价格高于该时间段15分钟K线图MA5、10、20，3条均线的平均价格
def get_res8(context,stock):
    today_str=str(context.current_dt)[:10]
    start_str=str(context.current_dt)[:10]+' 13:45:00'
    end_str=today_str+' 14:29:00'
    h=get_price(stock, start_date=start_str, end_date=end_str, frequency='15m', fields=['high','low'], skip_paused=True)
    h_average=sum(h['high'])/3

    pma=get_price(stock,  end_date=end_str, frequency='15m', fields=['high','low','close'], skip_paused=True,count=35)

    #ma5，ma10,ma20三根均线计算
    hma51=sum(pma['close'][-8:-3])/5
    hma52=sum(pma['close'][-7:-2])/5
    hma53=sum(pma['close'][-6:-1])/5
    hma5_average=(hma51+hma52+hma53)/3
    hma101=sum(pma['close'][-13:-3])/10
    hma102=sum(pma['close'][-12:-2])/10
    hma103=sum(pma['close'][-11:-1])/10
    hma10_average=(hma101+hma102+hma103)/3
    hma201=sum(pma['close'][-23:-3])/20
    hma202=sum(pma['close'][-22:-2])/20
    hma203=sum(pma['close'][-21:-1])/20
    hma20_average=(hma201+hma202+hma203)/3

    if h_average<hma5_average:
        return 0
    if h_average<hma10_average:
        return 0
    if h_average<hma20_average:
        return 0

    return 1

# 成交量判断   返回0 表示不符合以下2条条件，返回1表示符合
#（2）13:45—14:30，15分钟K线中的MAVOL5、MAVOL10相差低于20%
#（3）13:45—14:30平均成交量均低于MAVOL5
def is_mavol_ok(context,stock):
    today_str=str(context.current_dt)[:10]
    start_str=today_str+' 13:45:00'
    end_str =today_str+' 14:29:00'

    vol=get_price(stock,  end_date=end_str, frequency='15m', fields=['volume'], skip_paused=True,count=15)
    vol_average=sum(vol['volume'][-3:])/3

    vol_arr=numpy.array(vol['volume'])
    vol_ma5=tl.MA(vol_arr,timeperiod=5, matype=0)  #需要 import talib as tl
    vol_ma10=tl.MA(vol_arr,timeperiod=10, matype=0)
    #print 'vol_ma5,vol_ma10',vol_ma5,vol_ma10

    #选择13:45—14:30平均成交量均低于MAVOL5
    if vol_average>max(vol_ma5[-3:]):
        return 0

    #选择 13:45—14:30，15分钟K线中的MAVOL5、MAVOL10相差低于20%
    for i in range(3):
        devirate=get_devirate(vol_ma5[-i-1],vol_ma10[-i-1])
        if devirate>0.2:
            return 0
    return 1

    #求（较大值-较小值）/较小值 注意：两个值均为正数
def get_devirate(a,b):
    c=abs(a-b)
    d=c/min(a,b)
    return d

def stock_check(stock):
    try :
        if stock==g.checkstock:
            print '===========get the checkstock==========='
    except:
        return
