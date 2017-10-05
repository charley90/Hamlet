import pandas as pd
import numpy as np
import jqdata

def initialize(context):

   # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')
    #设置参数
    set_params()
    #设置中间变量
    set_backtest(context)
    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      # 开盘前运行
    #run_daily(before_market_open, time='before_open', reference_security='000300.XSHG')
      # 开盘时运行
    run_daily(market_open, time='open', reference_security='000300.XSHG')
      # 收盘后运行
    #run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')

#设置回测环境条件，手续费，复权规则等
def set_backtest(context):
    #基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 过滤掉order系列API产生的比error级别低的log
    #log.set_level('order', 'error')
    # 设置滑点
    set_slippage(PriceRelatedSlippage(0.00246))
    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')

def set_params():
    g.stock_list = []
    g.set_stock = set()
## 开盘时运行函数
## 开盘函数
def market_open(context):
    pools = []
    #取昨日PB小于1，大于0的股票
    all_data = get_current_data()
    all_stock_list = get_all_securities(date = context.current_dt).index
    stock_list = [stock for stock in all_stock_list if all_data[stock].paused != True]
    q = query(valuation.code,valuation.pb_ratio,valuation.circulating_market_cap).filter(valuation.code.in_(stock_list),\
                                                       valuation.pb_ratio >0,\
                                                       valuation.pb_ratio < 1)
    df = get_fundamentals(q)
    df = df.sort('pb_ratio',ascending = 1)
    #print(df)
    print('有%s 个股票满足条件' % len(df['code'].values))
    changed = set(df['code'].values) - g.set_stock
    changed_old = g.set_stock - set(df['code'].values)
    g.set_stock = set(df['code'].values)
    log.info('新满足条件的股票%s' % changed)
    log.info('淘汰的股票%s' % changed_old)


    for stock in context.portfolio.positions.keys():
        if stock not in g.set_stock:
            order_target(stock,0)

    hold_num = len(context.portfolio.positions.keys())
    log.info('hold_num',hold_num)
    for stock in context.portfolio.positions.keys():
        pools.append(stock)
    if hold_num <5:
        pools_1 = list(g.set_stock)[:5-hold_num]
        for stock in pools_1:
            if stock not in context.portfolio.positions:
                pools.append(stock)
    if pools == g.stock_list:
        pass
    else:
        log.info('pools',pools)
        change_positions(pools,context)
    g.stock_list = pools

def get_ES_ratio(context,stock_list):
        #获取股票历史数据
        def get_history_price_es(stock_list):
            hprice = history(180, '1d', 'close', stock_list, df=True)
            dReturns = hprice.resample('D',how='last').pct_change().fillna(value=0, method=None, axis=0)
            stock_ES ={}
            total_es_ratio = 0
            for stock in stock_list:
                Returns_sort =  sorted(dReturns[stock])
                count = 0
                sum_value = 0
                for i in range(len(dReturns)):
                    if i < 1.8:
                        sum_value += Returns_sort[i]
                        count += 1
                if count == 0:
                    ES = 0
                else:
                    ES = -(sum_value / 2)
                    total_es_ratio += 1.0/ES
                stock_ES[stock]=1.0/ES
            return stock_ES,total_es_ratio
        stock_ES,total = get_history_price_es(stock_list)
        stock_es_ratio = {}
        for stock in stock_ES.keys():
            tmp_es_ratio = stock_ES[stock]/total
            stock_es_ratio[stock] = round(tmp_es_ratio,4)
        return stock_es_ratio

def get_portfolio_ratio(context,ratio):
    def get_portfolio_return(ratio):
        #portfolio_se = pd.Series([0]*180,index = se.index)
        es_np = []
        ratio_np =[]
        for stock in ratio.keys():
            #hStocks = history(180, '1d', 'close', stock, df=True)
            hStocks = attribute_history(stock, count=180, unit='1d', fields=['close'])['close']
            #做成日回报数据
            dReturns = hStocks.pct_change().fillna(value=0, method=None, axis=0).values
            #print('type',type(dReturns))
            es_np.append(dReturns)
            ratio_np.append(ratio[stock])
        #print('shape',np.shape(es_np),np.shape(ratio_np))
        portfolio_return = np.dot(np.array(es_np).T,ratio_np)

        #portfolio_return = sorted(portfolio_return)
        return portfolio_return
    risk_money = (context.portfolio.total_value /5)* 0.03*len(g.stock_list)
    print('risk_money',risk_money)
    maxrisk_money = risk_money*1.7
    if len(ratio)>0:
        p = get_portfolio_return(ratio)
    else:
        return 0
    #print('p',type(p),np.shape(p))
    portfolio_var = 1.96*np.std(p)
    portfolio_value_var = risk_money/portfolio_var
    '''
    #print('type',type(p),np.shape(p),p[:3])
    if p.any()==0.0:# and p.all()==None:
        return 0
    else:
        '''
    p = list(p)
    p.sort()
    #print('p',type(p),p)
    portfolio_es = -sum(p[:9])/9
    portfolio_value_es = maxrisk_money/portfolio_es
    if portfolio_value_es < 0:
        portfolio_value_es = context.portfolio.total_value
    tmp_max_value = min(portfolio_value_var,portfolio_value_es,context.portfolio.total_value)
    portfolio_max_value = round(tmp_max_value,2)
    #portfolio_max_value = g.portfolio_max_value
    #g.portfolio_max_value = portfolio_max_value
    print('portfolio_max_value',portfolio_max_value,'var',portfolio_value_var,'es',portfolio_value_es)
    return portfolio_max_value

def change_positions(signal,context):
    all_stock_list = signal
    ratio = get_ES_ratio(context,all_stock_list)
    #print('ratio',ratio)

    #a = len(hold_list)
    #获取总仓位风险配比
    value = get_portfolio_ratio(context,ratio)
    #print('portfolio_trade_value',value)

    price_now = history(1,'1d','close',all_stock_list)
    #print(type(price_now))
    for stock in all_stock_list:
        stock_value_should=ratio[stock]*value
        if stock not in context.portfolio.positions:
            order_target_value(stock,stock_value_should)
        else:
            stock_value_now = context.portfolio.positions[stock].total_amount*price_now[stock].values
            cash = context.portfolio.available_cash
            if (abs(stock_value_should-stock_value_now)/stock_value_should) > 0.25 and cash >= 0.25*stock_value_should:
                order_target_value(stock,stock_value_should)
