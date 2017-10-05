'''
策略思路：
'''

'''
================================
初始化部分
'''
#导入库
import datetime as dt
from jqdata import gta
import pandas as pd
import numpy as np

#初始化
def initialize(context):
    #设置参数
    set_params(context)
    #设置中间变量
    set_variables()
    #设置回测环境条件
    set_backtest(context)
    #定时函数
    run_daily(open_func,time = 'open')

def set_params(context):

    g.days = 0
    g.zhouqi = 30
    g.peg = 0.6
    g.stock_list = []
    g.universe = []

def set_variables():
    g.portfolio_max_value=0

def set_backtest(context):
    set_slippage(FixedSlippage(0))
    set_order_cost(OrderCost(0,0,0,0), 'fund')
    set_order_cost(OrderCost(0,0,0,0), 'stock')
    #用沪深 300 做回报基准
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    # 关闭部分log
    log.set_level('order', 'error')

'''
================================
函数部分
'''
def open_func(context):
    #是否交易
    g.iftrade = iftrade(context)
    # 配仓，分配持股比例
    stock_ratio = {}
    if g.iftrade:
        #获取股票列表
        get_universe(context)
        stock_list = get_stock_list(context)
        g.stock_list = stock_list
        #print(g.stock_list)
        #获取股票风险平价配比
        ES_ratio = get_ES_ratio(context,stock_list)
        print('ES_ratio',ES_ratio)
        #获取总仓位风险配比
        portfolio_trade_value = get_portfolio_ratio(context,ES_ratio)
        print('portfolio_trade_value',portfolio_trade_value)
        #进行交易
        goto_trade(context,ES_ratio,portfolio_trade_value)

    #===============

'''
====================================
自定义函数部分
'''

#计算调仓周期信号函数
def iftrade(context):
    if len(g.stock_list) == 0:
        g.days = g.zhouqi
        return True

    if g.days == 0:
        g.days = g.zhouqi
        return True
    else:
        g.days -= 1
        return False

def get_universe(context):
    #获取非停牌股票
    def unpaused(stocklist):
        current_data = get_current_data()
        return [s for s in stocklist if not current_data[s].paused]
    #选择行业股
    def industry_stock(stock_list):
        cycle_industry = ['B06', # 煤炭开采和洗选业 	1994-01-06
                          'B07', # 石油和天然气开采业 	1996-06-28
                          'B08', # 黑色金属矿采选业 	1997-07-08
                          'B09', # 有色金属矿采选业 	1996-03-20
                          'B11', # 开采辅助活动 	2002-02-05
                          'C25', # 石油加工、炼焦及核燃料加工业 	1993-10-25
                          'C26', # 化学原料及化学制品制造业 	1990-12-19
                          'C28', # 化学纤维制造业 	1993-07-28
                          'C29', # 橡胶和塑料制品业 	1992-08-28
                          'C30', # 非金属矿物制品业 	1992-02-28
                          'C31', # 黑色金属冶炼及压延加工业 	1994-01-06
                          'C32', # 有色金属冶炼和压延加工业 	1996-02-15
                          'C33', # 金属制品业 	1993-11-30
                          'C34', # 通用设备制造业 	1992-03-27
                          'C35', # 专用设备制造业 	1992-07-01
                          'C36', # 汽车制造业 	1992-07-24
                          'C37', # 铁路、船舶、航空航天和其它运输设备制造业 	1992-03-31
                          'C38', # 电气机械及器材制造业 	1990-12-19
                          'C41', # 其他制造业 	1992-08-14
                          'D44', # 电力、热力生产和供应业 	1993-04-16
                          'E47', # 房屋建筑业 	1993-04-29
                          'E48', # 土木工程建筑业 	1994-01-28
                          'E50', # 建筑装饰和其他建筑业 	1997-05-22
                          'G53', # 铁路运输业 	1998-05-11
                          'G54', # 道路运输业 	1991-01-14
                          'G55', # 水上运输业 	1993-11-19
                          'G56', # 航空运输业 	1997-11-05
                          'G58', # 装卸搬运和运输代理业 	1993-05-05
                          'J66', # 货币金融服务 	1991-04-03
                          'J67', # 资本市场服务 	1994-01-10
                          'J68', # 保险业 	2007-01-09
                          'J69', # 其他金融业 	2012-10-26
                          'K70', # 房地产业 	1992-01-13
                          'M74', # 专业技术服务业 	2007-02-15
                          ]
        for industry in cycle_industry:
            stocks = get_industry_stocks(industry)
            stock_list = list(set(stock_list).difference(set(stocks)))
        return stock_list
    # 剔除上市时间较短的产品

    def delnewstock_func(context,buylist,deltaday):
        deltaDate = context.current_dt.date() - dt.timedelta(deltaday)
        tmpList = []
        for stock in buylist:
            if get_security_info(stock).start_date < deltaDate:
                tmpList.append(stock)
        return tmpList

    #获取PEG符合条件的股票
    def get_peg_stock(context,stock_list):
        #获取数据
        def get_data(pool, periods):
            q = query(valuation.code, income.statDate, income.pubDate).filter(valuation.code.in_(pool))
            df = get_fundamentals(q)
            df.index = df.code
            stat_dates = set(df.statDate)
            stat_date_stocks = { sd:[stock for stock in df.index if df['statDate'][stock]==sd] for sd in stat_dates }

            def quarter_push(quarter):
                if quarter[-1]!='1':
                    return quarter[:-1]+str(int(quarter[-1])-1)
                else:
                    return str(int(quarter[:4])-1)+'q4'

            q = query(valuation.code,
                      indicator.inc_net_profit_year_on_year,
                      valuation.market_cap,
                      valuation.pe_ratio
                      )

            stat_date_panels = { sd:None for sd in stat_dates }

            for sd in stat_dates:
                quarters = [sd[:4]+'q'+str(int(sd[5:7])/3)]
                for i in range(periods-1):
                    quarters.append(quarter_push(quarters[-1]))
                nq = q.filter(valuation.code.in_(stat_date_stocks[sd]))
                pre_panel = { quarter:get_fundamentals(nq, statDate = quarter) for quarter in quarters }
                for thing in pre_panel.values():
                    thing.index = thing.code.values
                panel = pd.Panel(pre_panel)
                panel.items = range(len(quarters))
                stat_date_panels[sd] = panel.transpose(2,0,1)

            final = pd.concat(stat_date_panels.values(), axis=2)

            return final#.dropna(axis=2)
        #返回eps的值和pe的值
        def get_stock_info(context,stock_list):
            k = get_data(stock_list,5)
            df_eps = k.inc_net_profit_year_on_year
            df_pe = k.pe_ratio
            df_mktcap=k.market_cap
            se_pe=df_pe[-1:]
            se_eps_0=df_eps[-1:]
            se_eps_1=df_eps[-2:-1]
            se_eps_2=df_eps[-3:-2]
            se_eps_3=df_eps[-4:-3]
            se_eps_std=df_eps[-4:].std()
            se_eps_mean=df_eps[-4:].mean()
            se_mktcap = df_mktcap[-1:]
            #计算eps的标准差，最后一个数据，和市盈率
            return se_pe,se_eps_0,se_eps_1,se_eps_2,se_eps_3,se_mktcap

        se_pe,se_eps_0,se_eps_1,se_eps_2,se_eps_3,mktcap = get_stock_info(context, stock_list)
        #print(type(se_pe),type(se_eps_mean))
        #print('pe',se_pe,'eps',se_eps,'se_eps_mean')#,se_eps_mean,'se_eps_std',se_eps_std)
        #se_interest = get_stock_interest(context,stock_list)
        #获取二次中信号的已持仓股票
        #print('=====================inc',len(se_eps.index),se_eps)#,se_eps_1,se_eps_2)
        pre_stock_list = []
        for stock in context.portfolio.positions:
            if stock in stock_list:
                pre_stock_list.append(stock)
        #获取PEG的值
        #peg = se_pe/se_eps


        #print('df_interest',df_interest,'len',df_interest.shape)
        df_eps = pd.concat([se_eps_0,se_eps_1,se_eps_2,se_eps_3])#,join='inner')
        df_eps.index = ['eps_0','eps_1','eps_2','eps_3']
        df_eps = df_eps.where(df_eps<100,100)
        df_eps = df_eps.where(df_eps>-100,-100)
        df_1 = df_eps.T
        df_1['mean'] = df_1.mean(axis = 1)
        df_1['std'] = df_1.std(axis = 1)
        #print('==========================eps',df_1.T.shape,df_1.T)
        df_fin = pd.concat([df_1,se_pe.T,mktcap.T],axis = 1)
        #df_fin = pd.concat([df_1,df_pec],axis = 1)
        #print('========df_fin',df_fin.shape,df_fin)
        df_fin.columns = ['eps_0','eps_1','eps_2','eps_3','mean','std','pe','mktcap']
        #G = df_fin['eps_0']+100*df_fin['divpercent']
        G = df_fin['eps_0']#+100*df_fin['divpercent']
        G = G.where(G!=0,0.0001)
        df_fin['PEG'] = (df_fin['pe']/G).fillna(-1)
        #print('=========================peg',df_fin.shape,df_fin)
        #df_fin = df_fin.dropna(axis = 0)
        #df.index = ['pe','eps','std','peg']
        #df = pd.DataFrame([se_pe,se_eps,se_eps_mean,se_eps_std,peg],index=['pe','eps','mean','std','peg'])
        return df_fin

    if g.universe==context.universe:
        today = context.current_dt
        stock_list = list(get_all_securities(['stock'], today).index)

        stock_list = unpaused(stock_list)
        stock_list = industry_stock(stock_list)

        stock_list = delnewstock_func(context,stock_list,181)

        #print('=====================1',len(stock_list))
        df_fin = get_peg_stock(context,stock_list)
        df_fin_peg = df_fin[(df_fin['pe']>0) &(df_fin['eps_0']<=50) &(df_fin['eps_0']>0)&(df_fin['PEG']>0) &(df_fin['PEG']<g.peg) & \
        (df_fin['eps_0']>df_fin['std'])]#& (df_fin['eps_1']>0) &(df_fin['eps_2']>0) ]
        #df_fin_peg = df_fin_peg.sort('mktcap',ascending = 1)
        df_fin_peg = df_fin_peg.sort('mktcap',ascending = 1)
        fin_stock_list = list(df_fin_peg.index)[:10]
        g.universe=fin_stock_list
        set_universe(fin_stock_list)


def get_stock_list(context):
    #获取非停牌股票
    def unpaused(stocklist):
        current_data = get_current_data()
        return [s for s in stocklist if not current_data[s].paused]
    #选择行业股
    def industry_stock(stock_list):
        cycle_industry = ['B06', # 煤炭开采和洗选业 	1994-01-06
                          'B07', # 石油和天然气开采业 	1996-06-28
                          'B08', # 黑色金属矿采选业 	1997-07-08
                          'B09', # 有色金属矿采选业 	1996-03-20
                          'B11', # 开采辅助活动 	2002-02-05
                          'C25', # 石油加工、炼焦及核燃料加工业 	1993-10-25
                          'C26', # 化学原料及化学制品制造业 	1990-12-19
                          'C28', # 化学纤维制造业 	1993-07-28
                          'C29', # 橡胶和塑料制品业 	1992-08-28
                          'C30', # 非金属矿物制品业 	1992-02-28
                          'C31', # 黑色金属冶炼及压延加工业 	1994-01-06
                          'C32', # 有色金属冶炼和压延加工业 	1996-02-15
                          'C33', # 金属制品业 	1993-11-30
                          'C34', # 通用设备制造业 	1992-03-27
                          'C35', # 专用设备制造业 	1992-07-01
                          'C36', # 汽车制造业 	1992-07-24
                          'C37', # 铁路、船舶、航空航天和其它运输设备制造业 	1992-03-31
                          'C38', # 电气机械及器材制造业 	1990-12-19
                          'C41', # 其他制造业 	1992-08-14
                          'D44', # 电力、热力生产和供应业 	1993-04-16
                          'E47', # 房屋建筑业 	1993-04-29
                          'E48', # 土木工程建筑业 	1994-01-28
                          'E50', # 建筑装饰和其他建筑业 	1997-05-22
                          'G53', # 铁路运输业 	1998-05-11
                          'G54', # 道路运输业 	1991-01-14
                          'G55', # 水上运输业 	1993-11-19
                          'G56', # 航空运输业 	1997-11-05
                          'G58', # 装卸搬运和运输代理业 	1993-05-05
                          'J66', # 货币金融服务 	1991-04-03
                          'J67', # 资本市场服务 	1994-01-10
                          'J68', # 保险业 	2007-01-09
                          'J69', # 其他金融业 	2012-10-26
                          'K70', # 房地产业 	1992-01-13
                          'M74', # 专业技术服务业 	2007-02-15
                          ]
        for industry in cycle_industry:
            stocks = get_industry_stocks(industry)
            stock_list = list(set(stock_list).difference(set(stocks)))
        return stock_list
    # 剔除上市时间较短的产品

    def delnewstock_func(context,buylist,deltaday):
        deltaDate = context.current_dt.date() - dt.timedelta(deltaday)
        tmpList = []
        for stock in buylist:
            if get_security_info(stock).start_date < deltaDate:
                tmpList.append(stock)
        return tmpList

    #获取PEG符合条件的股票
    def get_peg_stock(context,stock_list):
        #获取数据
        def get_data(pool, periods):
            q = query(valuation.code, income.statDate, income.pubDate).filter(valuation.code.in_(pool))
            df = get_fundamentals(q)
            df.index = df.code
            stat_dates = set(df.statDate)
            stat_date_stocks = { sd:[stock for stock in df.index if df['statDate'][stock]==sd] for sd in stat_dates }

            def quarter_push(quarter):
                if quarter[-1]!='1':
                    return quarter[:-1]+str(int(quarter[-1])-1)
                else:
                    return str(int(quarter[:4])-1)+'q4'

            q = query(valuation.code,
                      indicator.inc_net_profit_year_on_year,
                      valuation.market_cap,
                      valuation.pe_ratio
                      )

            stat_date_panels = { sd:None for sd in stat_dates }

            for sd in stat_dates:
                quarters = [sd[:4]+'q'+str(int(sd[5:7])/3)]
                for i in range(periods-1):
                    quarters.append(quarter_push(quarters[-1]))
                nq = q.filter(valuation.code.in_(stat_date_stocks[sd]))
                pre_panel = { quarter:get_fundamentals(nq, statDate = quarter) for quarter in quarters }
                for thing in pre_panel.values():
                    thing.index = thing.code.values
                panel = pd.Panel(pre_panel)
                panel.items = range(len(quarters))
                stat_date_panels[sd] = panel.transpose(2,0,1)

            final = pd.concat(stat_date_panels.values(), axis=2)

            return final#.dropna(axis=2)
        #返回eps的值和pe的值
        def get_stock_info(context,stock_list):
            k = get_data(stock_list,5)
            df_eps = k.inc_net_profit_year_on_year
            df_pe = k.pe_ratio
            df_mktcap=k.market_cap
            se_pe=df_pe[-1:]
            se_eps_0=df_eps[-1:]
            se_eps_1=df_eps[-2:-1]
            se_eps_2=df_eps[-3:-2]
            se_eps_3=df_eps[-4:-3]
            se_eps_std=df_eps[-4:].std()
            se_eps_mean=df_eps[-4:].mean()
            se_mktcap = df_mktcap[-1:]
            #计算eps的标准差，最后一个数据，和市盈率
            return se_pe,se_eps_0,se_eps_1,se_eps_2,se_eps_3,se_mktcap

        se_pe,se_eps_0,se_eps_1,se_eps_2,se_eps_3,mktcap = get_stock_info(context, stock_list)
        #print(type(se_pe),type(se_eps_mean))
        #print('pe',se_pe,'eps',se_eps,'se_eps_mean')#,se_eps_mean,'se_eps_std',se_eps_std)
        #se_interest = get_stock_interest(context,stock_list)
        #获取二次中信号的已持仓股票
        #print('=====================inc',len(se_eps.index),se_eps)#,se_eps_1,se_eps_2)
        pre_stock_list = []
        for stock in context.portfolio.positions:
            if stock in stock_list:
                pre_stock_list.append(stock)
        #获取PEG的值
        #peg = se_pe/se_eps


        #print('df_interest',df_interest,'len',df_interest.shape)
        df_eps = pd.concat([se_eps_0,se_eps_1,se_eps_2,se_eps_3])#,join='inner')
        df_eps.index = ['eps_0','eps_1','eps_2','eps_3']
        df_eps = df_eps.where(df_eps<100,100)
        df_eps = df_eps.where(df_eps>-100,-100)
        df_1 = df_eps.T
        df_1['mean'] = df_1.mean(axis = 1)
        df_1['std'] = df_1.std(axis = 1)
        #print('==========================eps',df_1.T.shape,df_1.T)
        df_fin = pd.concat([df_1,se_pe.T,mktcap.T],axis = 1)
        #df_fin = pd.concat([df_1,df_pec],axis = 1)
        #print('========df_fin',df_fin.shape,df_fin)
        df_fin.columns = ['eps_0','eps_1','eps_2','eps_3','mean','std','pe','mktcap']
        #G = df_fin['eps_0']+100*df_fin['divpercent']
        G = df_fin['eps_0']#+100*df_fin['divpercent']
        G = G.where(G!=0,0.0001)
        df_fin['PEG'] = (df_fin['pe']/G).fillna(-1)
        #print('=========================peg',df_fin.shape,df_fin)
        #df_fin = df_fin.dropna(axis = 0)
        #df.index = ['pe','eps','std','peg']
        #df = pd.DataFrame([se_pe,se_eps,se_eps_mean,se_eps_std,peg],index=['pe','eps','mean','std','peg'])
        return df_fin
    today = context.current_dt
    stock_list = context.universe

    stock_list = unpaused(stock_list)
    stock_list = industry_stock(stock_list)

    stock_list = delnewstock_func(context,stock_list,181)

    #print('=====================1',len(stock_list))
    try:
        df_fin = get_peg_stock(context,stock_list)
        df_fin_peg = df_fin[(df_fin['pe']>0) &(df_fin['eps_0']<=50) &(df_fin['eps_0']>0)&(df_fin['PEG']>0) &(df_fin['PEG']<g.peg) & \
        (df_fin['eps_0']>df_fin['std'])]#& (df_fin['eps_1']>0) &(df_fin['eps_2']>0) ]
        #df_fin_peg = df_fin_peg.sort('mktcap',ascending = 1)
        df_fin_peg = df_fin_peg.sort('mktcap',ascending = 1)
        fin_stock_list = list(df_fin_peg.index)[:5]
        print('new_stock_list',len(fin_stock_list),fin_stock_list)

        old_stock_list = []
        for stock in context.portfolio.positions:
            if stock in stock_list:
                if stock not in fin_stock_list:
                    old_stock_list.append(stock)

        if len(fin_stock_list)<5 and len(old_stock_list)>0:
            df_old_fin = get_peg_stock(context,old_stock_list)
            df_old_fin_peg = df_old_fin[(df_fin['pe']>0) &(df_old_fin['eps_0']<=50) &(df_old_fin['eps_0']>0)&(df_old_fin['PEG']>0) &(df_old_fin['PEG']<1) & \
            (df_old_fin['eps_0']>df_old_fin['std'])]#& (df_fin['eps_1']>0) &(df_fin['eps_2']>0) ]
            #df_fin_peg = df_fin_peg.sort('mktcap',ascending = 1)
            df_old_fin_peg = df_old_fin_peg.sort('mktcap',ascending = 1)
            fin_old_stock_list = list(df_old_fin_peg.index)[:5-len(fin_stock_list)]
            print('old_stock_list',len(fin_old_stock_list),fin_old_stock_list)
            fin_stock_list = fin_stock_list+fin_old_stock_list
        return fin_stock_list
    except:
        return []
    #计算

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
#
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

def goto_trade(context,ratio,value):
    trade_stock_list = ratio.keys()

    for stock in context.portfolio.positions:
        if stock not in trade_stock_list:
            order_target(stock,0)

    price_now = history(1,'1d','close',trade_stock_list)
    #print(type(price_now))
    for stock in trade_stock_list:
        stock_value_should=ratio[stock]*value
        if stock not in context.portfolio.positions:
            order_target_value(stock,stock_value_should)
        else:
            stock_value_now = context.portfolio.positions[stock].total_amount*price_now[stock].values
            cash = context.portfolio.available_cash
            if (abs(stock_value_should-stock_value_now)/stock_value_should) > 0.25 and cash >= 0.25*stock_value_should:
                order_target_value(stock,stock_value_should)
