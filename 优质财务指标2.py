# 导入函数库
import jqdata
import pandas as pd
# 初始化函数，设定基准等等
def initialize(context):
    set_slippage(FixedSlippage(0))
    set_order_cost(OrderCost(0,0,0,0), 'fund')
    set_order_cost(OrderCost(0,0,0,0), 'stock')
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    g.universe=[]
    g.list1 = []
    g.b_l = {}
    g.hp = {}
    g.i = 20
    g.lose = 0.8
    g.ornum = 40
    g.list_q = []
    run_daily(func, 'open')


def trade(to_buy, context):
    for stock in context.portfolio.positions:
        if stock not in to_buy:
            order_target(stock, 0)

    val = context.portfolio.total_value/5

    downs = []
    ups = []
    for stock in to_buy:
        d = context.portfolio.positions.get(stock, 0)
        if d==0 or d.value<val:
            ups.append(stock)
        else:
            downs.append(stock)

    for s in downs:
        order_target_value(s, val)
    for s in ups:
        order_target_value(s, val)


def func(context):
    g.i += 1
    stop(context)
    if g.i-20 == 1:
        cy = context.current_dt.year
        py = cy-1
        get_universe(context, py)
        bjsky(context, py)
        trade(g.list1, context)
        g.i = 0
    else:
        pass


def no_pause(pool):
    dt = get_current_data()
    return [stock for stock in pool if dt[stock].paused==False]


def bjsky(context, abc):
    pool = no_pause(context.universe)
    q1 = query(indicator.code,valuation.pb_ratio,indicator.roe,indicator.roa,balance.total_current_liability,balance.total_current_assets,balance.inventories,cash_flow.net_operate_cash_flow,income.net_profit)
    df1 = get_fundamentals(q1)
    df1 = df1.sort(['pb_ratio'])
    a = int(0.1*len(df1.index))
    #df1 = df1.iloc[:a,:]
    df1['current_ratio'] = df1.total_current_assets/df1.total_current_liability
    df1['quick_ratio'] = (df1.total_current_assets-df1.inventories)/df1.total_current_liability
    df1['gap']= df1.net_operate_cash_flow-df1.net_profit
    df1.index = df1.code
    df2 = df1.sort(['current_ratio'])
    df3 = df1.sort(['quick_ratio'])
    b = int(0.2*len(df2.index))
    #df2 = df2.iloc[:b,:]
    #df3 = df3.iloc[:b,:]
    df4 = pd.concat([df2,df3],axis = 0)
    q2 = query(indicator.code,indicator.roe,indicator.roa)
    df = get_fundamentals(q2, statDate = abc )
    df.index = df.code
    df = df.ix[:,1:]
    df.columns = ['proe','proa']
    dff = pd.concat([df1,df],axis =1)
    dff['groe'] = dff.roe - dff.proe
    dff['groa'] = dff.roa - dff.proa
    dff = dff.dropna(axis = 0, how = 'any')
    dff = dff[(dff.pb_ratio>0)&(dff.roe>0)&(dff.roa>0)&(dff.gap>0)&(dff.proe>0)&(dff.proa>0)]
    dff = dff[(dff.groe>0)&(dff.groa>0)]
    dff = dff.loc[pool]
    dff = dff.sort(['pb_ratio'])
    g.list1 = list(dff.code)
    g.list1 = g.list1[:5]
    return g.list1


def get_universe(context, abc):
    if context.universe==g.universe:
        q1 = query(indicator.code,valuation.pb_ratio,indicator.roe,indicator.roa,balance.total_current_liability,balance.total_current_assets,balance.inventories,cash_flow.net_operate_cash_flow,income.net_profit)
        df1 = get_fundamentals(q1)
        df1 = df1.sort(['pb_ratio'])
        a = int(0.1*len(df1.index))
        #df1 = df1.iloc[:a,:]
        df1['current_ratio'] = df1.total_current_assets/df1.total_current_liability
        df1['quick_ratio'] = (df1.total_current_assets-df1.inventories)/df1.total_current_liability
        df1['gap']= df1.net_operate_cash_flow-df1.net_profit
        df1.index = df1.code
        df2 = df1.sort(['current_ratio'])
        df3 = df1.sort(['quick_ratio'])
        b = int(0.2*len(df2.index))
        #df2 = df2.iloc[:b,:]
        #df3 = df3.iloc[:b,:]
        df4 = pd.concat([df2,df3],axis = 0)
        q2 = query(indicator.code,indicator.roe,indicator.roa)
        df = get_fundamentals(q2, statDate = abc )
        df.index = df.code
        df = df.ix[:,1:]
        df.columns = ['proe','proa']
        dff = pd.concat([df1,df],axis =1)
        dff['groe'] = dff.roe - dff.proe
        dff['groa'] = dff.roa - dff.proa
        dff = dff.dropna(axis = 0, how = 'any')
        dff = dff[(dff.pb_ratio>0)&(dff.roe>0)&(dff.roa>0)&(dff.gap>0)&(dff.proe>0)&(dff.proa>0)]
        dff = dff[(dff.groe>0)&(dff.groa>0)]
        dff = dff.sort(['pb_ratio'])
        g.list1 = list(dff.code)
        g.list1 = g.list1[:10]
        set_universe(g.list1)
        g.universe=g.list1

#止损
def stop(context):
    s = context.portfolio.positions.keys()
    list4 = list(s)
    price = history(1,'1d', 'close', security_list=list4)
    for security in list4:
        price_now = price[security][-1]
        price_ji = context.portfolio.positions[security].avg_cost
        if security not in g.hp.keys():
            g.hp[security] = price_now
        elif price_now >= g.hp[security]:
            g.hp[security] = price_now
        elif price_now < g.lose*g.hp[security]:
            g.b_l[security] = 0
            order_target_value(security, 0)
        else:
            pass

def blacklist(context):
    g.list_q = list(g.b_l.keys())
    for security in g.list_q:
        g.b_l[security]+= 1
        if g.b_l[security] > g.ornum:
            del g.b_l[security]

def clearance(context):
    s = context.portfolio.positions.keys()
    lists = list(s)
    for security in lists:
        order_target_value(security, 0)
