import pandas as pd

g.n = 5
g.c = 0
g.universe = []


def no_pause(pool):
    dt = get_current_data()
    return [stock for stock in pool if dt[stock].paused==False]


def get_cum_np(pool=[]):

    if pool:
        q = query(income.code, income.statDate, income.pubDate).filter(income.code.in_(pool))
    else:
        q = query(income.code, income.statDate, income.pubDate)
    df = get_fundamentals(q)
    df.index = df.code
    stat_dates = set(df.statDate)
    stat_date_stocks = { sd:[stock for stock in df.index if df['statDate'][stock]==sd] for sd in stat_dates }

    def quarter_push(quarter):
        if quarter[-1]!='1':
            return quarter[:-1]+str(int(quarter[-1])-1)
        else:
            return str(int(quarter[:4])-1)+'q4'

    q = query(income.code,
              income.np_parent_company_owners
              )

    stat_date_panels = { sd:None for sd in stat_dates }
    for sd in stat_dates:
        quarters = [sd[:4]+'q'+str(int(sd[5:7])/3)]
        for i in range(11):
            quarters.append(quarter_push(quarters[-1]))
        nq = q.filter(valuation.code.in_(stat_date_stocks[sd]))
        pre_panel = { quarter:get_fundamentals(nq, statDate = quarter) for quarter in quarters }
        for thing in pre_panel.values():
            thing.index = thing.code.values
        panel = pd.Panel(pre_panel)
        panel.items = range(len(quarters))
        stat_date_panels[sd] = panel.transpose(2,0,1)

    final = pd.concat(stat_date_panels.values(), axis=2)
    pnl = final.dropna(axis=2)
    df = pnl.np_parent_company_owners
    df = df.loc[:, np.min(df)>0]

    cum = pd.DataFrame([np.sum(df[i-4:i]) for i in range(4,len(df)+1)])
    goods = cum.T[(np.sum(cum-cum.shift(1)[1:]<0))<3].T

    return goods


def get_roe(cum_np):

    q = query(income.code, income.statDate, income.pubDate).filter(income.code.in_(cum_np.columns))
    df = get_fundamentals(q)
    df.index = df.code
    stat_dates = set(df.statDate)
    stat_date_stocks = { sd:[stock for stock in df.index if df['statDate'][stock]==sd] for sd in stat_dates }

    def quarter_push(quarter):
        if quarter[-1]!='1':
            return quarter[:-1]+str(int(quarter[-1])-1)
        else:
            return str(int(quarter[:4])-1)+'q4'

    q = query(balance.code,
              balance.total_assets
              )

    stat_date_panels = { sd:None for sd in stat_dates }
    for sd in stat_dates:
        quarters = [sd[:4]+'q'+str(int(sd[5:7])/3)]
        for i in range(len(cum_np)):
            quarters.append(quarter_push(quarters[-1]))
        nq = q.filter(valuation.code.in_(stat_date_stocks[sd]))
        pre_panel = { quarter:get_fundamentals(nq, statDate = quarter) for quarter in quarters[1:] }
        for thing in pre_panel.values():
            thing.index = thing.code.values
        panel = pd.Panel(pre_panel)
        panel.items = range(len(quarters)-1)
        stat_date_panels[sd] = panel.transpose(2,0,1)

    final = pd.concat(stat_date_panels.values(), axis=2)
    assets = final.total_assets
    roe = (cum_np/assets).dropna(axis=1)
    pred = 0.85*((np.mean(roe)-np.std(roe))-(np.mean(roe)-np.min(roe)))
    return pred, cum_np.iloc[-1]


def growth_factor(pred, cum_np):
    q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_(pred.index))
    df = get_fundamentals(q)
    df.index = df.code.values
    mc = df.market_cap
    peg = (mc/cum_np)/pred
    stocks = (cum_np>0)&(pred>0)
    return peg[stocks]


def get_signals(context):
    pool = no_pause(context.universe)
    K = get_cum_np(pool)
    Q,R = get_roe(K)
    S = growth_factor(Q, R)
    S.sort(inplace=False)
    return S[:g.n].index


def trade(to_buy, context):
    for stock in context.portfolio.positions:
        if stock not in to_buy and stock!='511880.XSHG':
            order_target(stock, 0)
    val = context.portfolio.total_value/max(len(to_buy), g.n)

    downs = []
    ups = []
    for stock in to_buy:
        d = context.portfolio.positions.get(stock, 0)
        if d==0 or d.value<val:
            ups.append(stock)
        else:
            downs.append(stock)

    cash_v = context.portfolio.total_value - val*len(to_buy)
    v = context.portfolio.positions.get('511880.XSHG', 0)
    if v==0 or v.value<cash_v:
        for s in downs:
            order_target_value(s, val)
        order_target_value('511880.XSHG', cash_v)
        for s in ups:
            order_target_value(s, val)
    else:
        order_target_value('511880.XSHG', cash_v)
        for s in downs:
            order_target_value(s, val)
        for s in ups:
            order_target_value(s, val)


def initialize(context):
    set_slippage(FixedSlippage(0))
    set_order_cost(OrderCost(0,0,0,0), 'stock')
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    run_daily(func, 'open')


def get_universe(context):
    if context.universe==g.universe:
        K = get_cum_np()
        Q,R = get_roe(K)
        S = growth_factor(Q, R)
        print type(S)
        S.sort(inplace=False)
        pool = list(S[:50].index)
        set_universe(pool)
        g.universe = pool


def func(context):
    if g.c == 0:
        get_universe(context)
        tb = get_signals(context)
        trade(tb, context)
        g.c = 10
    else:
        g.c -= 1
