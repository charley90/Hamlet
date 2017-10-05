g.n = 10
g.c = 0
g.days = 50
g.ratio = 0.1
g.universe = []


def no_pause(pool):
    dt = get_current_data()
    return [stock for stock in pool if dt[stock].paused==False]

def get_4q_np(pool):

    q = query(income.code, income.statDate, income.pubDate).filter(income.code.in_(pool))
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
        for i in range(4-1):
            quarters.append(quarter_push(quarters[-1]))
        nq = q.filter(valuation.code.in_(stat_date_stocks[sd]))
        pre_panel = { quarter:get_fundamentals(nq, statDate = quarter) for quarter in quarters }
        for thing in pre_panel.values():
            thing.index = thing.code.values
        panel = pd.Panel(pre_panel)
        panel.items = range(len(quarters))
        stat_date_panels[sd] = panel.transpose(2,0,1)

    try:
        final = pd.concat(stat_date_panels.values(), axis=2)
        nps = final.dropna(axis=2).np_parent_company_owners
        nps = nps.T[np.sum(nps>0)==4].T
        return np.sum(nps)
    except:
        return []


def get_pe(pool):
    nps = get_4q_np(pool)
    if len(nps)>0:
        prices = history(1, '1d', 'close', nps.index).iloc[0]
        pe = prices/nps
        return pe.sort(inplace=False)
    else:
        return []


def each_ind_pick(ratio):
    industries = ['HY00'+str(x) for x in range(1, 10)] + ['HY010', 'HY011']
    res = []
    for ind in industries:
        stocks = get_industry_stocks(ind)
        if len(stocks)>0:
            pes = get_pe(stocks)
            if len(pes)>0:
                low_pes = list(pes[:int(len(pes)*ratio)].index)
                res += low_pes
    return res


def low_vol(pool, days):
    pausation = history(days+20, '1d', 'paused', pool)
    nonpausation = (1-pausation).all()
    nonpaused = nonpausation[nonpausation].index

    prices = history(days, '1d', 'close', nonpaused)
    std = np.std(prices)
    mean = np.mean(prices)
    vol = (std/mean).sort(inplace=False)
    return vol


def get_signals(context):
    pool = no_pause(context.universe)
    vol = low_vol(pool, g.days)
    return vol[:g.n].index


def trade(to_buy, context):
    for stock in context.portfolio.positions:
        if stock not in to_buy:
            order_target(stock, 0)

    val = context.portfolio.total_value/g.n

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



def initialize(context):
    set_slippage(FixedSlippage(0))
    set_order_cost(OrderCost(0,0,0,0), 'fund')
    set_order_cost(OrderCost(0,0,0,0), 'stock')
    set_option('use_real_price', True)
    log.set_level('order', 'error')


def get_universe(context):
    if context.universe == g.universe:
        pool = each_ind_pick(g.ratio)
        vol = low_vol(pool, g.days)
        universe = list(vol[:10].index)
        set_universe(universe)
        g.universe=universe


def handle_data(context, data):
    if g.c == 0:
        get_universe(context)
        tb = get_signals(context)
        trade(tb , context)
        g.c = 10
    else:
        g.c -= 1
