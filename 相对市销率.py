g.c = 0
g.n = 5

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
              income.np_parent_company_owners,
              balance.total_assets,
              balance.total_non_current_liability,
              balance.total_non_current_assets,
              balance.total_current_assets,
              income.operating_revenue
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

    return final.dropna(axis=2)

def first_sieve(pnl):
    return pnl[:,:,(pnl.np_parent_company_owners>0).all()]

def second_sieve(pnl):
    yearly_np = pd.DataFrame([np.sum(pnl.np_parent_company_owners[i-4:i]) for i in range(5,len(pnl.major_axis)+1)])
    total_assets = pnl.total_assets[3:-1]
    total_assets.index = range(len(total_assets))
    roa = yearly_np/total_assets
    diffs = (roa - roa.shift(1))[1:]
    return pnl[:,:,(diffs>0).all()]

def third_sieve(pnl):
    leverage = pnl.total_non_current_liability/ pnl.total_non_current_assets
    stocks = np.sum(leverage[-4:])<np.sum(leverage[-8:-4])
    return pnl.loc[:,:,stocks]

def fourth_sieve(pnl):
    or1 = np.mean(pnl.operating_revenue[-8:-4])
    or2 = np.mean(pnl.operating_revenue[-4:])
    tca1 = np.mean(pnl.total_current_assets[-8:-4])
    tca2 = np.mean(pnl.total_current_assets[-4:])
    stocks = (or2/tca2) > (or1/tca1)
    return pnl[:,:,stocks]

def fifth_sieve(pnl):
    or1 = np.mean(pnl.operating_revenue[-8:-4])
    or2 = np.mean(pnl.operating_revenue[-4:])
    ta1 = np.mean(pnl.total_assets[-8:-4])
    ta2 = np.mean(pnl.total_assets[-4:])
    stocks = (or2/ta2) > (or1/ta1)
    return pnl[:,:,stocks]


def get_signal(context):
    q = query(valuation.code, valuation.pb_ratio)
    df = get_fundamentals(q)
    df.index = df.code.values
    low_pb = df.pb_ratio.sort(inplace=False)[:len(df)/5].index

    K = get_data(low_pb, 8)
    K = first_sieve(K)
    K = second_sieve(K)
    K = third_sieve(K)
    K = fourth_sieve(K)
    K = fifth_sieve(K)
    stocks = K.minor_axis.values

    print len(stocks)
    return stocks[:5]


def get_4q_np():

    q = query(valuation.code, income.statDate, income.pubDate)
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

    final = pd.concat(stat_date_panels.values(), axis=2)

    return np.sum(final.dropna(axis=2).np_parent_company_owners)


def get_pe(nps):
    q = query(valuation.market_cap).filter(valuation.code.in_(nps.index))
    mc = get_fundamentals(q).market_cap
    tmc = np.sum(mc)
    tnp = np.sum(nps)
    return tmc/tnp


def get_ratios():
    pe = get_pe(get_4q_np())*10000000
    cash_ratio = min(1, max(pe-1.5, 0)/2.5)
    print pe, 1-cash_ratio
    return 1-cash_ratio


def trade(to_buy, stock_ratio, context):

    for stock in context.portfolio.positions:
        if stock not in to_buy:
            order_target(stock, 0)

    val = stock_ratio*context.portfolio.total_value/g.n

    for stock in to_buy:
        order_target_value(stock, val)
	for stock in to_buy:
	    order_target_value(stock, val)

    order_value('511880.XSHG', context.portfolio.available_cash)


def func(context):
    if g.c==0:
        to_buy = get_signal(context)
        stock_ratio = get_ratios()
        trade(to_buy, stock_ratio, context)
        g.c = 21
    else:
        g.c -= 1

def initialize(context):
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    run_daily(func, '9:30')
