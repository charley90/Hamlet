g.n = 3
g.c = 0


def get_good():
    df = get_fundamentals(
        query(valuation.code, valuation.market_cap, valuation.pe_ratio, valuation.ps_ratio, valuation.pb_ratio)
    )

    # 1) 总市值全市场从大到小前80%
    fCap = df.sort(['market_cap'], ascending=[False])
    fCap = fCap.reset_index(drop = True)
    fCap = fCap[0:int(len(fCap)*0.8)]
    sListCap = list(fCap['code'])

    # 2）市盈率全市场从小到大前40%（剔除市盈率为负的股票）
    fPE = df.sort(['pe_ratio'], ascending=[True])
    fPE = fPE.reset_index(drop = True)
    fPE = fPE[fPE.pe_ratio > 0]
    fPE = fPE.reset_index(drop = True)
    fPE = fPE[0:int(len(fPE)*0.05)]
    sListPE = list(fPE['code'])

    # 3）pb全市场从小到大前40%（剔除pb为负的股票）
    fPB = df.sort(['pb_ratio'], ascending=[True])
    fPB = fPB.reset_index(drop = True)
    fPB = fPB[fPB.pb_ratio > 0]
    fPB = fPB.reset_index(drop = True)
    fPB = fPB[0:int(len(fPB)*0.05)]
    sListPB = list(fPB['code'])

    # 4）市收率小于2.5
    fPS = df[df.ps_ratio < 1.5]
    sListPS = list(fPS['code'])

    # 5）同时满足上述3条的股票，按照股息率从大到小排序，选出股息率最高的 n 只股票
    good_stocks = list(set(sListCap) & set(sListPE) & set(sListPS) & set(sListPB))

    return good_stocks


def get_data(pool):

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
              income.operating_revenue,
              income.operating_cost
              )

    stat_date_panels = { sd:None for sd in stat_dates }
    for sd in stat_dates:
        quarters = [sd[:4]+'q'+str(int(sd[5:7])/3)]
        for i in range(3):
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

    df = pnl.operating_revenue-pnl.operating_cost

    return np.sum(df)


def get_data2(pool):

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

    q = query(balance.code,
              balance.total_assets
              )

    stat_date_panels = { sd:None for sd in stat_dates }
    for sd in stat_dates:
        quarters = [sd[:4]+'q'+str(int(sd[5:7])/3)]
        for i in range(3):
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

    df = pnl.total_assets

    return np.sum(df)



def func(context):
    if g.c==0:
        good = get_good()
        if len(good)==0:
            tb = []
        else:
            df1 = get_data(good)
            df2 = get_data2(good)
            gp = (df1/df2).dropna().sort(ascending=False, inplace = False)
            tb = gp[5:8].index
        trade(tb, context)
        g.c = 10
    else:
        g.c -= 1

def trade(to_buy, context):
    for stock in context.portfolio.positions:
        if stock not in to_buy:
            order_target(stock, 0)

    val = context.portfolio.total_value/max(len(to_buy),1)

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
    set_benchmark('000922.XSHG')
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    log.set_level('history', 'error')
    log.set_level('strategy', 'error')
    run_daily(func, 'open')
