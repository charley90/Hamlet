def initialize(context):
    # 定义一个全局变量, 保存要操作的股票
    # 初始化此策略
    # 设置我们要操作的股票池, 这里我们只操作一支股票
    g.prepare_to_sell = {}
    g.balance_info = {}
    g.max_value = 1000000
    g.balanced = 1
    set_benchmark('000300.XSHG')
    set_universe(['000300.XSHG'])
    #set_universe(get_index_stocks('000001.XSHG','2015-12-08')+get_index_stocks('399106.XSHE','2015-12-08')+get_index_stocks('399006.XSHE','2015-12-08'))


# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context, data):

    smalls = get_fundamentals(query(
        valuation.code, valuation.market_cap, valuation.capitalization,valuation.circulating_cap,valuation.circulating_market_cap
    ).filter(
        valuation.market_cap <= 20
    ))
    new_smalls = []
    for idx in range(0, len(smalls['code']), 1):
        si = attribute_history(smalls['code'][idx], 1, '1d', ('open','close','price','pre_close','high_limit','low_limit','high','low','volume'), False)
        if (si['volume'][-1] > 0):
            new_smalls.append({'code':smalls['code'][idx], 'close':si['close'][-1], 'market_cap':smalls['market_cap'][idx]})


    sorted_smalls = sorted(new_smalls, key=lambda small : small['market_cap'])


    for stock in sorted_smalls:
        buy = 0
        if context.portfolio.cash < 10000:
            break;
        buy = min(context.portfolio.cash,min(context.portfolio.portfolio_value/10,g.max_value))
        try:
            if None != order_value(stock['code'], buy):
                log.info("buy %s about %s money %s market_cap %s cash %s" % (stock['code'],stock['close'], buy, stock["market_cap"],context.portfolio.cash))
        except:
            log.info("order_value error")


    handle_sell(context,data, sorted_smalls)

def handle_sell(context, data, sorted_smalls):

    #for code,idx in g.prepare_to_sell.items():
    #    if order_target(code, 0) != None:
    #        del(g.prepare_to_sell[code])
    #        log.info("retry sell %s" % (code))
    day = context.current_dt.strftime("%y-%m")
    if g.balance_info.get(day,None) != None:
        return
    log.info('balance %s' % day)
    sell=0
    g.balance_info[day] = 1
    for security in context.portfolio.positions:
        si = attribute_history(security, 1, '1d', ('open','close','price','pre_close','high_limit','low_limit','high','low','volume'), False)
        if si['volume'][-1] == 0:
            continue;
        for idx in range(0,len(sorted_smalls),1):
            if idx > len(context.portfolio.positions) and (sorted_smalls[idx]["market_cap"]>10 or g.balanced == 0):
                order_target(security, 0)
                sell += 1
                log.info("sell %s market_cap %s sell1 cash %s" % (security,sorted_smalls[idx]["market_cap"],context.portfolio.cash))
                break

            if sorted_smalls[idx]["code"] == security:
                if sorted_smalls[idx]["market_cap"] > 20:
                    order_target(security, 0)
                    sell += 1
                    log.info("sell %s market_cap %s sell2 cash %s" % (security,sorted_smalls[idx]["market_cap"],context.portfolio.cash))
                break
    if sell <= 1:
        g.balanced = 0
    else:
        g.balanced = 1
