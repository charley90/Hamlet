



enable_profile()
# 定义一个全局变量, 保存要操作的证券

g.security = ["600519.XSHG"]
# 初始化此策略
# 设置我们要操作的股票池, 这里我们只操作一支股票
set_universe(g.security)

#股票数量
stocknum=len(g.security)
#均分资金
g.money =100000/stocknum




# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context, data):
    buylist=[]

    for stock in g.security:

        if stock not in context.portfolio.positions:
            buylist.append(stock)

    buylist=list(set(buylist))


    #股票数量
    stocknum=len(g.security)-len(context.portfolio.positions)

    #当前现金
    cash=context.portfolio.cash
    #均分资金
    if not stocknum ==0:
        g.money =cash/stocknum

    for stock in buylist:
        if not data[stock].isnan():
            amount = int(g.money/data[stock].price/100.0) * 100
            order(stock, +amount)
            log.info(str(context.current_dt) + " Buying %s" % (stock)+"%s"%(+amount))
