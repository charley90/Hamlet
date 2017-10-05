def initialize(context):
    # 使用真实价格回测
    set_option('use_real_price', True)
    #设置滑点
    set_slippage(PriceRelatedSlippage(0.01))

    # 交易总次数(全部卖出时算一次交易)
    g.trade_total_count = 0
    # 交易成功次数(卖出时盈利即算一次成功)
    g.trade_success_count = 0
    # 统计数组(win存盈利的，loss 存亏损的)
    g.statis = {'win': [], 'loss': []}

    run_daily(watch,'9:34')
    run_daily(change, '9:35')

    log.info('initialize')

def watch(context):
    log.info('watching...')
    log.info(context.current_dt)
    stocks = list(get_all_securities(['fjb']).index)
    df1=get_price(stocks, count=4, frequency='1m',end_date=context.current_dt,fields=['close'])
    #df=history(4, unit='1m', field='close', security_list=stocks)

    #df=(df.iloc[0]-df.iloc[-1])/df.iloc[0]
    df1=(df1['close'].iloc[0]-df1['close'].iloc[-1])/df1['close'].iloc[0]

    #df= df.dropna().order(ascending=False)
    df1= df1.dropna().order(ascending=False)

    #df=df.head(3)
    df1=df1.head(3)

    print df1

    holds=list(df1.index)

    log.info(holds)

    have_set = set(context.portfolio.positions.keys())
    hold_set = set(holds)

    g.to_buy = hold_set - have_set
    g.to_sell = have_set - hold_set

    #发送持仓信息
   # message1='准备卖出：%s'%(g.to_sell)
   # send_message(message1, channel='weixin')
   # print message1
   # message2='准备买入：%s'%(g.to_buy)
   # send_message(message2, channel='weixin')
   # print message2


def change(context):
    print '--------------------------------------------------------'
    stocks = list(get_all_securities(['fjb']).index)
    df=history(4, unit='1m', field='close', security_list=stocks)

    df=(df.iloc[0]-df.iloc[-1])/df.iloc[0]

    df= df.dropna().order(ascending=False)

    df=df.head(3)

    print df




    log.info('changing...')
    log.info(context.current_dt)

    for stock in g.to_sell:
        sell_amount(context,stock,0)
        print 'sell:%s'%stock

    if len(g.to_buy) == 0:
        return
    each = context.portfolio.cash/len(g.to_buy)

    for stock in g.to_buy:

        #price=get_price(stock, count=1, frequency='1m',end_date=context.current_dt, fields='close')['close']
        price=round(float(history(1, unit='1m', field='close', security_list=stock)[stock]),3)

        try:
            volume = int(each/price/100*0.998) * 100
        except:
            volume=0

        if volume > 0:
            buy_amount(stock,volume)
            print 'buy:%s'%stock



# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context, data):

    pass

# 买入指定数量股票
def buy_amount(stock, amount):
    if 100 <= amount:
        order(stock, +amount)

# 卖出指定数量股票，若amount为0则表示清空该股票的所有持仓
def sell_amount(context, stock, amount):
    if 0 == amount:
        record_trade_count(context, stock)
        __amount = context.portfolio.positions[stock].sellable_amount
        order_target_value(stock, 0)
    else:
        order(stock, -amount)

# 记录交易次数便于统计胜率
def record_trade_count(context, stock):
    g.trade_total_count += 1
    amount = context.portfolio.positions[stock].total_amount
    avg_cost = context.portfolio.positions[stock].avg_cost
    price = context.portfolio.positions[stock].last_sale_price

    current_value = amount * price
    cost = amount * avg_cost

    percent = round((current_value - cost) / cost * 100, 2)
    if current_value > cost:
        g.trade_success_count += 1
        win = [stock, percent]
        g.statis['win'].append(win)
    else:
        loss = [stock, percent]
        g.statis['loss'].append(loss)

# 打印胜率
def print_win_rate(current_date, print_date, context):
    print current_date,print_date
    if str(current_date) == str(print_date):
        win_rate = 0
        if 0 < g.trade_total_count and 0 < g.trade_success_count:
            win_rate = round(g.trade_success_count / float(g.trade_total_count), 2)

        most_win = statis_most_win_percent()
        most_loss = statis_most_loss_percent()
        starting_cash = context.portfolio.starting_cash
        total_profit = statis_total_profit(context)

        if len(most_win)!=0 and len(most_loss)!=0:
            print "-"
            print '------------绩效报表------------'
            print '交易次数: {0}, 盈利次数: {1}, 胜率: {2}'.format(g.trade_total_count, g.trade_success_count, str(win_rate * 100) + str('%'))
            print '单次盈利最高: {0}, 盈利比例: {1}%'.format(most_win['stock'], most_win['value'])
            print '单次亏损最高: {0}, 亏损比例: {1}%'.format(most_loss['stock'], most_loss['value'])
            print '总资产: {0}, 本金: {1}, 盈利: {2}'.format(starting_cash + total_profit, starting_cash, total_profit)
            print '--------------------------------'
            print "-"
        else:
            print len(most_win),len(most_loss)
# 统计单次盈利最高的股票
def statis_most_win_percent():
    result = {}
    for statis in g.statis['win']:
        if {} == result:
            result['stock'] = statis[0]
            result['value'] = statis[1]
        else:
            if statis[1] > result['value']:
                result['stock'] = statis[0]
                result['value'] = statis[1]

    return result

# 统计单次亏损最高的股票
def statis_most_loss_percent():
    result = {}
    for statis in g.statis['loss']:
        if {} == result:
            result['stock'] = statis[0]
            result['value'] = statis[1]
        else:
            if statis[1] < result['value']:
                result['stock'] = statis[0]
                result['value'] = statis[1]

    return result

# 统计总盈利金额
def statis_total_profit(context):
    return context.portfolio.portfolio_value - context.portfolio.starting_cash
    
