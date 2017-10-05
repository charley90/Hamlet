enable_profile()
import pandas as pd
import numpy as np
import math

from statsmodels import regression
import time
from datetime import date
from jqdata import *
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def initialize(context):
    g.s1 = "000001.XSHG"
    g.heging='159937.XSHE'
    g.max_num_stocks = 100
    g.valid_num_stocks = 5

    g.OBSERVATION=20
    g.ma = 0
    g.lastma = 0

    # 交易总次数(全部卖出时算一次交易)
    g.trade_total_count = 0
    # 交易成功次数(卖出时盈利即算一次成功)
    g.trade_success_count = 0
    # 统计数组(win存盈利的，loss 存亏损的)
    g.statis = {'win': [], 'loss': []}

    # 使用真实价格回测
    set_option('use_real_price', True)

    #设置滑点
    set_slippage(FixedSlippage(0.1))

    g.blacklist=['300372.XSHE']

    g.i = 0 # 表明持有天数
    # 关闭订单提醒
    log.set_level('order', 'error')
    # 设定保证金比例
    set_option('futures_margin_rate', 0.30)
    # 设定手续费
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, \
                    close_commission=0.0003, min_commission=5), type='stock')
    # # 金融期货close_today_commission可不用设定，2015-09-07 之前不分平今平昨，之后平今仓默认0.00023
    set_order_cost(OrderCost(open_commission=0.000023, close_commission=0.000023, \
                    ), type='index_futures')


    init_cash = 5000000 # 定义股票初始资金
    # 设定subportfolios[0]为股票和基金仓位，初始资金为 init_cash 变量代表的数值
    # 设定subportfolios[1]为金融期货仓位，初始资金为初始资金减去 init_cash
    set_subportfolios([SubPortfolioConfig(cash=init_cash ,type='stock'),\
                   SubPortfolioConfig(cash=context.portfolio.starting_cash- init_cash,type='index_futures'),\
                   SubPortfolioConfig(cash=0,type='index_futures')])


    run_daily(IF_trading, time='open')
    run_daily(trading, time='open')
    run_daily(liquidation, time='14:45', reference_security='000300.XSHG')


    print 'initialize'

def before_trading_start(context):
    fundamental_df = get_fundamentals(
        query(valuation.code, valuation.market_cap)
        .order_by(valuation.market_cap.asc())
        .limit(g.max_num_stocks)
    )

    #Update context
    stocks = list(fundamental_df['code'])
    g.stocks = list(fundamental_df['code'])
    g.fundamental_df = fundamental_df
    stocks.append(g.s1)
    stocks.append(g.heging)
    set_universe(stocks)

    print 'before_trading_start'

    if g.i==0:
        rebalance(context)


def rebalance(context):
    stock_value=context.subportfolios[0].total_value
    IF_value=context.subportfolios[1].total_value
    mid_value=(stock_value+IF_value)/2

    if stock_value>IF_value:
        change=stock_value-mid_value
        log.info('转移前stock的可取资金：%s 元' %context.subportfolios[0].transferable_cash)
        log.info('转移前IF的可取资金：%s 元' %context.subportfolios[1].transferable_cash)
        transfer_cash(0,1,change)
        log.info('transfer_cash from 0 to 1,money=%s'%change)
        log.info('stock money=%s'%context.subportfolios[0].total_value)
        log.info('IF money=%s'%context.subportfolios[1].total_value)

    if stock_value<IF_value:
        change=IF_value-stock_value
        log.info('转移前stock的可取资金：%s 元' %context.subportfolios[0].transferable_cash)
        log.info('转移前IF的可取资金：%s 元' %context.subportfolios[1].transferable_cash)
        transfer_cash(1,0,change)
        log.info('transfer_cash from 1 to 0,money=%s'%change)
        log.info('stock money=%s'%context.subportfolios[0].available_cash)
        log.info('IF money=%s'%context.subportfolios[1].available_cash)

def IF_trading(context):

    IF = get_current_month_future(context, 'IF')
    g.IF = IF
    stock_list = [IF]
    hist = history(1, '1m', 'close', stock_list)
    # 股指期货价格
    IF_close = hist[IF][0]

    end_data = get_CCFX_end_date(IF)

    if (context.current_dt.date() == end_data):
        pass
    else:
        if g.i == 0 and (len(context.subportfolios[1].short_positions) == 0):

            g.amount = int(context.subportfolios[1].available_cash*0.9 / (IF_close * 300 * 0.3))

            print '计划交易%s手'%g.amount
            print '交易前可用现金：%s'%context.subportfolios[1].available_cash

            order(IF, g.amount, side='short', pindex=1)
            print '交易后可用现金：%s'%context.subportfolios[1].available_cash
            # log.info('0', context.subportfolios[0])
            # log.info('1', context.subportfolios[1])

    g.i += 1
    print '持有天数：%s'%g.i

def clear_all_stocks(context):
    for stock in list(context.subportfolios[0].long_positions.keys()):
        sell_amount(context,stock,0)
        print 'sell:%s'%stock
    if len(context.subportfolios[0].long_positions.keys())==0:
        print '已清仓'




# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def trading(context):
    print 'handle_data'
    have_set = set(context.portfolio.positions.keys())

    holds = g.stocks

    #去掉新股
    not_new_filter = not_new_stock(context)
    holds = list(filter(not_new_filter, holds))

    #去掉停牌
    current_data = get_current_data()
    is_trading_filter = is_trading(current_data)
    holds = list(filter(is_trading_filter, holds))

    #去掉ST
    st_data = get_extras('is_st',holds, start_date='2005-01-01', end_date=context.current_dt.strftime("%Y-%m-%d"))
    not_st_filter = not_st_stock(st_data)
    holds = list(filter(not_st_filter, holds))

    #去掉退市
    not_delisted_filter=not_delisted()
    holds=list(filter(not_delisted_filter,holds))

    #去掉带*的
    not_star_filter=not_star()
    holds=list(filter(not_star_filter,holds))

    #去掉黑名单
    for s in g.blacklist:
        if s in holds:
            holds.remove(s)

    #if not len(holds) == 5:
        #print(holds)

    holds = holds[:g.valid_num_stocks]

    #止损策略


    if should_clear(context):
        holds = []

    hold_set = set(holds)

    to_buy = hold_set - have_set
    to_sell = have_set - hold_set
    print 'tobuy:%s'%to_buy
    print 'tosell:%s'%to_sell
    print_win_rate(context.current_dt.strftime("%Y-%m-%d"), context.current_dt.strftime("%Y-%m-%d"), context)

    for stock in to_sell:
        if not low_enough(stock):
            #order_target(stock, 0)
            sell_amount(context,stock,0)
            print 'sell:%s'%stock

    if len(to_buy) == 0:
        return

    each = context.portfolio.cash/len(to_buy)
    # if each > context.portfolio.portfolio_value/(context.valid_num_stocks-2):
    #     each = context.portfolio.portfolio_value/(context.valid_num_stocks-2)

    for stock in to_buy:
        if not high_enough(stock):
            pricenow = history (1, '1m', 'close')[stock].ix[0]
            volume = int(each/pricenow/100*0.998) * 100
            if volume > 0:
                #order_target(stock, volume)
                buy_amount(stock,volume)
                print 'buy:%s'%stock

    record(stock=context.subportfolios[0].available_cash)
    record(IF=context.subportfolios[1].available_cash)




#去掉新股
def not_new_stock(context):
    log.info('not_new_stock')
    def make_filter(stock):
        if stock in context.portfolio.positions.keys():
            return True
        else:
            return not high_enough(stock)
    return make_filter

#去掉ST
def not_st_stock(data):
    log.info('not_st_stock')
    def make_filter(stock):
        return not list(data[stock])[-1]
    return make_filter

#去掉退市
def not_delisted():
    log.info('not_delisted')
    def make_filter(stock):
        return not '退' in get_security_info(stock).display_name
    return make_filter

#去掉带*股票
def not_star():
    log.info('not_star')
    def make_filter(stock):
        return not '*' in get_security_info(stock).display_name
    return make_filter

#去掉停牌
def is_trading(bar_dict):
    log.info('is_trading')
    def make_filter(stock):
        return not bar_dict[stock].paused
    return make_filter

#大盘止损
def should_clear(context):
    log.info('should_clear')
    g.lastma = g.ma
    g.ma = sum(history(g.OBSERVATION,'1d','close',security_list=[g.s1])[g.s1][:])/g.OBSERVATION
    #if g.lastma == 0:
        #return True

    return g.ma < g.lastma

#止盈
def high_enough(stock):
    price = history (2, '1d', 'close')[stock].ix[0]
    pricenow = history (1, '1m', 'close')[stock].ix[0]
    pct_change = (pricenow - price) / price
    if math.isnan(price):
        return True
    return pct_change>0.09

#止损
def low_enough(stock):
    price = history (2, '1d', 'close')[stock].ix[0]
    pricenow = history (1, '1m', 'close')[stock].ix[0]
    pct_change = (pricenow - price) / price
    if math.isnan(price):
        return True
    return pct_change<-0.09



#----------------------------------------------------------
def get_current_month_future(context, symbol):
    dt = context.current_dt
    month_begin_day = datetime.date(dt.year, dt.month, 1).isoweekday()
    third_friday_date = 20-month_begin_day + 7*(month_begin_day>5)
    # 如果没过第三个星期五或者第三个星期五（包括）至昨日的所有天都停盘
    if dt.day<=third_friday_date or (dt.day>third_friday_date and not any([datetime.date(dt.year, dt.month, third_friday_date+i) in get_all_trade_days() for i in range(dt.day-third_friday_date)])):
        year = str(dt.year)[2:]
        month = str(dt.month)
    else:
        year = str(dt.year+dt.month//12)[2:]
        month = str(dt.month%12+1)
    if len(month)==1:
        month = '0'+month
    return(symbol+year+month+'.CCFX')

def get_CCFX_end_date(fature_code):
    return get_security_info(fature_code).end_date

def liquidation(context):
    '''
    在到期日平仓
    '''
    # log.info(context.subportfolios[1].is_dangerous(0.2))
    code = get_current_month_future(context, 'IF')
    end_data = get_CCFX_end_date(code)
    print end_data
    print code
    if context.current_dt.date() == end_data:
        log.info('交割日:', context.current_dt.date())
        if len(context.subportfolios[1].short_positions) > 0:
            log.info('交割日平仓')
            order_target(g.IF, 0, side='short', pindex=1)
            #股票也清仓、重新平衡分配资金
            clear_all_stocks(context)
            g.i = 0



def transfer_cash_form_x(context,x,y,money):
    '''
    从序号为 from_pindex 的 subportfolio 转移 cash 到序号为 to_pindex 的 subportfolio
    '''

    log.info('转移前%s的可取资金：%s 元' %(x,context.subportfolios[x].transferable_cash))
    log.info('转移前%s的可取资金：%s 元' %(y,context.subportfolios[y].transferable_cash))

    if money==0:
        transferable_cash = context.subportfolios[x].transferable_cash
        transfer_cash(x, y,transfer_cash)


    transfer_cash(x,y,money)

    log.info('转移后%s的可取资金：%s 元' %(x,context.subportfolios[x].transferable_cash))
    log.info('转移后%s的可取资金：%s 元' %(y,context.subportfolios[y].transferable_cash))








#-------------------------------------------------------
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
