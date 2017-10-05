import statsmodels.api as sm
from statsmodels import regression
import numpy as np
import pandas as pd
#import time
#from datetime import date
from jqdata import *
import datetime
from dateutil.relativedelta import relativedelta
'''
================================================================================
总体回测前
================================================================================
'''

#总体回测前要做的事情
def initialize(context):
    set_params()        #1设置策参数
    set_variables()     #2设置中间变量
    set_backtest()      #3设置回测条件
    set_subportfolios([SubPortfolioConfig(cash=10000000,type='index_futures')])

#1
#设置策参数
def set_params():
    g.pre_future_IC=''     #用来装上次进入的期货合约名字
    g.pre_future_IF=''     #用来装上次进入的期货合约名字
    g.futures_margin_rate = 0.10   #股指期货保证金比例
    g.futures_multiplier_IC = 200  # IF和IH每点价值300元，IC为200元
    g.futures_multiplier_IF = 300  # IF和IH每点价值300元，IC为200元
    g.marketIndex_IC = '000905.XSHG'
    g.marketIndex_IF = '000300.XSHG'
#2
#设置中间变量
def set_variables():
    g.t = 0                     #运行天数
    g.in_position_stocks = []   #持仓股票

#3
#设置回测条件
def set_backtest():
    set_option('use_real_price', True) #用真实价格交易
    log.set_level('order', 'warning')
    # set_slippage(FixedSlippage(0))     #将滑点设置为0

'''
================================================================================
每天开盘前
================================================================================
'''
#每天开盘前要做的事情
def before_trading_start(context):
    log.info('---------------------------------------------------------------------')
    set_slip_fee(context)

#4 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 根据不同的时间段设置手续费
    dt=context.current_dt
    # log.info(type(context.current_dt))

    if dt>datetime.datetime(2013,1, 1):
        set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

    elif dt>datetime.datetime(2011,1, 1):
        set_commission(PerTrade(buy_cost=0.001, sell_cost=0.002, min_cost=5))

    elif dt>datetime.datetime(2009,1, 1):
        set_commission(PerTrade(buy_cost=0.002, sell_cost=0.003, min_cost=5))

    else:
        set_commission(PerTrade(buy_cost=0.003, sell_cost=0.004, min_cost=5))

    # 设置期货合约保证金
    if dt>datetime.datetime(2015,9,7):
        g.futures_margin_rate = 0.2
    else:
        g.futures_margin_rate = 0.1
    set_option('futures_margin_rate', g.futures_margin_rate)

'''
================================================================================
每天交易时
================================================================================
'''
#每个交易日需要运行的函数
def handle_data(context, data):

    # 获取下月连续合约 string
    current_future_IC = get_next_month_future(context, 'IC')
    current_future_IF = get_next_month_future(context, 'IF')

    # 获取期货指数价格
    index_price_IC = attribute_history(current_future_IC, 1, '1d', 'close').close.iloc[0]
    index_price_IF = attribute_history(current_future_IF, 1, '1d', 'close').close.iloc[0]

    #record(IC=index_price_IC)
    #record(IF=index_price_IF)

    IC=get_price('000905.XSHG',count=120, end_date=context.current_dt, frequency='1d', fields=['close'])
    IF=get_price('000300.XSHG',count=120, end_date=context.current_dt, frequency='1d', fields=['close'])
    IC=IC['close']/IC['close'][0]
    IF=IF['close']/IF['close'][0]

    x=IC-IF

    if x[-1]>x[-2]:
        g.IC='long'
        g.IF='short'
        log.info('多IC,空IF')
    else:
        g.IC='short'
        g.IF='long'
        log.info('多IF,空IC')
    #调仓
    rebalance(context)

    record(name=x[-1])
# 调仓函数
def rebalance(context):
    # 计算资产总价值
    total_value = context.portfolio.total_value
    # 计算预期的股票账户价值
    expected_value = np.round(total_value/2,0)


    # 计算相应的期货保证金价值
    futures_margin =expected_value * g.futures_margin_rate
    log.info('Target futures_margin: %.2f' % futures_margin)

    # 获取下月连续合约 string
    current_future_IC = get_next_month_future(context, 'IC')
    current_future_IF = get_next_month_future(context, 'IF')

    # 如果下月合约和原本持仓的期货不一样
    if g.pre_future_IC!='' and g.pre_future_IC!=current_future_IC:
        # 就把仓位里的期货平仓
        order_target(g.pre_future_IC, 0, side=g.IF)
    if g.pre_future_IF!='' and g.pre_future_IF!=current_future_IF:
        # 就把仓位里的期货平仓
        order_target(g.pre_future_IF, 0, side=g.IC)

    # 现有期货合约改为刚计算出来的
    g.pre_future_IC = current_future_IC
    # 现有期货合约改为刚计算出来的
    g.pre_future_IF = current_future_IF

    # 获取期货指数价格
    index_price_IC = attribute_history(current_future_IC, 1, '1d', 'close').close.iloc[0]
    log.info('Index futures_IC: %s, Price: %.2f' % (current_future_IC, index_price_IC))
    index_price_IF = attribute_history(current_future_IF, 1, '1d', 'close').close.iloc[0]
    log.info('Index futures_IF: %s, Price: %.2f' % (current_future_IF, index_price_IF))

    # 计算并调整需要的仓位
    nShortAmount = int(np.round(futures_margin/(index_price_IF* g.futures_multiplier_IF * g.futures_margin_rate),0))   # 目标手数
    nHoldAmount = context.portfolio.short_positions[current_future_IF].total_amount  #现持仓手数
    log.info('股指期货IF: %s, 现持仓手数: %d, 目标手数: %d' % (current_future_IF, nHoldAmount, nShortAmount))

    if nShortAmount != nHoldAmount:
        order = order_target(current_future_IF, nShortAmount, side=g.IF)
        if order != None and order.filled > 0:
            log.info('Futures: %s, action: short %s, filled: %d, price: %.2f' % \
                (order.security, ('平空' if order.is_buy else '开仓'), order.filled, order.price))
        else:
            log.info('Futures: %s, order failure' % (current_future_IF))

    # 计算并调整需要的仓位
    nLongAmount = int(np.round(futures_margin/(index_price_IC * g.futures_multiplier_IC * g.futures_margin_rate),0))   # 目标手数
    nHoldAmount = context.portfolio.long_positions[current_future_IC].total_amount  #现持仓手数
    log.info('股指期货IC: %s, 现持仓手数: %d, 目标手数: %d' % (current_future_IC, nHoldAmount, nLongAmount))

    if nLongAmount != nHoldAmount:
        order = order_target(current_future_IC, nLongAmount, side=g.IC)
        if order != None and order.filled > 0:
            log.info('Futures: %s, action: long %s, filled: %d, price: %.2f' % \
                (order.security, ('平多' if order.is_buy else '开仓'), order.filled, order.price))
        else:
            log.info('Futures: %s, order failure' % (current_future_IC))

# 取下月连续string
# 输入 context 和一个 string，后者是'IF'或'IC'或'IH'
# 输出一 string，如 'IF1509.CCFX'
# 进入本月第三周即切换到下月合约，而不等第三周的周五本月合约结束
def get_next_month_future(context, symbol):
    dt = context.current_dt
    month_begin_day = datetime.date(dt.year, dt.month, 1).isoweekday() # 本月1号是星期几(1-7)
    third_monday_date = 16 - month_begin_day + 7*(month_begin_day>5) #本月的第三个星期一是几号
    # 如果今天没过第三个星期一
    if dt.day < third_monday_date:
        next_dt = dt #本月合约
    else:
        next_dt = dt + relativedelta(months=1)  #切换至下月合约

    year = str(next_dt.year)[2:]
    month = ('0' + str(next_dt.month))[-2:]

    return (symbol+year+month+'.CCFX')
