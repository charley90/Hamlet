# 克隆自聚宽文章：https://www.joinquant.com/post/1402
# 标题：【量化课堂】雪球云蒙银行股搬砖
# 作者：JoinQuant量化课堂

# 雪球云蒙银行股搬砖
# 2009-11-01 到 2016-04-01, ￥1000000, 每天

import copy
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from six import StringIO
'''
================================================================================
总体回测前
================================================================================
'''
#总体回测前要做的事情
def initialize(context):
    set_params()    #1设置策参数
    set_variables() #2设置中间变量
    set_backtest()  #3设置回测条件

#1
#设置策略参数
#云蒙更改这里
def set_params():
    g.num = 4  # 持仓数量
    g.k = 0.05 # 每次搬砖必须保证的利润
    # 获取“货币金融服务”行业当日的所有股票，设为初始股票池
    g.stocks = get_industry_stocks('J66')
    g.data_body = read_file("bankshares.csv")
    g.rating_data = pd.read_csv(StringIO(g.data_body))


#2
#设置中间变量
def set_variables():
    # 生成存放当前评级的字典
    g.current_rate = dict.fromkeys(g.stocks, 0)
    # 生成存放当前权重的字典
    g.current_weight = dict.fromkeys(g.stocks, 0)
    # Whether it is initial transaction
    g.first = 1

#3
#设置回测条件
def set_backtest():
    set_benchmark("000300.XSHG") # 更改bench回测基准（银行股指数从2009年9月开始）
    set_option('use_real_price',True) # 用真实价格交易
    log.set_level('order','error')    # 设置报错等级




'''
================================================================================
每天开盘前
================================================================================
'''
#每天开盘前要做的事情
def before_trading_start(context):
    # 设置手续费与手续费
    set_slip_fee(context)

#4
# 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 根据不同的时间段设置手续费
    dt=context.current_dt

    if dt>datetime.datetime(2013,1, 1):
        set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

    elif dt>datetime.datetime(2011,1, 1):
        set_commission(PerTrade(buy_cost=0.001, sell_cost=0.002, min_cost=5))

    elif dt>datetime.datetime(2009,1, 1):
        set_commission(PerTrade(buy_cost=0.002, sell_cost=0.003, min_cost=5))

    else:
        set_commission(PerTrade(buy_cost=0.003, sell_cost=0.004, min_cost=5))




'''
================================================================================
每天交易时
================================================================================
'''
# 每日回测时要做的事
def handle_data(context, data):
    if g.first == 1:
        initial_transaction(context, data)
    else:
        routine_transaction(context, data)

#5
# 每日更新银行股评级
# 输入：context, data（见API文档）
# 输出：none
def renew_rating(context, data):
    # 查询当日财务数据
    fundamentals = get_fundamentals(
            query(valuation.code, valuation.pb_ratio,balance.statDate, balance.pubDate
            ).filter(valuation.code.in_(g.stocks)), date=context.current_dt)
    # 得到相关评级数据并存在字典中
    for code in fundamentals['code']:
        i = list(fundamentals['code']).index(code)
        report_date = fundamentals['statDate'][i]

        j = list(g.rating_data['code']).index(code)
        r = g.rating_data[report_date][j]/fundamentals['pb_ratio'][i]

        g.current_rate[code] = r

#6
# 初始化第一个交易日的信息，并执行买卖操作
# 输入：context, data（见API文档）
# 输出：none
def initial_transaction(context, data):
    renew_rating(context, data)

    # 初始排序
    sorted_initial = sorted(g.current_rate.iteritems(),
        key=lambda x:x[1], reverse = True)

    # 初始交易
    for i in range(0, g.num):
        i_code = sorted_initial[i][0]
        g.current_weight[i_code] = g.num - i
        if g.num==1:
            order_amt=context.portfolio.cash
        elif g.num>0:
            order_amt=context.portfolio.cash*2*(g.num-i)/(g.num*(g.num-1))
        order_value(i_code,order_amt)
    g.first = 0

#7
# 日常交易
# 输入：context, data（见API文档）
# 输出：none
def routine_transaction(context, data):
    # stocks = g.stocks
    old = g.current_rate
    # 计算旧评级数据
    sorted_old = sorted(old.iteritems(),
        key=lambda x:x[1], reverse  = True)
    # 计算旧权重数据
    old_weight = copy.deepcopy(g.current_weight)
    renew_rating(context, data)
    new = g.current_rate
    # Order the new rating data
    sorted_new = sorted(new.iteritems(),
        key=lambda x:x[1], reverse = True)
    new_weight = dict.fromkeys(g.stocks, 0)
    for i in range(0, g.num):
        i_code = sorted_new[i][0]
        new_weight[i_code] = g.num - i

    # Loop from right
    for i in range(0, len(g.stocks)):
        i_code = sorted_new[-(i+1)][0]
        # If old weight is larger then new weight
        if old_weight[i_code] > new_weight[i_code]:
            # Loop from right
            for j in range(0, min(g.num, len(g.stocks)-i-1))[::-1]:
                j_code = sorted_new[j][0]
                # If exceed the discount ratio
                if g.current_rate[j_code] > (1+g.k)*g.current_rate[i_code]:
                    # Changing the weight
                    if g.current_weight[j_code] < new_weight[j_code]:
                        while g.current_weight[j_code] < new_weight[j_code] and g.current_weight[i_code] > new_weight[i_code]:
                            g.current_weight[j_code] += 1
                            g.current_weight[i_code] -= 1
                    elif g.current_weight[j_code] > new_weight[j_code]:
                        while g.current_weight[j_code] > new_weight[j_code] and g.current_weight[i_code] < new_weight[i_code]:
                            g.current_weight[j_code] -= 1
                            g.current_weight[i_code] += 1

    # print g.current_weight
    total_shares = sum(g.current_weight.values())
    # Daily routine transaction
    for code in g.stocks:
        if g.current_weight[code] == 0:
            if code in context.portfolio.positions:
                order_target(code, 0)
        elif g.current_weight[code] < old_weight[code]:
            order_target_value(code,
                context.portfolio.portfolio_value*g.current_weight[code]/total_shares)
        elif g.current_weight[code] > old_weight[code]:
            order_target_value(code,
                context.portfolio.portfolio_value*g.current_weight[code]/total_shares)




'''
================================================================================
每天收盘后
================================================================================
'''
# 每日收盘后要做的事情（本策略中不需要）
def after_trading_end(context):
    return
