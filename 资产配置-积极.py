#第六行的 mu 分别设为 0.0003、4、5、6、7 是风险偏好从低到高的五档。


import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers

mu = 0.0006
etfs = ['510630.XSHG', '510650.XSHG', '510660.XSHG', '159930.XSHE', '511880.XSHG', '165513.XSHE', '162703.XSHE']

maxweight = 0.5
recent_days = 120

def allocate(context):

    stocks = etfs

    prices = get_price(stocks, '2013-09-16', context.previous_date, '1d', 'close').close
    returns = (prices/prices.shift(1)).iloc[1:]
    rets = returns.dropna(axis=1)
    meanrets = np.array(np.mean(rets))-1

    recent_prices = history(recent_days, '1d', 'close', rets.columns)
    recent_rets = (recent_prices/recent_prices.shift(1)).iloc[1:]
    recent_meanrets = np.array(np.mean(recent_rets))-1

    meanrets = (meanrets + recent_meanrets) / 2


    covmat = np.cov(rets.T)

    stocks = list(rets.columns)

    P = matrix(covmat*10000, tc='d')
    q = matrix([0]*len(stocks), tc='d')
    G = matrix(np.concatenate((np.diag([-1]*len(stocks)), np.diag([1]*len(stocks)))), tc='d')
    h = matrix([0]*len(stocks)+[maxweight]*len(stocks), tc='d')
    A = matrix(np.array([meanrets*10000, [1]*len(stocks)]), tc='d')
    b = matrix([mu*10000, 1], tc='d')

    sol = solvers.qp(P,q,G,h,A,b)
    weights = np.array(list(sol['x']))

    usable_value = context.portfolio.total_value

    downs = []
    ups = []
    for i in range(len(stocks)):
        if weights[i]*usable_value < context.portfolio.positions[stocks[i]].value:
            downs.append(i)
        else:
            ups.append(i)
    for i in downs:
        order_target_value(stocks[i], weights[i]*usable_value)
    for i in ups:
        order_target_value(stocks[i], weights[i]*usable_value)

g.i=0
def run(context):
    if g.i==0:
        allocate(context)
        g.i=20
    else:
        g.i -= 1


def initialize(context):
    set_slippage(FixedSlippage(0))
    set_order_cost(OrderCost(0,0,0,0), 'fund')
    solvers.options['show_progress'] = False
    set_benchmark('000001.XSHG')
    log.set_level('order', 'error')
    run_daily(run, time='before_open')
