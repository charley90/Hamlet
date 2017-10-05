# 克隆自聚宽文章：https://www.joinquant.com/post/8143
# 标题：对MACD底背离策略改写的改写——30秒极速版
# 作者：jqz1226

import talib
import numpy as np
from enum import Enum


class MacdSignal(Enum):
    gold = 0
    dead = 1
    other = 2


def initialize(context):
    set_order_cost(
        OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
        'stock')
    g.owning = 5  # 持仓股票数####
    g.summit = {}  # 区间最高价
    set_option('use_real_price', True)
    log.set_level('order', 'error')


def before_trading_start(context):
    # 设置预选股票(小市值+0<PE<90，剔除ST)
    g.muster = do_select()

    # 清理股票峰值信息
    tmp_summit = {}
    for stock in g.summit:
        if stock in context.portfolio.positions:
            tmp_summit[stock] = g.summit[stock]

    g.summit = tmp_summit


def handle_data(context, data):
    curr_data = get_current_data()
    # H1、清仓操作--------------------------------------------------遍历持仓股票
    for stock in context.portfolio.positions:
        # SS、记录股票峰值信息
        if g.summit.get(stock, 0) < data[stock].high:
            g.summit[stock] = data[stock].high

        # SP、跳过3停:停牌、涨停、跌停的股票
        if curr_data[stock].paused:
            continue

        if not (curr_data[stock].low_limit < curr_data[stock].day_open < curr_data[stock].high_limit):
            continue

        # S1、目前持仓不在预选股票池中(g.muster)则清仓
        if stock not in g.muster:
            close_position(stock, '越界清仓')
            continue

        # S2、回撤10%则清仓
        if curr_data[stock].day_open / g.summit[stock] < 0.9:
            close_position(stock, '回撤清仓')
            continue

        # S3、指标卖出信号：死叉清仓
        if make_decision(stock) == MacdSignal.dead:
            close_position(stock, '死叉清仓')
            continue

    # HS、趋势判断--------------------------------------------------判断大盘走势
    if not market_safety('000300.XSHG'):
        return

    # H2、建仓操作--------------------------------------------------遍历预选股票
    gold_stocks = []
    hold_num = len(context.portfolio.positions)  # 已经持仓的只数
    if hold_num < g.owning:
        for stock in g.muster:
            # B1、指标买入信号选股
            if make_decision(stock) == MacdSignal.gold:
                gold_stocks.append(stock)

            if len(gold_stocks) + hold_num == g.owning:
                break

    if len(gold_stocks) > 0:
        open_positions(context, gold_stocks)  # 建仓


# 择时：大盘趋势======================================================================
def market_safety(index, fastperiod=11, slowperiod=26, signalperiod=5):
    rows = slowperiod * 3
    grid = attribute_history(security=index, count=rows, unit='1d', fields=['close'], df=False)
    _close = grid['close']  # type: np.ndarray
    _dif, _dea, _macd = talib.MACD(_close, fastperiod, slowperiod, signalperiod)

    # MA_Type: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
    wma20 = talib.MA(_close, 20, matype=2)

    # macd > 0 且收盘价站在20日WMA均线之上
    return _macd[-1] > 0 and _close[-1] > wma20[-1]


# --------------------- 选股 ------------------------------------
# 过滤停牌、退市、ST股票、涨停、跌停
def filter_specials(stock_list):
    curr_data = get_current_data()
    stock_list = [stock for stock in stock_list if \
                  (not curr_data[stock].paused)  # 未停牌
                  and (not curr_data[stock].is_st)  # 非ST
                  and ('ST' not in curr_data[stock].name)
                  and ('*' not in curr_data[stock].name)
                  and ('退' not in curr_data[stock].name)
                  and (curr_data[stock].low_limit < curr_data[stock].day_open < curr_data[stock].high_limit)  # 未涨跌停
                  ]

    return stock_list


def do_select():
    """
    stock_to_choose = get_fundamentals(query(
        valuation.code, valuation.pe_ratio,
        valuation.pb_ratio,valuation.market_cap,
        indicator.eps, indicator.inc_net_profit_annual
    ).filter(
        valuation.pe_ratio < 40,
        valuation.pe_ratio > 0,#30,
        indicator.eps > 0.2,
        indicator.inc_net_profit_annual > 0.20,
        indicator.roe > 0.20
    ).order_by(
        valuation.pb_ratio.asc()
    ).limit(
        50), date=None)
    """
    # 399649.XSHE	中小红利
    # 000015.XSHG    红利指数
    # 000826.XSHG	民企红利 OK
    # 399324.XSHE    深圳红利
    # 399411.XSHE    国证红利
    # 000035.XSHG	上证可选 OK
    # 000036.XSHG	上证消费
    # 000937.XSHG	中证公用
    # 399006.XSHE	创业板指
    # 399409.XSHE	小盘高贝
    # 399405.XSHE	大盘高贝
    # 000134.XSHG	上证银行 OK
    # 000849.XSHG	沪深300非银行金融指数
    # 399966.XSHE	800非银
    # HY493	多元化银行	1991-04-04
    # HY494	区域性银行	2007-07-20
    # HY004	可选消费指	1999-12-30

    stockpool = get_industry_stocks('HY493') + get_industry_stocks('HY494') + get_index_stocks(
        '000826.XSHG')

    q = query(
        valuation.code, valuation.pe_ratio,
        valuation.pb_ratio, valuation.market_cap,
        indicator.eps, indicator.inc_net_profit_annual
    ).filter(
        # valuation.pe_ratio < 40,
        valuation.pe_ratio > 0,  # 30,
        indicator.eps > 0,
        # indicator.inc_net_profit_annual > 0.05,
        # indicator.roe > 0.20,
        valuation.code.in_(stockpool)
    ).order_by(
        valuation.pb_ratio.desc()  # asc()valuation.market_cap.asc()
    ).limit(
        550
    )

    stock_to_choose = get_fundamentals(q)
    stockpool = list(stock_to_choose['code'])
    log.info('选出股票只数: %d' % len(stockpool))

    # 过滤停牌、退市、ST股票、涨停、跌停
    stockpool = filter_specials(stockpool)

    return stockpool


def make_decision(stock, fastperiod=11, slowperiod=26, signalperiod=5):
    ret_val = MacdSignal.other

    rows = (fastperiod + slowperiod + signalperiod) * 5
    h = attribute_history(security=stock, count=rows, unit='1d', fields=['close'], df=False)
    _close = h['close']  # type: np.ndarray
    _dif, _dea, _macd = talib.MACD(_close, fastperiod, slowperiod, signalperiod)

    # ----------- 底背离 ------------------------
    # 1.昨天[-1]金叉
    # 1.昨天[-1]金叉close < 上一次[-2]金叉close
    # 2.昨天[-1]金叉Dif值 > 上一次[-2]金叉Dif值
    if _macd[-1] > 0 > _macd[-2]:  # 昨天金叉
        # idx_gold: 各次金叉出现的位置
        idx_gold = np.where((_macd[:-1] < 0) & (_macd[1:] > 0))[0] + 1  # type: np.ndarray
        if len(idx_gold) > 1:
            if _close[idx_gold[-1]] < _close[idx_gold[-2]] and _dif[idx_gold[-1]] > _dif[idx_gold[-2]]:
                ret_val = MacdSignal.gold

    # ----------- 顶背离 ------------------------
    # 1.昨天[-1]死叉
    # 1.昨天[-1]死叉close > 上一次[-2]死叉close
    # 2.昨天[-1]死叉Dif值 < 上一次[-2]死叉Dif值
    if _macd[-1] < 0 < _macd[-2]:  # 昨天死叉
        # idx_dead: 各次死叉出现的位置
        idx_dead = np.where((_macd[:-1] > 0) & (_macd[1:] < 0))[0] + 1  # type: np.ndarray
        if len(idx_dead) > 1:
            if _close[idx_dead[-1]] > _close[idx_dead[-2]] and _dif[idx_dead[-1]] < _dif[idx_dead[-2]]:
                ret_val = MacdSignal.dead

    return ret_val


# 下单函数======================================================================
def my_log_order(o_order, c_action):
    # type: (Order, str) -> None
    if o_order is not None:
        if o_order.filled > 0:
            log.info('[%s %s]成交: %s %s, 数量: %d, 价格: %.2f' % (
                c_action, '买入' if o_order.is_buy else '卖出',
                o_order.security, get_security_info(o_order.security).display_name,
                o_order.filled, o_order.price))


def open_positions(context, buy_stocks):
    val = context.portfolio.available_cash / len(buy_stocks)
    for stock in buy_stocks:
        o_ord = order_target_value(stock, val)
        my_log_order(o_ord, '金叉建仓')


def close_position(stock, action):
    o_ord = order_target(stock, 0)
    my_log_order(o_ord, action)
