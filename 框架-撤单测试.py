# 开基策略-0001 created by fangwei in 20170719
# 本策略设定为面向开放式货币基金（T+0 ,无手续费）
# 目前已测：000973、511880、511810 、511860
# 这个策略的当前目标是为了拉成交量，兼保本
# 20170727:改一下临时策略，每次只买卖1手，每日必清盘

#TODO: 重新改下落单函数！！！

import jqdata
import time


# 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    # 定义一个全局变量, 保存要操作的股票
    g.security = '000973.XSHE'
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 开放式货币基金的手续费暂设为0
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0000, close_commission=0.0000, close_today_commission=0, min_commission=0), type='fund')
    # 买卖标志
    g.s_flag = True
    # 委托单价格控制
    g.p_p = 100.00
    #撤单次数
    g.cancel_num=0
    # 前一个委托单
    g.l_order = None
    # 设置滑点（由于是货币基金，设低一点）
    set_slippage(FixedSlippage(0.01))
    # hold次数
    g.l_count = 0
    g.b_count = 0
    # 卖出价格控制线（000973 一般设为-0.0001）
    g.sell_ctrl_line = -0.0001




# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context, data):
#get_current_data 为返回现在市场的一个快照函数
    current_data = get_current_data()
    security = g.security
    #总权益
    total_value = context.portfolio.total_value
    # 取得当前的现金
    cash = context.portfolio.cash
    #当前市值
    mv = context.portfolio.positions_value
# 权益，现金，市值
    log.info('total_value = %d, current cash: %d , and positions value:%d' % (total_value, cash, mv))
#调取当前股票的跌停价个，作为最低价格
    p_p = round(current_data[security].low_limit + 0.01, 2)
    log.info("order 委托数量 100, 委托价格 %s" % p_p)
    o1 = order(security, 100, style=LimitOrderStyle(p_p))
    log.info(1)
    time.sleep(10)
    log.info(2)
    log.info(o1)
    log.info("order_target 委托数量 600, 委托价格 %s" % p_p)
    o2 = order_target(security=security, amount=600, style=LimitOrderStyle(p_p))
    log.info(o2)
    log.info('--------------------------------------------------------------------------------------')


    if context.current_dt.hour == 13 and context.current_dt.minute == 30:
        1/0
