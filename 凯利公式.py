#【演示 2】动量策略（Kelly公式优化版）
# 回测 ： 2007-1-1 到 2016-8-23  ￥1000000 ，每天

'''
================================================================================
总体回测前
================================================================================
'''
def initialize(context):
    # 计时器
    g.t=0
    # 调仓周期
    g.tc=2
    # 动量周期
    g.N=60
    # 初始化 tatal_value 最高值
    g.high=0
    # 最大回撤的上限
    g.drawdown=0.6
    # 控制handle_data的全局变量
    g.if_trade=False
    # 设置参考基准
    set_benchmark('000300.XSHG')
    # 设置报错等级
    log.set_level('order','error')


'''
================================================================================
每天开盘前
================================================================================
'''
def before_trading_start(context):
    # 每g.tc天，调仓一次
    if g.t%g.tc==0:
        g.if_trade=True
        # 设置股票池
        g.stock='000300.XSHG'
    g.t+=1


'''
================================================================================
每天交易时
================================================================================
'''
# 每天回测时做的事情
def handle_data(context,data):
    if g.if_trade == True:
        # 更新 total_value 最高价
        g.high= max(g.high,context.portfolio.total_value)
        # 取一个动量周期的每日收盘价
        close=list(attribute_history(g.stock, g.N, '1d', 'close').close)

        #~~~~~~~~~~~~~  Kelly公式优化 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 计算资产变动率v
        v=[y/x for (x,y) in zip(close[:-1],close[1:])]
        # 先计算资产变动率的均值，标准差
        mean1 =mean(v)
        var1 =(np.std(v))**2
        # 计算每个二项分布的均值，方差
        mean2=np.log(mean1)
        var2=np.log(var1/mean1**2 + 1)
        # 计算出kelly最佳下注比
        f_star=mean2/var2
        # 设置要买入的额度
        position = g.high*g.drawdown*f_star
        #~~~~~~~~~~~~~  Kelly公式优化 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # 如果今日收盘价大于动量周期开始时的值，发出动量买入信号
        if close[-1]>close[0]:
            # 买入
            order_target_value(g.stock, position)
        else:
            # 若不满足动量买入信号，则空仓
            order_target_value(g.stock, 0)
    g.if_trade = False

'''
================================================================================
每天收盘后
================================================================================
'''
# 若收盘后需要进行长运算，可调用此函数（本策略中不需要）
def after_trading_end(context):
    return
