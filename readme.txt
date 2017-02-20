Hamlet 项目为测试性量化投资项目.

从TradeStation ,MT4等软件中学习择时和风险控制模块,结合ML的预测技术,希望能够为账户提供稳定的绝对收益.

如果有任何问题可以联系 3097420997@qq.com.


#TODO: 完成psp类的方法: 1调用时间序列 2,python调用psql调用python从断点处同步数据 3 能够进行时间框架的切换 4 能够进行K线绘图
#TODO: 完成pfp类的方法: 1规整数据 2热敏图 3组合的alpha,beta,SR,IR等信息 4 绘制可行曲线图
#TODO: 完成fp类的方法:1添加各种特征,与psp类隔绝一个公共知识一个是私人使用 2 后期使用标识符号来掩盖特征名称,项目保密
#TODO: 完成fp-tap类方法:1 主要以自适应的均线作为基础特征 2 添加hurst指标识别混沌系统
#TODO: 完成fp-fqp类方法:1 以<主动投资组合管理> 介绍的为基准
#TODO: 完成fp-lp?类方法:1 与市场,行业指数的特异度  2,影子股的相关度
#TODO: 完成fp-mdp类方法:1 添加book信息
#TODO: 完成fp-op?类方法:1 添加其他信息如 关注度,情绪识别
#TODO: 使用机器学习的方法,生成模型,在回测的时候调用生成的模型进行判断
#TODO: 采用logistics回归,样本正例为15日内涨幅超过15%的且绝对跌幅不超过5%的,样本负例为15日内跌幅超5%.特征基本为左侧交易特征.学习模型,每日返回胜率最高的股票,结合3的风报比自动调整头寸.
#TODO: 实现用日线级别决定买卖,用分钟线级别决定进出场
#TODO: 完成pp类方法:1 添加风险控制方法 2 添加仓位调整方法
#TODO: 完成op类方法:1 调用内存数据库存储定制化的订单信息
#TODO: 回测框架中的订单中添加magicnumber ,完成复杂的订单动作
#TODO: 完成ot类方法: 1 结合模型和 op返回的magicnumber pp 等信息 2 落单完成
#TODO: 结合1 股指期货空头 2 逆回购  做风险控制以及闲散资金管理
#TODO: 使用ricequant的回测平台回测自己策略:1资金曲线图的解读
#TODO: 将模型作为包导入到ricequant 上面作为信号源输出,发送微信信号,发送到雪球持仓.





##Quantopian
— 可以做空
— 如果不检查仓位,仓位满后会融资操作
- 在个bar跑的时候使用recode来追踪指定的指标,如leveage
- schedule_founction (fuc,date_rulse,time_rules.market_opne(hours=1))定期操作
- 有较习惯的动态展示页面
- fromquantopian.pipeline.filters.morningstar importQ1500US 从moringstar导入
- 元数据使用的blaze
- Pipeline  用于得到复合索引的一种方法
from quantopian.pipeline import Pipeline
def make_pipeline():
	return Pipeline()
from quantopian.research import run_pipeline

result=run_pipeline(make_pipeline(),sd,ed)
- alphalens 做因子分析 tearsheet
import alphalens
alphalens.tears.create_factor_tear_sheet(factor=result[‘sentiment’],
								price=pricing,
								quantiles=2,
								periods=(1,5,10))
factor 需要传入复合索引的序列值
详细解释显示tearsheet图表的含义 https://www.youtube.com/watch?v=BCLgXjxYONg
- 将因子联合使用
- 将因子运用于回测之中
- research 中调用回测结果
 bt=get_backtest(’回测哈希码’)
bt.create_full_tear_sheet()

##使用 IBpy
#使用IBpy进行实盘交易 https://www.youtube.com/watch?v=Bu0kpU-ozaw
github.com/blampe/IbPy
from ib.opt import Connection,message
from ib.ext.Contract import Contract
from ib.ext.Order import Order

def make_contract(symbol,sec_type,exch,prim_exch,curr):
	Contract.m_symbol=symbol
	Contract.m_secType=sec_type
	Contract.m_primaryExch=prim_exch
	Contract.m_currency=curr
	return Contract
def make_under(action,quantity,price=None):
	if price is not None:
		order=Order()
		order.m_orderType=‘LMT’
		order.m_totalQuantity=quantity
		order_m_action=action
		order.m_lmtPrice=price
	else:
		order=Order()
		order.m_orderType=‘MKT’
		order.m_totalQuantity=quantity
		order_m_action=action
	return order

def main():
	conn=Connection.create(port=,clientId=)
	conn.connect()

	oid=1
	cont=make_contract(‘tSLA’,’STK’,’’SMART,’SMART’,’USD’)
	offer=make_order(‘BUY’,1,200)

      	conn.PlaceOrder(oid,cont,offer)
	conn.disconnect()

## 使用IBridgePy  在Spyder中进行实盘交易
