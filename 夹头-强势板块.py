#TODO：目前考虑指标就收益，感觉方法比较单一
#TODO：应能比较方便的调出板块个股的收益分布，或者通过分布部的偏度说明一下是否是个别股票的行为，
#TODO：应该考虑到板块的成交量信息，似乎应该作为第二点
#TODO: 最后返回的应该是一个或者多个强势龙头股，如何定义龙头，收益和还手率，量比，人气等因素


import math

def initialize(context):
    # 定义行业指数list
    g.indexList = ['A01','A02','A03','A04','A05','B06',\
     'B07','B08','B09','B11','C13','C14','C15','C17','C18',\
     'C19','C20','C21','C22','C23','C24','C25','C26','C27',\
     'C28','C29','C30','C31','C32','C33','C34','C35','C36',\
     'C37','C38','C39','C40','C41','C42','D44','D45','D46',\
     'E47','E48','E50','F51','F52','G53','G54','G55','G56',\
     'G58','G59','H61','H62','I63','I64','I65','J66','J67',\
     'J68','J69','K70','L71','L72','M73','M74','N77','N78',\
     'P82','Q83','R85','R86','R87','S90']

#每天运行函数find
    run_daily(find,time='10:00')

def find(context):

#做了一个强制的整体定义平均收益大于2%，才能算强势
    most_change=0.02
    most_change_industry=0
##清空所有仓位
    for stock in context.portfolio.positions:
        order_target_value(stock, 0)
#一个行业一个行业的看
    for industry_code in g.indexList:
        print "计算%s中..."%(industry_code)
#返回行业平均收益率（整体收益/股票个数） 和 最后一个股票
        industry_change,A_stock=get_mean_change(industry_code)
#比较各个板块之间的平均收益
        if industry_change>most_change :
            most_change=industry_change
            most_change_industry=industry_code
        print "目前涨幅最大的是%s，涨幅%s"%(most_change_industry,most_change)

    if not most_change_industry==0 and not A_stock==0:
        buy(context,A_stock)

#通过函数化的方法来提高复用，其实应该是三重的循环
def get_mean_change(industry_code):
#获取该行业的股票列表
    stocks=get_industry_stocks(industry_code)
#加载一分前所有的股票快照
    dic=history(1, unit='1m', field='close', security_list=stocks, df=False)
#加载昨日的收盘价快照，df=False 按照就转换成了字典的方式了
    dic_yesterday=history(1, unit='1d', field='close', security_list=stocks, df=False)

    change=0
    number=0
    most_change_stock=10
    A_stock=0

#股票池中安股票迭代
    for stock in stocks:
#跳过停牌的股票
        if not math.isnan(dic[stock]) and not math.isnan(dic_yesterday[stock]):
#计算该只股票的收益
            stock_change=(1-(float(dic[stock])/float(dic_yesterday[stock])))*100
#计算整个板块的收益
            change=change+stock_change
#统计共享的股票数
            number=number+1
#？这块有问题，逻辑是如果板块大于10就积累随机最后的股票并返回
            if change>most_change_stock:
                most_change_stock=stock_change
                A_stock=stock
#通过均值算板块平均收益？感觉不科学，1至少还应该返回一个分布 2没有考虑到市值的因素
    mean_change=change/number

    return mean_change,A_stock


def buy(context,buystock):
    cash=context.portfolio.cash
    order_value(buystock, cash)
