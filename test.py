
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt











if __name__="__main__":
    dates = pd.date_range('2010-01-01', '2016-12-31')  # 时间跨度范围
    symbols = ['GOOG', 'IBM', 'GLD']  # 股票池
    df = get_date(symbols, dates)  # 直接取到股票池中指定日期的部件