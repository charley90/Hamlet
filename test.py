# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts


_CODE_INDEX = pd.DataFrame({'code':['000001','399001','399006'],'name':['上证指数','深证指数','创业板指数'],'c_name':['指数','指数','指数']})
code_index = _CODE_INDEX.set_index('code')
dat = ts.get_industry_classified()
dat = dat.drop_duplicates('code')


#print dat

import xgboost as xgb

xgb.




#if __name__=="__main__":





