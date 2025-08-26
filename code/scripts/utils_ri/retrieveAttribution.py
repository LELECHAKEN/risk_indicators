'''
Description: retrieve Asset Attribution from database.
Author: Wangp
Date: 2021-03-02 16:35:15
LastEditTime: 2021-04-22 14:00:59
LastEditors: Wangp
'''

import pandas  as pd
import numpy as np

import cx_Oracle as cx
from ..db import OracleDB, sqls_config

def AssetAllocation_t(conn, prods, t, fundtype='3'):
    if type(prods) != list:
        prods = [prods]
    prods_str = ','.join('\'' + x + '\'' for x in prods)
    tablename = {'1': 'ALLOCATIONDETAIL_INTER', '3': 'ALLOCATIONDETAIL_INTER_ZH'}

    q = sqls_config['asset_allocation_t']['Sql']%(tablename[fundtype], prods_str, t)
    res = conn.read_sql(q)
    
    return res


def AssetAllocation_interval(conn, prods, startDate, endDate, fundtype='3'):
    if type(prods) != list:
        prods = [prods]
    prods_str = ','.join('\'' + x + '\'' for x in prods)
    tablename = {'1': 'ALLOCATIONDETAIL_INTER', '3': 'ALLOCATIONDETAIL_INTER_ZH'}

    q = sqls_config['asset_allocation_interval']['Sql']%(tablename[fundtype], prods_str, startDate, endDate)
    res = conn.read_sql(q)

    return res


def AssetAllocation_all(conn, prods, dateDF, fundtype='1'):
    dateDF.columns = ['C_FULLNAME', 'startDate', 'endDate']
    res = pd.DataFrame()
    for prod in prods:
        # print(prod)
        startDate = dateDF.loc[dateDF['C_FULLNAME'] == prod, 'startDate'].values[0]
        endDate = dateDF.loc[dateDF['C_FULLNAME'] == prod, 'endDate'].values[0]
        temp = AssetAllocation_interval(conn, prod, startDate, endDate, fundtype)
        res = res.append(temp, sort=False)
    
    res = res.sort_values(by=['c_fullname', 'd_date'])
    return res


def AttributionCumulation(x, method='selfCum', col_cum=''):
    if method == 'selfCum':
        res = (1+x).cumprod().iloc[-1] - 1
    else:
        y1 = (1+x[col_cum]).cumprod().shift(1).fillna(1)
        res = (1+x).multiply(y1, axis='index').iloc[-1] - 1
        
    return res