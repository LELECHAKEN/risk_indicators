'''
Description: 
Author: Wangp
Date: 2020-09-09 09:08:35
LastEditTime: 2020-09-09 14:53:08
LastEditors: Wangp
'''
import json
import time
import numpy as np
import pandas as pd

from ..db.db_utils import JYDB_Query

def __retrieveJYData(code_list, date_list):    # 从JY取YTM和Duration数据
    if type(date_list) != list:
        if type(date_list) != str:
            date_list = date_list.strftime('%Y-%m-%d')
        date_list = [date_list]       

    dq = JYDB_Query()
    data_jy = pd.DataFrame(columns=['ENDDATE', 'SECUCODE','SECUMARKET', 'VPYIELD', 'VPADURATION', 'VPCONVEXITY', 'CREDIBILITYCODE', 'CREDIBILITYDESC', 'YIELDCODE'])
    for temp_date in date_list:
        jy_temp = dq.sec_query('bond_yield', code_list, temp_date)
        data_jy = data_jy.append(jy_temp, sort=False)
    data_jy = _dealCode(data_jy)
    data_jy = data_jy.rename(columns={'ENDDATE': 'D_DATE', 'VPADURATION': 'Duration', 'VPCONVEXITY': 'Convexity'})
    dq.close_query()

    return data_jy

def _retrieveClause(code_list):
    dq = JYDB_Query()
    sec_data = dq.sec_query('bond_clause', code_list)
    sec_data = _dealCode(sec_data)
    dq.close_query()

    return sec_data

def _dealCode(data):
    data0 = data.copy()
    data0['mkt'] = data0['SECUMARKET'].map({89: '.IB', 83: '.SH', 90: '.SZ'})
    data0['code'] = data0['SECUCODE'] + data0['mkt']

    return data0.drop(columns=['SECUCODE', 'SECUMARKET', 'mkt'])

def _dealRepurchaseDate(code_list):
    dq = JYDB_Query()
    data_rep = dq.sec_query('bond_repurchase', code_list, 201)   # 201为回售权
    data_rep = _dealCode(data_rep).drop(columns=['OPTYPE']).rename(columns={'EXPECTEDEXERCISEDATE': 'REPURCHASEDATE'}).dropna(subset=['REPURCHASEDATE'])
    data_rep = data_rep.groupby('code')['REPURCHASEDATE'].apply(lambda x: ','.join(i.strftime('%Y-%m-%d') for i in x)).reset_index()    # 将JY所有回售日期合并为一条

    data_call = dq.sec_query('bond_repurchase', code_list, 101)  # 101为赎回权
    data_call = _dealCode(data_call).drop(columns=['OPTYPE']).rename(columns={'EXPECTEDEXERCISEDATE': 'CALLDATE'}).dropna(subset=['CALLDATE'])
    data_call = data_call.groupby('code')['CALLDATE'].apply(lambda x: ','.join(i.strftime('%Y-%m-%d') for i in x)).reset_index()

    data_option = pd.merge(data_rep, data_call, on='code', how='outer')
    dq.close_query()

    return data_option

def _dealInterestChg(wind_code):
    dq = JYDB_Query()
    jy = dq.sec_query('bond_interest_chg', wind_code)
    jy = self._dealCode(jy)
    dq.close_query()

    # 处理发生过票面利率调整的债券现金流信息
    jy['CASHFLOW_orig'] = jy['CASHFLOW'].copy()
    jy['CASHFLOW'] = jy['INTERESTTHISYEAR'] + jy['PAYMENTPER']

    return jy

def _dealOptionDate(date_put, baseDay):
    date_list = sorted(date_put.replace(' ','').split(','))
    baseDay = pd.to_datetime(baseDay)
    date_list = pd.to_datetime(date_list)
    if len(date_list[date_list >= baseDay]) == 0:
        return pd.to_datetime('2099-01-01')
    date_target = date_list[date_list >= baseDay][0]   # 选择距离baseDay最近的一个行权日

    return date_target

def _dealPuttableCash(x):
    total_cash = x.loc[x['PAYMENTDATE'] >= x['date_target'], 'PAYMENTPER'].sum()
    date_target = x['date_target'].iloc[0]

    res = pd.DataFrame([date_target, total_cash], index=['PAYMENTDATE', 'CASHFLOW']).T
    res['CASHFLOWTYPE'] = 4
    res['CASHFLOWTYPEDESC'] = '回售行权'

    return res

def _idOptionEmbedded(code_list, baseDay):
    '''划分行权&不行权的债券清单'''
    data_jy = __retrieveJYData(code_list, baseDay)
    data_clause = _retrieveClause(code_list)
    data = pd.merge(data_clause, data_jy, on='code', how='left')

    data0 = _dealRepurchaseDate(code_list)
    data0 = data.loc[data['CLAUSE'].notna(), ['code', 'CLAUSE', 'REPURCHASEDATE', 'CALLDATE']].drop_duplicates()
    data1 = data_jy.loc[data_jy['YIELDCODE'] == 2, ['D_DATE', 'code']].drop_duplicates()
    data0['D_DATE'] = baseDay
    data = pd.merge(data0, data1, on=['D_DATE', 'code'], how='left')
    if data.shape[0] == 0:
        return [], []
    
    data['票面利率调整'] = [1 if '调整票面利率' in x else 0 for x in data['CLAUSE']]
    data['回售or赎回'] = [1 if ('回售' in x ) or ('赎回' in x) else 0 for x in data['CLAUSE']]
    codes0 = data.loc[data['票面利率调整'] == 1, 'code'].unique().tolist()
    codes1 = data.loc[(data['回售or赎回'] == 1) & (data['REPURCHASEDATE'].notna()|data['CALLDATE'].notna()), 'code'].unique().tolist()
    codes0 = list(set(codes0) - set(codes1))    # 回售优先于调整票面利率

    return codes0, codes1

def _deal2Option(wind_code, tb, baseDay):
    '''处理含权债的行权日期，并按照行权日重新归集现金流'''
    if type(wind_code) != list:
        wind_code = [wind_code]
    date_option = _dealRepurchaseDate(wind_code)
    date_option = date_option.loc[date_option['code'].isin(wind_code) & (date_option['CALLDATE'].notna() | date_option['REPURCHASEDATE'].notna()), ['code', 'REPURCHASEDATE', 'CALLDATE']].drop_duplicates()
    date_option['rep_target'] = [_dealOptionDate(x, baseDay) if x != '*' else np.nan for x in date_option['REPURCHASEDATE'].fillna('*')]
    date_option['call_target'] = [_dealOptionDate(x, baseDay) if x != '*' else np.nan for x in date_option['CALLDATE'].fillna('*')]
    date_option['date_target'] = date_option[['rep_target', 'call_target']].min(axis=1)
    
    tb_1 = tb.loc[tb['code'].isin(wind_code), :].copy()
    tb_1 = pd.merge(tb_1, date_option, on=['code'], how='left')
    tb_2 = tb_1.groupby('code').apply(_dealPuttableCash).reset_index()        # 回售日后所有现金流归集到回售日当天
    tb_f = tb_1.loc[tb_1['PAYMENTDATE'] < tb_1['date_target'], :].append(tb_2, sort=False)
    tb_f = tb_f.sort_values(['code', 'PAYMENTDATE'])

    return tb_f

def _map_key_year_t1(key_years, year):
    idx = key_years.index(year)
    year_t1 = key_years[idx+1]

    return year_t1
    
def _calc_key_cash(tb):
    '''转换各期限现金流'''
    key_cash_t0 = tb.groupby(['code', 'key_year'])['cash_t'].sum()
    key_cash_t1 = tb.groupby(['code', 'key_year_t1'])['cash_t1'].sum()
    key_cash = key_cash_t0.append(key_cash_t1).reset_index().rename(columns={0: 'cash'})
    key_cash = key_cash.groupby(['code', 'key_year'])['cash'].sum().reset_index()

    return key_cash

def _pivot_key_cash(key_cash):
    '''提取关键久期'''
    data = key_cash.drop_duplicates()
    data['new_col'] = [str(x) + 'Y_Duration' for x in data['key_year']]
    res = data.pivot(values='key_duration', columns='new_col', index='code')
    res['D_DATE'] = key_cash['baseDay'].values[0]

    return res

def _key_duration(wind_code, baseDay):
    '''计算给定日期的关键久期，返回关键期限现金流数据'''
    dq = JYDB_Query()
    tb = dq.sec_query('bond_cash_flow', wind_code)
    tb = self._dealCode(tb)
    dq.close_query()

    key_years = [-1, 0, 1, 3, 5, 10, np.inf]
    delta = 10  # bp
    baseDay = pd.to_datetime(baseDay)

    # 处理含权债的现金流：回售、赎回和调整票面利率
    codes0, codes1 = _idOptionEmbedded(wind_code, baseDay)
    if len(codes0) > 0:
        tb_0 = _dealInterestChg(codes0)
    else:
        tb_0 = pd.DataFrame()
    if len(codes1) > 0:
        tb_1 = _deal2Option(codes1, tb, baseDay)
    else:
        tb_1 = pd.DataFrame()
    tb = tb.loc[~tb['code'].isin(codes0+codes1), :].copy()
    tb = tb.append(tb_0, sort=False)
    tb = tb.append(tb_1, sort=False)

    tb = tb[tb['PAYMENTDATE'] >= baseDay].copy()
    tb['baseDay'] = baseDay
    tb['days'] = tb['PAYMENTDATE'] - baseDay
    tb['year'] = [x.days/365 for x in tb['days']]
    tb['key_year'] = pd.cut(tb['year'], bins=key_years, labels=key_years[:-1]).astype(float)
    tb['key_year_t1'] = [_map_key_year_t1(key_years, x) for x in tb['key_year']]
    tb['cash_t'] = (1 - (tb['year'] - tb['key_year'])/(tb['key_year_t1'] - tb['key_year'])) * tb['CASHFLOW']    # 当期按比例划转的现金流
    tb['cash_t1'] = tb['CASHFLOW'] - tb['cash_t']

    # 按关键久期归总现金流，可能涉及到一个关键期限有多条现金流
    key_cash = _calc_key_cash(tb)
    ytm = dq.bond_yield(wind_code, baseDay.strftime('%Y-%m-%d'))
    ytm = _dealCode(ytm)
    key_cash = pd.merge(key_cash, ytm[['code', 'VPYIELD']], on=['code'], how='left').rename(columns={'VPYIELD': 'yield'})
    key_cash['baseDay'] = baseDay
    if key_cash['yield'].count() == 0:
        print(baseDay)
        return None
    
    key_cash['yield'] = key_cash['yield'].fillna(0)   # 若无YTM则暂取0
    # key_cash = key_cash[(key_cash['key_year'] > 0) & (key_cash['key_year'] < np.inf)].copy()
    key_cash = key_cash[(key_cash['key_year'] < np.inf)].copy()
    
    key_cash["price0"]=[x["cash"]/pow((1+x["yield"]/100),x["key_year"])  for idx,x in key_cash.iterrows()]
    key_cash["price-"]=[x["cash"]/pow((1+(x["yield"]-delta/100)/100),x["key_year"])  for idx,x in key_cash.iterrows()]
    key_cash["price+"]=[x["cash"]/pow((1+(x["yield"]+delta/100)/100),x["key_year"])  for idx,x in key_cash.iterrows()]
    key_cash["key_duration0"]=(key_cash["price-"]-key_cash["price+"])/(2*delta/10000*key_cash["price0"])
    key_cash["weight"] = key_cash.groupby(['code'])['price0'].apply(lambda x: x/x.sum())
    key_cash["key_duration"]=key_cash["key_duration0"]*key_cash["weight"]

    return key_cash

def dealKey_duration(code_list, date_list):
    '''汇总债券关键久期，返回的是关键久期、各期限现金流数据'''
    res_dura_list = []
    res_duraP_list = []
    for date_temp in date_list:
        if len(code_list) > 0:
            key_dura_temp = _key_duration(code_list, date_temp)
            if key_dura_temp is None:
                continue
            dura_pivot_temp = _pivot_key_cash(key_dura_temp)
            res_dura_list.append(key_dura_temp)
            res_duraP_list.append(dura_pivot_temp)
    
    res_dura = pd.DataFrame(columns=key_dura_temp.columns)
    for x in res_dura_list:
        res_dura = res_dura.append(x, sort=False)
    res_dura = res_dura.rename(columns={'baseDay': 'D_DATE'})
    res_duraP = pd.DataFrame(columns=dura_pivot_temp.columns)
    for x in res_duraP_list:
        res_duraP = res_duraP.append(x, sort=False)
    
    return res_duraP, res_dura


if __name__ == '__main__':
    start = time.perf_counter()
    code_list = pd.read_excel(r'C:\Users\wangp\Desktop\bondlist20200908.xlsx')['万德代码'].drop_duplicates().tolist()
    date_list = ['2020-09-08']
    # data0为债券的关键久期，data1为各期限上的现金流数据
    data0, data1 = dealKey_duration(code_list, date_list)
    end = time.perf_counter()
    print('Total Bond: %d'%len(code_list))
    print('Total Time Cost: %.4f (s).'%(end - start))

    writer = pd.ExcelWriter(r'C:\Users\wangp\Desktop\KeyDuration1.xlsx')
    data0.to_excel(writer, sheet_name='KeyDuration')
    data1.to_excel(writer, sheet_name='KeyCash')
    writer.save()