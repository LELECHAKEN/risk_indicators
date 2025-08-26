'''
Description: 
Author: Wangp
Date: 2021-06-02 17:32:55
LastEditTime: 2021-06-02 17:42:29
LastEditors: Wangp
'''
import pandas as pd
from ConnectingDatabase import Data_Query


yield_map = pd.read_excel('yield_map.xlsx')
bond_holdings = pd.DataFrame(columns=['D_DATE', 'code', 'WINDL2TYPE', 'MUNICIPALBOND', 'RATE_LATESTMIR_CNBD', 'ENDDATE', 'YEARSTOMATURITY', 'YIELD_CNBD', 'PTMYEAR'])

# 获取基础Benchmark
bond_holdings, benchmark_gk = retrieveBenchmark_gk(bond_holdings)    # 国开债曲线
# 匹配对应评级的曲线收益率，数据来源：JY
bond_holdings['WINDL2TYPE_curve'] = bond_holdings['WINDL2TYPE'].copy()
bond_holdings.loc[bond_holdings['MUNICIPALBOND'] == '是', 'WINDL2TYPE_curve'] = '城投债'        # 城投债优先
bond_holdings = pd.merge(bond_holdings, yield_map, on=['WINDL2TYPE_curve', 'MUNICIPALBOND', 'RATE_LATESTMIR_CNBD'], how='left')
bond_holdings = bond_holdings.drop(columns=['WINDL2TYPE_curve'])  
# 获取评级benchmark
bond_holdings1, benchmark_rating = retrieveBenchmark_Rating(bond_holdings)



def retrieveBenchmark_gk(data=''):
    if type(data) == str:
        bond_holdings = self.bond_holdings.copy()
    else:
        bond_holdings = data.copy()

    dq = Data_Query()

    date_list = bond_holdings['D_DATE'].unique()
    benchmark = pd.DataFrame(columns=['ENDDATE', 'YEARSTOMATURITY', 'YIELD'])
    for temp_date in date_list:
        temp_date = pd.to_datetime(temp_date).strftime('%Y-%m-%d')
        temp_b = dq.curve_yield_all(curvecode='195', t0=temp_date)    # 步长只有0.1, 195为中债国开债收益率曲线
        benchmark = benchmark.append(temp_b, sort=False)
    benchmark['YIELD'] = benchmark['YIELD']*100
    benchmark.columns = ['ENDDATE_gk', 'YEARSTOMATURITY_gk', 'benchmark_gk']

    bond_holdings['剩余期限'] = [round(x, 1) for x in bond_holdings['PTMYEAR'].fillna(0)]    # 将取不出剩余期限的PTM取为0
    bond_holdings1 = pd.merge(bond_holdings, benchmark, left_on=['D_DATE', '剩余期限'], right_on=['ENDDATE_gk', 'YEARSTOMATURITY_gk'], how='left')
    bond_holdings1['spread_gk'] = bond_holdings1['YIELD_CNBD'] - bond_holdings1['benchmark_gk']

    return bond_holdings1, benchmark


# 获取聚源对应隐含评级、对应剩余期限的收益率
def retrieveBenchmark_Rating(data=''):
    if type(data) == str:
        bond_holdings = self.bond_holdings.copy()
    else:
        bond_holdings = data.copy()
    
    dq = Data_Query()

    startDate = bond_holdings['D_DATE'].min().strftime('%Y-%m-%d')
    endDate = bond_holdings['D_DATE'].max().strftime('%Y-%m-%d')
    curve_list = bond_holdings['曲线代码'].dropna().drop_duplicates().astype(str).tolist()
    benchmark = dq.curve_yield_interval(curve_list, startDate, endDate)
    benchmark['YIELD'] = benchmark['YIELD']*100

    bond_holdings['剩余期限'] = [round(x, 1) for x in bond_holdings['PTMYEAR'].fillna(0)]
    bond_holdings1 = pd.merge(bond_holdings.drop(columns=['ENDDATE', 'YEARSTOMATURITY']), benchmark, left_on=['D_DATE', '曲线代码', '剩余期限'], right_on=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY'], how='left')
    bond_holdings1 = bond_holdings1.rename(columns={'YIELD': 'benchmark_rating'})
    bond_holdings1['spread_rating'] = bond_holdings1['YIELD_CNBD'] - bond_holdings1['benchmark_rating']

    return bond_holdings1, benchmark  