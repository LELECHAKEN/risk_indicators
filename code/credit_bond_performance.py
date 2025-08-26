# @Time : 2021/10/21 12:40 
# @Author : for wangp
# @File : credit_bond_performance.py 
# @Software: PyCharm
import os
import numpy as np
import pandas as pd
from scripts.settings import config
from scripts.db import OracleDB, sqls_config

db = OracleDB(config['data_base']['QUANT']['url'])

def retrieve_bond_data(prod_list, fund_type, start_date, end_date):
    table_name = 'campisi_bond' if fund_type == '公募' else 'campisi_bond_zh'
    prod_str = ','.join('\'' + x + '\'' for x in prod_list)
    q = sqls_config['bond_return_data']['Sql']%(prod_str, start_date, end_date)
    data = db.read_sql(q)
    data = data.sort_values(by=['c_fundname', 'securityname', 'c_date'])

    return data

def retrieve_bond_data_aa(prod_list, fund_type, start_date, end_date):
    table_name = 'aa_allocation_detail'
    prod_str = ','.join('\'' + x + '\'' for x in prod_list)
    q = sqls_config['bond_attribution_aa']['Sql']%(table_name, prod_str, start_date, end_date)
    data = db.read_sql(q)
    data = data.sort_values(by=['c_fundname', 'securityname', 'c_date'])

    return data


def retrieve_port_data(prod_list, start_date, end_date):
    prod_str = ','.join('\'' + x + '\'' for x in prod_list)
    q_return = sqls_config['port_return']['Sql'] % (prod_str, start_date, end_date)
    data_return = db.read_sql(q_return)
    data_return['c_date'] = [x.strftime('%Y-%m-%d') for x in data_return['d_date']]

    q_asset = sqls_config['port_asset']['Sql'] % (prod_str, start_date, end_date)
    data_asset = db.read_sql(q_asset).rename(columns={'d_date': 'c_date'})
    data = pd.merge(data_return, data_asset, on=['c_fundname', 'c_date'], how='left')
    data = data.sort_values(by=['c_fundname', 'c_date'])
    data['net_asset'] = data.groupby(['c_fundname'])['net_asset'].fillna(method='bfill')
    return data

def calc_bond_return(data_bond, data_port):
    data = pd.merge(data_bond, data_port, on=['c_fundname', 'c_date'], how='left')
    data['weight'] = data['mktval_x'] / data['net_asset']
    res_weight = data.groupby(['c_fundname', 'securityname'])['weight'].mean().rename('weight_avg').reset_index()
    res_port_ret = data.groupby(['c_fundname', 'securityname'])['ret'].apply(lambda x: (1+x).product() - 1).rename('port_ret_cum').reset_index()
    # res_return = data.groupby(['c_fundname', 'securityname'])[['r_day', 'attri_day', 'r_hold_ytm', 'r_capital_gain', 'attri_hold_ytm', 'attri_capital_gain']].apply(lambda x: (1+x).product() - 1).reset_index()
    res_return = data.groupby(['c_fundname', 'securityname'])[['r_day', 'attri_day']].apply(lambda x: (1+x).product() - 1).reset_index()
    res_period = data.groupby(['c_fundname', 'securityname'])['c_date'].apply(lambda x: pd.DataFrame([x.min(), x.max(), (pd.to_datetime(x.max()) - pd.to_datetime(x.min())).days], index=['buy_date', 'sell_date', 'hold_days']).T).reset_index().drop(columns='level_2')

    res = pd.merge(res_weight, res_period, on=['c_fundname', 'securityname'], how='left')
    res = pd.merge(res, res_return, on=['c_fundname', 'securityname'], how='left')
    res = pd.merge(res, res_port_ret, on=['c_fundname', 'securityname'], how='left')
    # res = res.reindex(columns=['securityname', 'c_fundname', 'buy_date', 'sell_date', 'hold_days', 'r_day', 'r_hold_ytm', 'r_capital_gain'
    #     ,  'port_ret_cum', 'weight_avg', 'attri_day', 'attri_hold_ytm', 'attri_capital_gain']).sort_values(by=['c_fundname', 'securityname'])
    # res.columns = ['债券简称', '持有组合', '买入日期', '卖出日期', '持有天数', '债券收益率', '其中：持有收益', '其中：资本利得',
    #                '同期组合收益率', '持仓平均权重', '对组合贡献', '其中：持有收益贡献', '其中：资本利得贡献']
    res = res.reindex(columns=['securityname', 'c_fundname', 'buy_date', 'sell_date', 'hold_days', 'r_day',
                               'port_ret_cum', 'weight_avg', 'attri_day']).sort_values(by=['c_fundname', 'securityname'])
    res.columns = ['债券简称', '持有组合', '买入日期', '卖出日期', '持有天数', '债券收益率',
                   '同期组合收益率', '持仓平均权重', '对组合贡献']
    return data, res

def save_data(data, data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    prod_list = data['c_fundname'].unique().tolist()
    for prod in prod_list:
        data[data['c_fundname'] == prod].to_excel(data_path + prod + '.xlsx', index=False)
        print(prod, 'saved.')

    print('complete.')


if __name__ == '__main__':
    data_path = r'E:\RiskQuant\★ 业绩归因\5. 其他需求\20231030-信评\\'
    prods = pd.read_excel(data_path + '账户筛选.xlsx', engine='openpyxl')
    prods_all = prods['组合简称'].tolist()
    prods_mf = prods.loc[prods['组合类型'] == '公募', '组合简称'].tolist()
    prods_smf = prods.loc[prods['组合类型'] == '专户', '组合简称'].tolist()
    start_date = '2023-01-01'
    end_date = '2023-10-27'

    data_bond_1 = retrieve_bond_data_aa(prods_mf, '公募', start_date, end_date)
    data_bond_2 = retrieve_bond_data_aa(prods_smf, '', start_date, end_date)
    data_bond = data_bond_1.append(data_bond_2, sort=False)
    data_port = retrieve_port_data(prods_all, start_date, end_date)
    data, res = calc_bond_return(data_bond, data_port)

    res.to_excel(data_path + 'data.xlsx', index=False)
    save_data(data, data_path + '各组合数据\\')