'''
@Description: 产品的风险指标文件夹拆分至各基金经理
@Author: Wangp
@Date: 2020-05-12 16:55:25
LastEditTime: 2021-05-17 13:59:56
LastEditors: Wangp
'''

import pandas as pd
import os
import shutil
from datetime import datetime
from scripts.db import OracleDB, sqls_config
from scripts.settings import config, DIR_OF_MAIN_PROG


db_risk = OracleDB(config['data_base']['QUANT']['url'])

def get_latest_tradeday(t):
    '''取最近的一个交易日日期'''
    q = sqls_config['check_latest_day']['sql'] % t
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    latest_tradeday = db_risk.read_sql(q)['c_date'].iloc[0]
    return latest_tradeday

def check_if_tradeday(t):
    '''检查t日是否为交易日，非交易日则不运行模型'''
    latest_tradeday = get_latest_tradeday(t)
    trade_status = latest_tradeday == t
    if not trade_status:
        exit(0)


today = (datetime.today() - pd.Timedelta(1, 'd')).strftime('%Y-%m-%d')
t = get_latest_tradeday(today)
print(t)

# step1: 按照基金经理拆分产品文件
file_path = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\日频数据\%s\\'%t.replace('-', '')
data_prod = pd.read_excel(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\PortfolioType\产品管理类型.xlsx', engine='openpyxl')
data_coding = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl')

data_m = pd.merge(data_prod, data_coding.drop(columns=['产品类型']), left_on='基金全称', right_on='产品名称', how='left')
data_n = data_m[~(data_m['到期日期'] <= datetime.strptime(t, '%Y-%m-%d'))].copy()
fundManager = data_n['基金经理'].dropna().unique().tolist()

# 基助名单
fmAssistant = db_risk.read_sql(sqls_config['mgr_assistant_list']['Sql'])['manager_name'].to_list()
data_fmAss = db_risk.read_sql(sqls_config['mgr_assistant_fund']['Sql'].format(t=t))

# 判断t日是否有风险指标的文件夹
if not os.path.exists(file_path):
    exit(0)


for fm_ass in fmAssistant:
    prods = data_fmAss[data_fmAss['manager_assistant'].str.contains(fm_ass)]['c_fundname'].to_list()
    if len(prods) == 0:
        continue
    fm_ass_path = file_path + fm_ass + '\\'
    if not os.path.exists(fm_ass_path):
        os.mkdir(fm_ass_path)
    for prod in prods:
        file_name = prod + '.xlsx'
        if os.path.exists(file_path + file_name):
            shutil.copy(file_path + file_name, fm_ass_path)

for fm in fundManager:
    prods = data_n.loc[data_n['基金经理'] == fm, '估值系统简称'].dropna().tolist()
    fm_list = fm.replace('，', ',').replace('、', ',').split(',')
    cnt = len(fm_list)
    for prod in prods:
        file_name = prod + '.xlsx'
        for i in range(cnt):
            fm_path = file_path + fm_list[i] + '\\'
            if not os.path.exists(fm_path):
                os.mkdir(fm_path)
            if os.path.exists(file_path + file_name):
                if i < cnt - 1:
                    shutil.copy(file_path + file_name, fm_path)
                else:
                    try:
                        if os.path.exists(fm_path + file_name):
                            os.remove(fm_path + file_name)
                        shutil.move(file_path + file_name, fm_path)
                    except Exception as e:
                        continue

print('Split done.')


# step2: 更新云文档
path_share = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\日频数据\%s' % t.replace('-', '')
path_cloud = r'E:\ShareCache\SHAShare\MWFiles\风险管理部 - 组合风险指标'
folders = os.listdir(path_share)

for folder in folders:
    manager_i = os.path.join(path_share, folder)
    manager_i_cloud = os.path.join(path_cloud, folder)
    # # 用于更新年度风险指标
    # manager_i_cloud_year = os.path.join(path_cloud, folder, '20231231')
    if os.path.exists(manager_i):
        if os.path.isdir(manager_i):
            files = os.listdir(manager_i)
            for file in files:
                file_share = os.path.join(manager_i, file)
                if os.path.exists(manager_i_cloud):
                    shutil.copy(file_share, manager_i_cloud)
                    # # 用于更新年度风险指标
                    # if not os.path.exists(manager_i_cloud_year):
                    #     os.mkdir(manager_i_cloud_year)
                    # shutil.copy(file_share, manager_i_cloud_year)


crt_folder = os.path.join(path_share, t.replace("-", ""))
file_manager_dict = {"风险管理部-组合风险指标-固收.xlsx": "吴玮", t.replace("-", "") + "_特殊产品关键久期.xlsx": "吴玮",
                     "风险管理部-组合风险指标-信用.xlsx": "杨凡颖"}

for file_name in list(file_manager_dict.keys()):
    file_path = os.path.join(crt_folder, file_name)
    if os.path.exists(file_path):
        shutil.copy(file_path, os.path.join(path_cloud, file_manager_dict[file_name]))

print('Transfer done.')