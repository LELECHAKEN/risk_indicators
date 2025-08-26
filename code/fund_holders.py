# @Time : 2021/9/17 12:18 
# @Author : for wangp
# @File : fund_holders.py 
# @Software: PyCharm
import os
import datetime
import numpy as np
import pandas as pd

from scripts.settings import config
from scripts.utils.log_utils import logger
from scripts.db import OracleDB, column, sqls_config
from scripts.db.db_utils import JYDB_Query
from scripts.db.util import DoesNotExist
from scripts.settings import config, DIR_OF_MAIN_PROG

db_risk = OracleDB(config['data_base']['QUANT']['url'])
folder_path = r'E:\StressTest\StressTest-MutualFund\20231229\底稿'

file_name = '[%s]持有人结构.xlsx'
# date_list = [x for x in os.listdir(folder_path) if x.isdigit()]
date_list = ['2023-12-29']
for t in date_list:
    file_t = file_name % pd.to_datetime(t).strftime('%Y-%m-%d')
    file_t_path = os.path.join(folder_path, file_t)
    if os.path.exists(file_t_path):
        data = pd.read_excel(file_t_path, engine='openpyxl')
        data = data.rename(columns={'基金份额（亿份）': 'total_shares', '个人投资者持有比例': 'individual_shares', '机构投资者持有比例': 'institution_shares'})
        data = data.dropna(subset=['total_shares'])
        data['institution_shares'] = data['institution_shares'].fillna(0)
        data.loc[data['individual_shares'].isna(), 'individual_shares'] = 1 - data['institution_shares'].fillna(0)
        data['D_DATE'] = pd.to_datetime(t).strftime('%Y-%m-%d')
        data['insert_time'] = datetime.datetime.now()
        db_risk.insert_dataframe('rc_fundholders', data, 'quant', 'append')
        print(t, 'done.')