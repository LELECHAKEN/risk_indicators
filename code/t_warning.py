#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time   : 2022/11/14 10:48
# !@File   : t_warning.py
# !@Author : shiyue

import os

import numpy as np
import pandas as pd
from WindPy import w
from datetime import datetime
from scripts.settings import config
from scripts.db import OracleDB, column, sqls_config
from scripts.utils_ri.RiskIndicators import RiskIndicators
from scripts.risk_warning import PortfolioWarning


# db_risk = OracleDB(config['data_base']['QUANT']['url'])
# db_wind = OracleDB(config['data_base']['WIND']['url'])
# bond_index_info = db_risk.read_sql(sqls_config["bond_index_fund"]["sql"])
# _dd_map = pd.read_excel(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\PortfolioType\回撤阈值.xlsx', engine='openpyxl').rename(columns={'基金名称': 'C_FUNDNAME', '回撤阈值': 'dd_threshold'})

t = "2023-03-09"

pw = PortfolioWarning(t=t, save_path="")
pw.drawdown_warning()




# sql_dd = sqls_config['drawdown_mkt']['Sql']%t
# dd_port = db_risk.read_sql(sql_dd)
# dd_port['d_date'] = pd.to_datetime(dd_port['d_date'])
# dd_port.columns = [x.upper() if x.islower() else x for x in dd_port.columns]
# dd_all = pd.merge(dd_port, _dd_map, on='C_FUNDNAME', how='left').dropna(subset=['dd_threshold'])
#
# for fund_name in list(bond_index_info.index):
#     dd_all.loc[dd_all["C_FUNDNAME"] == fund_name, "dd_threshold"] -= _bch_dd(fund_name, t)
# dd_all['alert'] = dd_all['DrawDown'] * (-1) >= dd_all['dd_threshold']











