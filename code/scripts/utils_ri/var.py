#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time   : 2022/9/2 13:19
# !@File   : var.py
# !@Author : shiyue

import os
import pandas as pd
import numpy as np
from ..db import OracleDB, sqls_config, column
from ..settings import DIR_OF_MAIN_PROG, config
from ..utils import logger
from ..db.util import DoesNotExist
from sqlalchemy import exc


def _delete_table(table, t, schema, column_name='C_DATE'):
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    condition = column(column_name) == t
    try:
        db_risk.delete(table.lower(), condition, schema)
        logger.info('%s删除成功，table: %s deleted from database.' % (t, table))
    except (DoesNotExist, exc.NoSuchTableError):
        logger.warning('%s删除失败，table: %s data not found.' % (t, table))
        pass


def _insert_table(table, data, t, schema, if_exists='append'):
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    db_risk.insert_dataframe(table=table.lower(), data=data, schema=schema, if_exists=if_exists)
    logger.info('%s数据插入成功，table: %s inserted to database.' % (t, table))


def calc_var_parameter(date: str):
    '''

    :param date: str, "YYYY-MM-DD"
    '''

    # --- 计算收益率
    db_jy = OracleDB(config['data_base']['JY']['url'])
    yield_df = db_jy.read_sql(sqls_config['var_yield']['sql'].format(t=date))
    yield_df["yield"] = yield_df["yield"] * 100

    yield_map = pd.read_excel(os.path.join(config['shared_drive_data']['risk_indicator']['path'], 'VAR模型映射表.xlsx'), sheet_name="收益率", engine='openpyxl').fillna(method="ffill")
    curve_dict = {66: "信用AA+", 67: "信用AA", 70: "信用AAA", 195: "利率债"}
    period_dict = {0.25: "3个月", 1.0: "1年", 3.0: "3年", 5.0: "5年", 10.0: "10年"}
    map_dict = {1: "25分位以下", 2: "25~50分位", 3: "50~75分位", 4: "75分位以上"}

    def _calc_yield_q(curve: int, period: float, crt_y: float):
        yields = yield_map.loc[yield_map["曲线品种"] == curve_dict[curve], period_dict[period]].to_list()
        yields.append(crt_y)
        return map_dict[sorted(yields).index(crt_y)]

    c_list = yield_df["curvecode"].to_list()
    p_list = yield_df["yearstomaturity"].to_list()
    y_list = yield_df["yield"].to_list()
    q_list = [_calc_yield_q(c_list[i], p_list[i], y_list[i]) for i in range(len(c_list))]

    col_names = ['aa_3m', 'aa_1y', 'aa_3y', 'aa_5y', 'aaminus_3m', 'aaminus_1y', 'aaminus_3y', 'aaminus_5y',
                 'aaaminus_3m', 'aaaminus_1y', 'aaaminus_3y', 'aaaminus_5y', 'ir_3m', 'ir_1y', 'ir_3y', 'ir_5y', 'ir_10y']
    yield_names = ["c_date"] + ["yield_" + i for i in col_names]
    yield_qtl_names = ["c_date"] + ["yield_qtl_" + i for i in col_names]
    yield_data = pd.DataFrame(dict(zip(yield_names, [date] + y_list)), index=[0])
    yield_qtl_data = pd.DataFrame(dict(zip(yield_qtl_names, [date] + q_list)), index=[0])

    # -------计算斜率
    chn_names = ["信用债AA+", "信用债AA", "信用债AAA", "利率债"]
    slope_map = pd.read_excel(os.path.join(config['shared_drive_data']['risk_indicator']['path'], 'VAR模型映射表.xlsx'), sheet_name="斜率", index_col=0, engine='openpyxl')
    slope_dict = dict(zip([1, 2, 3], list(slope_map.index)))

    slope_list = [y_list[3]-y_list[1], y_list[7]-y_list[5], y_list[11]-y_list[9], y_list[16]-y_list[13]]
    slope_qtl_list = []
    for i in range(len(chn_names)):
        data = slope_map[chn_names[i]].to_list()
        data.append(slope_list[i])
        slope_qtl_list.append(slope_dict[sorted(data).index(slope_list[i])])

    slope_names = ["c_date"] + ['slope_' + i for i in ['aaplus', 'aa', 'aaa', 'ir']]
    slope_qtl_names = ["c_date"] + ['slope_qtl_' + i for i in ['aaplus', 'aa', 'aaa', 'ir']]
    slope_data = pd.DataFrame(dict(zip(slope_names, [date] + slope_list)), index=[0])
    slope_qtl_data = pd.DataFrame(dict(zip(slope_qtl_names, [date] + slope_qtl_list)), index=[0])

    # ----导入数据库
    _delete_table("rc_mr_var_yield", date, 'quant')
    _insert_table("rc_mr_var_yield", yield_data, date, 'quant')
    _delete_table("rc_mr_var_yield_qtl", date, 'quant')
    _insert_table("rc_mr_var_yield_qtl", yield_qtl_data, date, 'quant')
    _delete_table("rc_mr_var_slope", date, 'quant')
    _insert_table("rc_mr_var_slope", slope_data, date, 'quant')
    _delete_table("rc_mr_var_slope_qtl", date, 'quant')
    _insert_table("rc_mr_var_slope_qtl", slope_qtl_data, date, 'quant')


def get_tb_params_ir():
    '''
    修改 收益率分位数 表格的格式
    :return:
    '''
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    yield_qtl = db_risk.read_sql(sqls_config['var_yield_qtl']['sql'])
    yield_qtl.set_index(keys='c_date', drop=True, inplace=True)
    yield_qtl.index = pd.to_datetime(yield_qtl.index)
    yield_qtl.columns = ['利率债：3个月', '利率债：1年', '利率债：3年', '利率债：5年', '利率债：10年', '信用债AAA：3个月',
                         '信用债AAA：1年', '信用债AAA：3年', '信用债AAA：5年', '信用债AA+：3个月', '信用债AA+：1年',
                         '信用债AA+：3年', '信用债AA+：5年', '信用债AA：3个月', '信用债AA：1年', '信用债AA：3年',
                         '信用债AA：5年']
    return yield_qtl


def get_tb_params_slope():
    '''
    修改 斜率分位数 表格的格式
    :return:
    '''
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    slope_qtl = db_risk.read_sql(sqls_config['var_slope_qtl']['sql'])
    slope_qtl.set_index(keys='c_date', drop=True, inplace=True)
    slope_qtl.index = pd.to_datetime(slope_qtl.index)
    slope_qtl.columns = ['利率债', '信用债AAA', '信用债AA+', '信用债AA']
    return slope_qtl


# 使用例子
# t = "2022-10-18"
# calc_var_parameter(t)
# yield_qtl = get_tb_params_ir()
# slope_qtl = get_tb_params_slope()




