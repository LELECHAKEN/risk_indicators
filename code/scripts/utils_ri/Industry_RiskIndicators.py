#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time   : 2022/11/24 10:53
# !@File   : Industry_RiskIndicators.py
# !@Author : shiyue


import os
import time
import warnings
import datetime
import math
import numpy as np
import pandas as pd
import cx_Oracle as cx
from sqlalchemy import exc
from tkinter import messagebox

from scripts.settings import config
from scripts.utils.catch_error_msg import show_catch_message
from scripts.utils.log_utils import logger
from scripts.db import OracleDB, column, sqls_config
from scripts.db.db_utils import JYDB_Query
from scripts.db.util import DoesNotExist
from scripts.settings import config, DIR_OF_MAIN_PROG


class industryRiskIndicators:

    def __init__(self, t):
        self.basedate = t
        self._connectingRiskDB()
        self._connectingWINDDB()

    def _connectingRiskDB(self):
        self.db_risk = OracleDB(config['data_base']['QUANT']['url'])

    def _connectingWINDDB(self):
        self.db_wind = OracleDB(config['data_base']['WIND']['url'])

    def insert_table(self, table, data, schema, if_exists='append'):
        if data.shape[0] == 0:
            return
        if 'db_risk' not in dir(self):
            self._connectingRiskDB()
        if 'C_DATE' in data.columns and type(data['C_DATE'].iloc[0]) == pd.Timestamp:
            data['C_DATE'] = [x.strftime('%Y-%m-%d') for x in data['C_DATE']]
        data['insert_time'] = datetime.datetime.now()
        self.db_risk.insert_dataframe(table=table.lower(), data=data, schema=schema, if_exists=if_exists)
        'quant'
        logger.info('%s数据插入成功，table: %s inserted to database.'%(self.basedate, table))

    def delete_table(self, table, condition, schema):
        if 'db_risk' not in dir(self):
            self._connectingRiskDB()
        condition = condition
        try:
            self.db_risk.delete(table.lower(), condition, schema)
            logger.info('%s删除成功，table: %s deleted from database.' % (t, table))
        except (DoesNotExist, exc.NoSuchTableError):
            logger.warning('%s删除失败，table: %s data not found.' % (t, table))
            pass

    def _get_db_data(self, sql_raw: str, code_list: list, **kwargs):
        '''
        用于查询多条代码导致sql语句过长的的情况下
        :param sql_raw: str, sql语句
        :param code_list: list, 代码列表
        :param kwargs: sql语句中需要传入的其他参数
        :return: pd.DataFrame
        '''
        if len(code_list) > 0:
            loop_length = 800  # 循环限制输入SQL的数量
            sql_loop = math.ceil(len(code_list) / loop_length)  # sql里输入的字符串长度有上限，此处规定一次只能查询800个万德代码
            data_df = pd.DataFrame()
            for loop in range(sql_loop):
                code_part = code_list[loop_length * loop: loop_length * (loop + 1)] \
                    if loop < sql_loop - 1 else code_list[loop_length * loop:len(code_list)]
                code_part = tuple(code_part) if len(code_part) > 1 else '(\'' + code_part[0] + '\')'
                ssql = sql_raw.format(windcode=code_part, **kwargs)
                df = self.db_wind.read_sql(sql=ssql)
                data_df = pd.concat([data_df, df])
            return data_df
        else:
            logger.warning("代码列表为空")


    def add_exchange_suffix(self, secucode, fundname):
        return secucode + ".OF" if "LOF" not in fundname else secucode + ".SZ"

    def calc_drawdown_thisyear(self, type_list: list, period="今年以来"):
        logger.info("%s 开始计算名单今年以来的中位回撤" % self.basedate)
        dd_list, maxdd_list= [], []
        dd_df = pd.DataFrame()
        for fund_type in type_list:
            sql_raw = sqls_config["fund_ranking_list"]["sql"]
            fund_df = self.db_risk.read_sql(sql_raw.format(t=self.basedate, period=period, fund_type=fund_type))
            fund_df["windcode"] = list(map(self.add_exchange_suffix, fund_df["secucode"].to_list(), fund_df["fundname"].to_list()))
            code_list = fund_df["windcode"].to_list()
            # 计算当前回撤
            nav_sql = sqls_config["industry_fund_nav"]["sql"]
            nav_df = self._get_db_data(sql_raw=nav_sql, code_list=code_list, t0=self.basedate[:4] + "0101",
                                       t=self.basedate.replace("-", ""))
            nav_df["crt_dd"] = nav_df["crt_nav"] / nav_df["max_nav"] - 1
            dd_list.append(nav_df["crt_dd"].median())
            # 计算最大回撤
            maxdd_sql = sqls_config["mutual_fund_maxdd_thisyear"]["sql"]
            maxdd_df = self._get_db_data(sql_raw=maxdd_sql, code_list=code_list, t=self.basedate.replace("-", ""))
            maxdd_list.append(maxdd_df["max_dd"].median())
            # 合并数据
            data_df = pd.merge(fund_df, nav_df[["windcode", "crt_dd"]], how='left', on="windcode")
            data_df = pd.merge(data_df, maxdd_df, how='left', on="windcode")
            data_df[["fund_type", "c_date", "period"]] = [fund_type, self.basedate, period]
            dd_df = pd.concat([dd_df, data_df])
            logger.info("%s is done." % fund_type)

        condition = (column("c_date") == self.basedate) & (column("period") == period)
        self.delete_table("fr_ranklist_dd", condition, 'quant')
        self.insert_table("fr_ranklist_dd", dd_df, 'quant')

        columns = ["c_date", "fund_type", "period", "crt_dd", "max_dd"]
        result = [[self.basedate]*len(type_list), type_list, [period]*len(type_list), dd_list, maxdd_list]
        result = pd.DataFrame(dict(zip(columns, result)))

        condition = (column("c_date") == self.basedate) & (column("period") == period)
        self.delete_table("fr_fundtype_dd", condition, 'quant')
        self.insert_table("fr_fundtype_dd", result, 'quant')

    def calc_maxdd_quarter(self, type_list: list, period="过去3个月"):
        logger.info("%s 开始计算名单过去3个月的中位回撤" % self.basedate)
        maxdd_qtr_list = []
        dd_df = pd.DataFrame()
        for fund_type in type_list:
            sql_raw = sqls_config["fund_ranking_list"]["sql"]
            fund_df = self.db_risk.read_sql(sql_raw.format(t=self.basedate, period=period, fund_type=fund_type))
            fund_df["windcode"] = list(map(self.add_exchange_suffix, fund_df["secucode"].to_list(), fund_df["fundname"].to_list()))
            code_list = fund_df["windcode"].to_list()
            maxdd_sql = sqls_config["mutual_fund_maxdd_quater"]["sql"]
            maxdd_df = self._get_db_data(sql_raw=maxdd_sql, code_list=code_list, t=self.basedate.replace("-", ""))
            maxdd_qtr_list.append(maxdd_df["max_dd"].median())
            data_df = pd.merge(fund_df, maxdd_df, on="windcode")
            data_df[["fund_type", "c_date", "period"]] = [fund_type, self.basedate, period]
            dd_df = pd.concat([dd_df, data_df])
            logger.info("%s is done." % fund_type)

        condition = (column("c_date") == self.basedate) & (column("period") == period)
        self.delete_table("fr_ranklist_dd", condition, 'quant')
        self.insert_table("fr_ranklist_dd", dd_df, 'quant')

        columns = ["c_date", "fund_type", "period", "crt_dd", "max_dd"]
        result = [[self.basedate] * len(type_list), type_list, [period]*len(type_list), [None]*len(type_list), maxdd_qtr_list]
        result = pd.DataFrame(dict(zip(columns, result)))

        condition = (column("c_date") == self.basedate) & (column("period") == period)
        self.delete_table("fr_fundtype_dd", condition, 'quant')
        self.insert_table("fr_fundtype_dd", result, 'quant')



# t = "2022-10-23"
# ird = industryRiskIndicators(t)
# ird.calc_drawdown(type_list)

# db_risk = OracleDB(config['data_base']['QUANT']['url'])
# type_sql = sqls_config["fund_types_all"]["sql"]
# type_list = db_risk.read_sql(type_sql)
# type_list = type_list["rankinglist"].to_list()
type_list = ['信用债', '股3债7', '商行债', '3-5年政金债指数',
             '欣益', '短债', '货基90天', '1-3年政金债指数', '纯利率', '股4债6', '股2债8', '1-5年政金债指数',
             '股1债9', '中短债', '标准债券', '信用一年定开']
# td_sql = sqls_config["trade_day_df"]["sql"]
# td_df = db_risk.read_sql(td_sql)
# dates = td_df["c_date"].to_list()

type_list= ['股3债7', '股2债8']

for t in ["2022-12-31"]:
    ird = industryRiskIndicators(t)
    # ird.calc_drawdown_thisyear(type_list)
    ird.calc_maxdd_quarter(type_list)


