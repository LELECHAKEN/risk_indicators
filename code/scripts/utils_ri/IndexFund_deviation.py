#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time   : 2022/10/21 13:18
# !@File   : IndexFund_deviation.py
# !@Author : shiyue

import os
import pandas as pd
import numpy as np
from ..db import OracleDB, sqls_config, column
from ..settings import config
from ..utils import logger
from ..db.util import DoesNotExist
from sqlalchemy import exc

def _loadIndexFundsFile(t):
    _index_excel = pd.read_excel(r'\\shaoafile01\RiskManagement\4. 静态风险监控\4.1 每日静态风险监控\报表八：指数基金跟踪偏离度与跟踪误差监控表（每日）.xlsx',
                                      sheet_name=None, engine='openpyxl')
    fund_list = []
    trk_err_a = []
    trk_err_c = []
    dev_daily_a = []
    dev_daily_c = []
    for fund in _index_excel.keys():
        if fund == '报表八':
            continue
        fund_list.append(fund)
        start_row = 5 if 'ETF' in fund else 4
        try:
            temp_df = _index_excel.get(fund).iloc[start_row:, :].dropna(thresh=5, axis=1)
            temp_df.columns = temp_df.loc[start_row, :].tolist()
            temp_df.columns = [x.replace('\n', '') for x in temp_df.columns]
            temp_df = temp_df.loc[start_row + 1:, :].reset_index(drop=True).set_index('日期')

            if pd.to_datetime(t) in temp_df.index:
                if '年化跟踪误差' in temp_df.columns:
                    trk_err_a.append(temp_df.loc[pd.to_datetime(t), '年化跟踪误差'])
                    trk_err_c.append(np.nan)
                    dev_daily_a.append(temp_df.loc[pd.to_datetime(t), '日均跟踪偏离度绝对值'])
                    dev_daily_c.append(np.nan)
                else:
                    trk_err_a.append(temp_df.loc[pd.to_datetime(t), 'A份额年化跟踪误差'])
                    trk_err_c.append(temp_df.loc[pd.to_datetime(t), 'C份额年化跟踪误差'])
                    dev_daily_a.append(temp_df.loc[pd.to_datetime(t), 'A份额日均跟踪偏离度绝对值'])
                    dev_daily_c.append(temp_df.loc[pd.to_datetime(t), 'C份额日均跟踪偏离度绝对值'])
            else:
                continue
        except:
            print('Error! ', fund)
            continue

    res = pd.DataFrame([fund_list, trk_err_a, trk_err_c, dev_daily_a, dev_daily_c],
                       index=['C_FUNDNAME', 'A份额年化跟踪误差', 'C份额年化跟踪误差', 'A份额日均跟踪偏离度绝对值', 'C份额日均跟踪偏离度绝对值']).T

    return res


class IndexFundIndicators():

    def __init__(self, t):
        self.basedate = t
        self.db_risk = OracleDB(config['data_base']['QUANT']['url'])
        self.db_wind = OracleDB(config['data_base']['WIND']['url'])
        self.db_jy = OracleDB(config['data_base']['JY']['url'])

        q_date = sqls_config["past_n_tradedays"]["sql"]
        monitor_bgdate = self.db_risk.read_sql(q_date.format(t=self.basedate, n=30))['c_date'].iloc[-1]
        q = sqls_config['index_fund_basic_info']['sql']
        self.basic_info = self.db_risk.read_sql(q.format(bg_date=monitor_bgdate, t=t))
        self.basic_info['deposit_rate'] = self.basic_info['deposit_rate'].fillna(0)
        self.b_index = self.basic_info[self.basic_info['type_l2'] == '债券指数']['portfolio_code'].to_list()
        self.e_index = self.basic_info[self.basic_info['type_l2'] == '股票指数']['portfolio_code'].to_list()

    def index_fund_deviation(self):

        # 债指产品走投决会监控口径, 股指走合同口径
        res_b = self.compute_track_deviation(period=20, index_type='wealth', fund_codes=self.b_index)
        res_q = self.compute_track_deviation(period=20, index_type='cont', fund_codes=self.e_index)
        res1 = pd.concat([res_b, res_q])

        self.delete_table(table='rc_mr_index_fund_monitor', t=self.basedate, schema='quant')
        self.insert_table(table='rc_mr_index_fund_monitor', data=res1, t=self.basedate, schema='quant')

        # 静态指标检测口径
        # res2 = self.compute_track_deviation(period=90, index_type='cont')
        # self.delete_table(table='index_fund_stats_dev', t=self.basedate, schema='quant')
        # self.insert_table(table='index_fund_stats_dev', data=res2, t=self.basedate, schema='quant')

        return res1

    def insert_table(self, table, data, t, schema, if_exists='append'):
        self.db_risk.insert_dataframe(table=table.lower(), data=data, schema=schema, if_exists=if_exists)
        logger.info('%s数据插入成功，table: %s inserted to database.' % (t, table))

    def delete_table(self, table, t, schema, column_name='c_date'):
        condition = column(column_name) == t
        try:
            self.db_risk.delete(table.lower(), condition, schema)
            logger.info('%s删除成功，table: %s deleted from database.' % (t, table))
        except (DoesNotExist, exc.NoSuchTableError):
            logger.warning('%s删除失败，table: %s data not found.' % (t, table))
            pass

    def bm_index_close(self, code_list: list, t0: str, t1: str):
        '''
        获取指数的行情序列
        :param code_list: 指数代码列表
        :param t0: str, yyyy-mm-dd 起始日期
        :param t1: str, yyyy-mm-dd 结束日期
        :return:
        '''
        code_dict = dict(zip([i[:i.find(".")] for i in code_list], code_list))
        sql_wind = sqls_config["index_nav_wind"]["sql"]
        nav_wind = self.db_wind.read_sql(sql_wind.format(code=tuple(code_list), t0=t0.replace("-", ""), t1=t1.replace("-", "")))

        sql_jy = sqls_config["index_nav_jy"]["sql"]
        nav_jy = self.db_jy.read_sql(sql_jy.format(code=tuple(code_dict.keys()), t0=t0, t1=t1))
        nav_jy["wind_code"] = nav_jy["wind_code"].map(code_dict)

        sql_csi = sqls_config["index_nav_csi"]["sql"]
        nav_csi = self.db_jy.read_sql(sql_csi.format(code=tuple(code_dict.keys()), t0=t0, t1=t1))
        nav_csi["wind_code"] = nav_csi["wind_code"].map(code_dict)

        sql_hk = sqls_config["index_nav_hk"]["sql"]
        nav_hk = self.db_jy.read_sql(sql_hk.format(code=tuple(code_dict.keys()), t0=t0, t1=t1))
        nav_hk["wind_code"] = nav_hk["wind_code"].map(code_dict)

        nav_df = pd.pivot_table(data=pd.concat([nav_wind, nav_jy, nav_csi, nav_hk]), values="nav", index="c_date",
                                columns="wind_code").fillna(method="ffill").sort_index()
        return nav_df

    def index_fund_nav(self, code_list: list, t0: str, t1: str):
        sql_nav = sqls_config["fund_nav_wind"]["sql"]
        nav_raw = self.db_wind.read_sql(sql_nav.format(code=tuple(code_list), t0=t0.replace("-", ""), t1=t1.replace("-", "")))
        nav_df = pd.pivot_table(data=nav_raw, values="nav", index="c_date", columns="wind_code").sort_index().fillna(method="ffill")
        return nav_df

    def compute_track_deviation(self, period: int = 20, index_type: str = 'wealth', fund_codes: list = None):
        fund_info = self.basic_info.copy() if fund_codes is None else self.basic_info[self.basic_info['portfolio_code'].isin(fund_codes)]

        # 年化跟踪偏离度的计算口径（取决于period)
        past_tds = self.db_risk.read_sql(sqls_config["past_n_tradedays"]["sql"].format(t=self.basedate, n=period+1))
        t0 = past_tds["c_date"].iloc[-1]
        t1 = past_tds["c_date"].iloc[0]

        # 获取基金净值
        fund_codes = fund_info["wind_code_a"].to_list() + fund_info["wind_code_c"].dropna().to_list()
        funds_nav = self.index_fund_nav(fund_codes, t0, t1)
        logger.info("已获取指数基金净值")

        # 获取相应指数的收盘价
        index_col = 'wealth_index_code' if index_type == 'wealth' else 'cont_index_code'
        bms_nav = self.bm_index_close(fund_info[index_col].to_list(), t0, t1)
        logger.info("已获取指数收益收盘价")

        # 计算跟踪偏离度
        results = pd.DataFrame()
        for idx, info in fund_info.iterrows():

            nav_df = pd.merge(funds_nav[info['wind_code_a']].to_frame(name="a_nav"),
                              bms_nav[info[index_col]].to_frame(name="bm_nav"),
                              how="left", left_index=True, right_index=True).fillna(method="ffill").dropna()

            # 计算基准指数收益
            nav_df['bm_chg'] = nav_df['bm_nav'].pct_change()
            if index_type != 'wealth' and info['cont_index_weight'] < 1:
                nav_df['bm_chg'] = nav_df['bm_chg'] * info['cont_index_weight'] + info['deposit_rate']/365 * (1 - info['cont_index_weight'])

            res = self._track_deviation(nav_df, 'a')

            if info['wind_code_c'] is not None:
                res = pd.merge(res, funds_nav[info['wind_code_a']].to_frame(name="c_nav"), how="left",
                               left_index=True, right_index=True).fillna(method="ffill").dropna()
                res = self._track_deviation(res, 'c')

            res['portfolio_code'] = info['portfolio_code']
            res['bm_index_code'] = info[index_col]
            res['fund_name'] = info['c_fundname']
            results = pd.concat([results, res[res.index == t1].copy()], ignore_index=True)

        results['c_date'] = self.basedate
        cols = ['c_date', 'portfolio_code', 'fund_name', 'bm_index_code', 'bm_chg',
                'a_nav', 'a_chg', 'a_dev', 'a_dev_abs', 'a_avg_dev_abs', 'a_ann_dev',
                'c_nav', 'c_chg', 'c_dev', 'c_dev_abs', 'c_avg_dev_abs', 'c_ann_dev']

        return results.reindex(columns=cols)

    def _track_deviation(self, nav_df: pd.DataFrame, share_type: str):
        res = nav_df.copy()
        res[share_type + '_chg'] = res[share_type + '_nav'].pct_change()
        res[share_type + '_dev'] = res[share_type + '_chg'] - res["bm_chg"]
        res[share_type + '_dev_abs'] = abs(res[share_type + '_dev'])
        res[share_type + '_avg_dev_abs'] = abs(np.average(res[share_type + '_dev'].dropna()))
        res[share_type + '_ann_dev'] = np.std(res[share_type + '_dev'].dropna(), ddof=1) * np.sqrt(250)
        return res


