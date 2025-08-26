#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time   : 2023/9/15 14:57
# !@File   : asset_type.py
# !@Author : shiyue

import pandas as pd
import numpy as np
from .db import OracleDB, sqls_config
from scripts.db.db_utils import JYDB_Query, WINDDB_Query
from .utils_ri import RiskIndicators
from .settings import config, DIR_OF_MAIN_PROG
from .utils.log_utils import logger
from datetime import datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta


def retrieve_n_tradeday(t, n):
    '''
    取给定日期过去第n个交易日日期
    :param t: string/datetime/timestamp, 需查询的给定日期
    :param n: int, 日期偏移量, 仅支持向历史偏移
    :return: string, 过去第n个交易日日期
    '''
    if type(t) != str:
        t = t.strftime('%Y-%m-%d')
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    q = sqls_config['past_tradeday']['Sql']%t
    tradeday = db_risk.read_sql(q).sort_values(by=['c_date'])
    return tradeday.iloc[(-1)*(n+1)][0]


class AllAsset(RiskIndicators):

    def __init__(self, t, ptf_codes=None):
        self.basedate = t
        self._format_ptf_codes(ptf_codes)
        self._connectingRiskDB()

    def dpe_asset_type(self, t="", insert2db=True):
        latest_td = self.get_latest_tradeday(self.basedate if t == "" else t)
        data = self.daily_asset_type(t=latest_td)

        # 补充当天卖出的债券
        df_t0 = self.daily_asset_type(t=self.get_offset_tradeday(latest_td, -1))
        data_fill = df_t0[~df_t0['sec_code'].isin(data['sec_code'].to_list())].copy()
        data = pd.concat([data, data_fill], ignore_index=True)

        data['c_date'] = latest_td
        # 检查是否存在重复证券
        check = data[data.duplicated(subset="sec_code")].copy()
        if check.empty:
            logger.info("无重复资产明细")
            self.delete_table('dpe_asset_type_check', t=latest_td, column_name='c_date')
        else:
            logger.error("存在重复资产明细，具体信息请见：dpe_asset_type_check")
            duplicate_df = data.loc[data["sec_code"].isin(check["sec_code"].to_list())]
            self.insert2db_single('dpe_asset_type_check', duplicate_df, t=latest_td, t_colname='c_date')

        if insert2db:
            self.insert2db_single('dpe_asset_type', data, t=latest_td, t_colname='c_date')

        return data

    def dpe_asset_type_spec(self, t="", asset_list=[]):
        latest_td = self.get_latest_tradeday(self.basedate if t == "" else t)
        data = self.daily_asset_type(t=latest_td, asset_list=asset_list)

        # 补充当天卖出的债券
        df_t0 = self.daily_asset_type(t=self.get_offset_tradeday(latest_td, -1), asset_list=asset_list)
        data_fill = df_t0[~df_t0['sec_code'].isin(data['sec_code'].to_list())].copy()
        data = pd.concat([data, data_fill], ignore_index=True)

        data['c_date'] = latest_td
        # 检查是否存在重复证券
        check = data[data.duplicated(subset="sec_code")].copy()
        if check.empty:
            logger.info("无重复资产明细")
            self.delete_table('dpe_asset_type_check', t=latest_td, column_name='c_date')
        else:
            logger.error("存在重复资产明细，具体信息请见：dpe_asset_type_check")
            duplicate_df = data.loc[data["sec_code"].isin(check["sec_code"].to_list())]
            self.insert2db_single('dpe_asset_type_check', duplicate_df, t=latest_td, t_colname='c_date')

        self.general_delete_in('dpe_asset_type', c_date=latest_td, asset_type='可转债')
        self.insert_table('dpe_asset_type', data, t=latest_td)

    def daily_asset_type(self, t, asset_list=[]):
        data = pd.DataFrame()
        # 基础数表：DPE_SDP_STOCKCHRC, dpe_sa_val, cba_bs_detail, dpe_portfoliobond,
        # rc_lr_factor_ir_data, DPE_CREDITBOND_INFO
        asset_list = asset_list if len(asset_list) > 0 else ['stock', 'cbond', 'bond', 'derivative', 'fund', 'abs']
        for sql_name in asset_list:
            temp = self.db_risk.read_sql(sqls_config['dpe_asset_type'][sql_name].format(t=t))
            data = pd.concat([data, temp], ignore_index=True)

        # 由于货币基金的债券名称中含有(总价)，所以需要特殊处理债券数据，以防重复'
        data['sec_abbr'] = data['sec_abbr'].str.replace("\(总价\)", "")
        data = data.drop_duplicates()

        return data

    def dpe_asset_type_nottd(self, t = ""):
        '''非交易日时运行该函数'''
        t = self.basedate if t == "" else t
        latest_td = self.get_latest_tradeday(t)

        if latest_td == t:
            logger.info('%s 为交易日，无需运行非交易日函数' % t)
            return None

        data = self.dpe_asset_type(t=latest_td, insert2db=False)
        data['c_date'] = t

        self.insert2db_single('dpe_asset_type', data, t=t, t_colname='c_date')









