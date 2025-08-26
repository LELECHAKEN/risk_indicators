'''
Description: update daily adjusted fund NAV from valuation.
Author: Wangp
Date: 2021-03-04 17:37:36
LastEditTime: 2021-04-28 13:36:35
LastEditors: Wangp
'''
import json
import datetime
import pandas as pd
import numpy as np
from WindPy import w
from sqlalchemy import exc

w.start()

from .settings import config
from .db.util import DoesNotExist
from .db import OracleDB, column, sqls_config, and_, column
from .utils.log_utils import logger


class PtfNav(object):

    def __init__(self):
        self.db_risk = OracleDB(config['data_base']['QUANT']['url'])
        self._ptf_name_mapping()

    def _ptf_name_mapping(self):
        q = sqls_config['portfolio_type']['Sql']
        ptf_type = self.db_risk.read_sql(q)
        self.ptf_map = ptf_type.set_index('c_fundcode')['c_fundname'].to_dict()

    def insert_table(self, table, data, t, schema='quant', if_exists='append'):
        data['insert_time'] = datetime.datetime.now()
        self.db_risk.insert_dataframe(table=table.lower(), data=data, schema=schema, if_exists=if_exists)
        logger.info('插入成功，table: %s - %s inserted into database.' % (table, t))

    def delete_table(self, table, t, schema='quant', t_type='date', **kwargs):
        date_t = pd.to_datetime(t) if t_type == 'date' else t
        condition = and_(column('D_DATE') == date_t, and_(column(key) == kwargs[key] for key in kwargs))

        try:
            self.db_risk.delete(table.lower(), condition, schema)
            logger.info('删除成功，table: %s - %s deleted from database.' % (
                table, (", ".join([key + "=" + str(kwargs[key]) for key in kwargs]))))
        except (DoesNotExist, exc.NoSuchTableError):
            logger.info('删除失败，table: %s - %s deleted from database.' % (
                table, (", ".join([key + "=" + str(kwargs[key]) for key in kwargs]))))

    def write_ptfnav(self, data, t, tablename='dpe_portfolionav', t_type='date', ptf_codes=[]):
        if len(ptf_codes) > 0:
            for ptf_code in ptf_codes:
                self.delete_table(tablename, t, t_type=t_type, portfolio_code=ptf_code)
        else:
            self.delete_table(tablename, t, t_type=t_type)
        self.insert_table(tablename, data, t)

    def retrieve_n_tradeday(self, t, n):
        '''
        取给定日期过去第n个交易日日期
        :param t: string/datetime/timestamp, 需查询的给定日期
        :param n: int, 日期偏移量, 仅支持向历史偏移
        :return: string, 过去第n个交易日日期
        '''
        if t == '2023-12-31':
            return '2023-12-29'
        if t == '2024-01-02':
            return '2023-12-31'
        t = t if isinstance(t, str) else t.strftime('%Y-%m-%d')
        q = sqls_config['past_tradeday']['Sql'] % t
        tradeday = self.db_risk.read_sql(q).sort_values(by=['c_date'])
        return tradeday.iloc[(-1) * (n + 1)][0]

    def insert_dividend(self, t):
        '''从估值表中清洗出分红信息，并写入数据库'''
        t0 = self.retrieve_n_tradeday(t, 1)
        q = sqls_config['bonus_val']['Sql']
        bonus_t0 = self.db_risk.read_sql(q % t0).rename(columns={'bonus': 'bonus_t0', 'c_date': 'd_date_t0'}).fillna({'bonus_t0': 0})
        bonus_t1 = self.db_risk.read_sql(q % t).rename(columns={'bonus': 'bonus_t1'}).fillna({'bonus_t1': 0})
        bonus = pd.merge(bonus_t1, bonus_t0, on=['portfolio_code', 'full_name'], how='left')
        bonus['dividend'] = bonus['bonus_t1'] - bonus['bonus_t0'].fillna(0)
        bonus = bonus.loc[bonus['dividend'] > 0, ['portfolio_code', 'full_name', 'c_date', 'dividend']].rename(
            columns={'c_date': 'd_date'})
        if bonus.empty:
            return

        bonus['l_fundtype'] = bonus['portfolio_code'].str.contains('SMA').map({True: '3', False: '1'})
        bonus['c_fundname'] = bonus['portfolio_code'].map(self.ptf_map)

        if bonus.shape[0] > 0:
            self.write_ptfnav(bonus.drop(columns=['full_name']), t, 'dpe_dividend', 'str')
        else:
            logger.info('%s未发生分红' % t)

    def loadNavData(self, t, ptf_codes=[], del_ptf_codes=[]):
        '''
        获取单位净值和累计单位净值，包含分级or不分级的基金（分级基金取A类的单位净值）
        :param t: 基期
        :param ptf_codes: 指定产品，默认为空，即全量产品
        :return: nav
        '''

        # 直接从DWS_PTFL_VALUATION_INDEX表中获取基金的单位净值与累计单位净值
        data = self.db_risk.read_sql(sqls_config['dws_ptfval_nav']['Sql'] % t)
        data['d_date'] = pd.to_datetime(data['c_date'])
        data['l_fundtype'] = data['portfolio_code'].str.contains('SMA').map({True: '3', False: '1'})

        keep_cols = ['portfolio_code', 'full_name', 'l_fundtype', 'd_date', 'nav', 'nav_cum']
        rename_cols = {'nav': 'Nav', 'nav_cum': 'Nav_Cum'}
        nav = data.drop_duplicates().reindex(columns=keep_cols).rename(columns=rename_cols)

        if len(ptf_codes) > 0:
            nav = nav[nav['portfolio_code'].isin(ptf_codes)].copy()

        if len(del_ptf_codes) > 0:
            nav = nav[~nav['portfolio_code'].isin(del_ptf_codes)].copy()

        return nav

    def dealBonusMF(self, t, prods, period=1):
        '''从wind取公募的分红数据'''
        if len(prods) == 0:
            return pd.DataFrame(columns=['portfolio_code', 'Bonus_t0', 'Bonus_t1'])
        t0 = self.retrieve_n_tradeday(t,  period)
        wind_temp1 = w.wss(prods, "div_accumulatedperunit", "tradeDate=%s" % t0.replace('-', '')).Data[0]
        wind_temp2 = w.wss(prods, "div_accumulatedperunit", "tradeDate=%s" % t.replace('-', '')).Data[0]
        bonus = pd.DataFrame([prods, wind_temp1, wind_temp2], index=['portfolio_code', 'Bonus_t0', 'Bonus_t1']).T
        bonus['Bonus_t0'] = bonus['Bonus_t0'].fillna(0)
        bonus['Bonus_t1'] = bonus['Bonus_t1'].fillna(0)
        bonus['Bonus'] = bonus['Bonus_t1'] - bonus['Bonus_t0']

        return bonus

    def dealBonusSP(self, t):
        '''从数据库取专户的历史分红信息'''
        bonus = self.db_risk.read_sql(sqls_config['dividend_info']['Sql'])
        bonus = bonus[bonus['l_fundtype'] == 3].drop(columns=['l_fundtype'])
        bonus['d_date'] = pd.to_datetime(bonus['d_date'])
        bonus = bonus.rename(columns={'dividend': 'Bonus'})
        return bonus

    def adjustNav(self, t, nav_m):
        '''单位净值及当日收益率的复权处理'''
        t0 = self.retrieve_n_tradeday(t, 1)
        q = sqls_config['nav_data_t']['Sql']
        nav_t0 = self.db_risk.read_sql(q % t0).rename(columns={'nav': 'Nav_t0', 'nav_adj': 'Nav_adj_t0'})

        nav_m = pd.merge(nav_m, nav_t0[['portfolio_code', 'Nav_t0', 'Nav_adj_t0']], on=['portfolio_code'], how='left')
        nav_m['Nav_t0'] = nav_m['Nav_t0'].fillna(1)
        nav_m['Nav_adj_t0'] = nav_m['Nav_adj_t0'].fillna(1)
        nav_m['ret'] = nav_m['Nav'] / (nav_m['Nav_t0'] - nav_m['Bonus']) - 1  # 先算复权日收益率
        nav_m['Nav_adj'] = nav_m['Nav_adj_t0'] * (1 + nav_m['ret'])

        return nav_m

    def adjustNav_all(self, t, nav):
        # 分红处理：公募
        prods = nav.loc[nav['l_fundtype'] == '1', 'portfolio_code'].tolist()
        if len(prods) == 0:
            nav_m = pd.DataFrame()
        else:
            bonus = self.dealBonusMF(t, prods)
            nav_m = pd.merge(nav[nav['l_fundtype'] == '1'], bonus, on=['portfolio_code'], how='left').fillna(0)
            nav_m = self.adjustNav(t, nav_m)

        # 分红处理：专户
        bonus_s = self.dealBonusSP(t).drop(['insert_time', 'c_fundname'], axis=1)
        nav_s = pd.merge(nav[nav['l_fundtype'] == '3'], bonus_s, on=['portfolio_code', 'd_date'], how='left')
        nav_s['Bonus'] = nav_s['Bonus'].fillna(0)
        nav_m_s = self.adjustNav(t, nav_s)

        # 净值整合
        nav_it = pd.concat([nav_m, nav_m_s], ignore_index=True)
        nav_it = nav_it.rename(columns={'ret': 'return', 'Nav_adj': 'nav_adj', 'Nav': 'nav', 'Nav_Cum': 'nav_cum'})
        nav_it['nav_adj'] = nav_it['nav_adj'].fillna(value=nav_it['nav'])

        return nav_it

    def integrate_nav(self, t, plus_n=None, ptf_codes=None):
        '''
        基金净值，包含：单位净值、累计单位净值、复权单位净值
        :param t: 基期
        :param plus_n: t+n估值，默认n=0
        :return:
        '''
        t = t if plus_n == 0 or plus_n is None else self.retrieve_n_tradeday(t, n=plus_n)
        logger.info('开始计算%s日的组合单位净值数据' % t)
        self.insert_dividend(t)  # 从估值表清洗专户产品分红

        tn_ptf = self.db_risk.read_sql(sqls_config['tn_portfolio']['Sql'])
        if ptf_codes is not None:
            ptf_codes = ptf_codes
        else:
            # 根据t+n获取特定产品，若plus=0，则ptf_codes为空
            ptf_codes = tn_ptf[tn_ptf['n_days'] == plus_n]['portfolio_code'].to_list()

        # 净值数据
        nav = self.loadNavData(t, ptf_codes)  # 从DWS数表中直接获取净值数据
        nav_it = self.adjustNav_all(t, nav).drop_duplicates()  # 调整净值数据

        # 获取基金估值简称
        nav_it['c_fundname'] = nav_it['portfolio_code'].map(self.ptf_map)
        nav_it['c_fundcode'] = [x.split('.')[0] for x in nav_it['portfolio_code']]

        # 数据存储
        cols = ['c_fundname', 'c_fundcode', 'd_date', 'nav', 'nav_cum', 'return', 'nav_adj', 'portfolio_code',
                'insert_time']
        nav_it = nav_it.dropna(subset=['c_fundname']).reindex(columns=cols)
        self.write_ptfnav(nav_it, t, tablename='dpe_portfolionav', t_type='date', ptf_codes=ptf_codes)