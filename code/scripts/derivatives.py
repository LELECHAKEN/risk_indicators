# @Time : 2021/6/25 14:21 
# @Author : for wangp
# @File : derivatives.py
# @Software: PyCharm
import datetime
import numpy as np
import pandas as pd
from WindPy import w
from sqlalchemy import exc

from . import demo_code
from .settings import config
from .utils.log_utils import logger
from .db import OracleDB, column, sqls_config
from .db.util import DoesNotExist
from .db.db_utils import JYDB_Query
from .utils_ri.RiskIndicators import RiskIndicators


class Derivatives(RiskIndicators):
    def __init__(self, t, ptf_codes=None):
        self.basedate = t
        self._format_ptf_codes(ptf_codes)
        self._loadFile()  # 加载val信息时，已经进行了产品代码的相关筛选

    def _get_derivt_holdings(self, sec_type):
        '''拉取衍生品持仓，默认拉取sec_type为6的持仓，覆盖期货合约'''
        cols = ['PORTFOLIO_CODE', 'D_DATE', 'L_SETCODE', 'C_FULLNAME', 'C_FUNDNAME', 'code', 'C_SUBNAME_BSH', 'C_STOPINFO',
                'F_MOUNT', 'F_PRICE', 'F_ASSET', 'F_ASSETRATIO', 'F_NETCOST', 'F_COST', 'F_COSTRATIO', 'L_STOCKTYPE',
                'L_FUNDKIND', 'L_FUNDKIND2', 'L_FUNDTYPE']
        data = self.val.reindex(columns=cols)
        # 取出有交易字段的持仓数据
        derivt_holdings = data[(data['L_STOCKTYPE'] == sec_type) & (data['code'] != '*')].drop_duplicates()
        return derivt_holdings

    def derivative_holdings(self):
        self.future_holdings = self._get_derivt_holdings(sec_type='6')
        self.crmw_holdings = self._get_derivt_holdings(sec_type='22')

    def run(self):
        self.derivative_holdings()

        # CRMW
        CRMW = Derivt_CRMW(self.basedate, self.ptf_codes, self.crmw_holdings)
        CRMW.run_crmw()

        # future
        Future = Derivt_Future(self.basedate, self.ptf_codes, self.future_holdings)
        Future.run_future()


class Derivt_Future(RiskIndicators):
    def __init__(self, t, ptf_codes, holdings):
        self.basedate = t
        self.ptf_codes = ptf_codes
        self.holdings = holdings

    def _define_contract_type(self):
        self.ct_dict = {'I': 'StockIndexFutures',
                        'T': 'TreasuryFutures',
                        '商品期货': 'CommodityFutures',
                        '货币期货': 'CurrencyFutures',
                        '利率期货': 'TreasuryFutures',
                        '股指期货': 'StockIndexFutures',
                        '虚拟货币期货': 'VirtualCurrencyFutures',
                        '股票期货': 'StockFutures'}
        self.mkt_dict = {'6': '.CFE', '22': '.IB'}

    def _get_contract_type(self, sec_list):
        db_jy = OracleDB(config['data_base']['JY']['url'])
        secs = ','.join('\'' + str(x.split('.')[0]) + '\'' for x in sec_list)
        q = sqls_config['contracttype']['Sql']%secs
        data = db_jy.read_sql(q)
        data['contractcode'] = [x + '.CFE' for x in data['contractcode']]
        return data

    def get_ctd(self, sec_list, t=''):
        '''
        从wind拉取国债期货合约对应的最廉券CTD代码\n
        :param sec_list: list, 国债期货合约代码清单
        :param t: string, yyyy-mm-dd格式, 计算基期
        :return: DataFrame
        '''
        if len(sec_list) == 0:
            return pd.DataFrame(columns=['code', 'TBF_CTD'])
        secs = ','.join(x for x in sec_list)
        tday = (self.basedate if t == '' else t).replace('-', '')

        wind_temp = w.wss(secs, "tbf_CTD", "tradeDate=%s;exchangeType=NIB" % tday)
        data_wind = pd.DataFrame(wind_temp.Data, columns=wind_temp.Codes, index=wind_temp.Fields).T
        data_wind = data_wind.reset_index().rename(columns={'index': 'code'})

        return data_wind

    def get_WindDuration(self, sec_list, t=''):
        '''
        从wind拉取债券久期\n
        :param sec_list: list, 债券代码清单
        :param t: string, yyyy-mm-dd格式, 计算基期
        :return: DataFrame
        '''
        if len(sec_list) == 0:
            return pd.DataFrame(columns=['code', 'MODIDURA_CNBD'])
        secs = ','.join(x for x in sec_list)
        tday = (self.basedate if t == '' else t).replace('-', '')

        wind_temp = w.wss(secs, "modidura_cnbd", "tradeDate=%s;credibility=1" % tday)
        data_wind = pd.DataFrame(wind_temp.Data, columns=wind_temp.Codes, index=wind_temp.Fields).T
        data_wind = data_wind.reset_index().rename(columns={'index': 'code'})

        return data_wind

    def formatFuture(self):
        '''清洗期货持仓, 覆盖标的合约代码、合约类型、ctd及对应等价久期等信息'''
        data = self.holdings.copy()

        sec_code = data['code'].unique().tolist()
        if len(sec_code) == 0:
            self.holdings = pd.DataFrame(columns=['PORTFOLIO_CODE', 'D_DATE', 'L_SETCODE', 'C_FULLNAME', 'C_FUNDNAME',
             'C_SUBNAME_BSH', 'C_STOPINFO', 'F_MOUNT', 'F_PRICE', 'F_ASSET',
             'F_ASSETRATIO', 'F_NETCOST', 'F_COST', 'F_COSTRATIO', 'L_STOCKTYPE',
             'L_FUNDKIND', 'L_FUNDKIND2', 'L_FUNDTYPE', 'code', 'CTD', 'CTD_dura', 'futures_type', 'PORTFOLIO_CODE'])
            logger.info('%s - 无期货持仓。' % self.basedate)
            return None

        res0 = self.get_ctd(sec_code).rename(columns={'TBF_CTD': 'CTD'})
        res1 = self.get_WindDuration(res0['CTD'].dropna().unique().tolist()).rename(columns={'MODIDURA_CNBD': 'CTD_dura', 'code': 'CTD'})
        res = pd.merge(res0, res1, on=['CTD'], how='left')
        data = pd.merge(data, res, on=['code'], how='left')

        # define contract type based on jydb data.
        future_type = self._get_contract_type(sec_code)[['contractcode', 'contracttype']].copy()
        data = pd.merge(data, future_type, left_on='code', right_on='contractcode', how='left')
        data['futures_type'] = data['contracttype'].apply(lambda x: self.ct_dict[x]).fillna('OtherFutures')
        data = data.drop(columns=['contractcode', 'contracttype'])

        if 'C_STOCKCODE_BSH' not in data.columns:
            data['C_STOCKCODE_BSH'] = None

        self.holdings = data.copy()
        self.insert2db_single('DPE_PORTFOLIODERIVT', self.holdings, ptf_code=self.ptf_codes)

    def calcEquivalentDuration(self):
        '''计算组合层面的国债期货等价久期'''
        data = self.holdings[self.holdings['futures_type'] == 'TreasuryFutures'].copy()
        if data.shape[0] == 0:
            logger.info('%s - 无国债期货持仓。' % self.basedate)
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'derivt_position', 'ctd_equi_dura'])

        data['CTD_dura_w'] = data['F_ASSETRATIO'] * data['CTD_dura'] / 100
        grouped = data.groupby(['PORTFOLIO_CODE', 'C_FUNDNAME', 'D_DATE'])
        res_position = grouped['F_ASSETRATIO'].sum()/100
        res_equi_dura = grouped['CTD_dura_w'].sum()
        res = pd.merge(res_position, res_equi_dura, left_index=True, right_index=True, how='left').reset_index()
        res = res.rename(columns={'F_ASSETRATIO': 'derivt_position', 'CTD_dura_w': 'ctd_equi_dura'})

        self.equi_dura = res
        self.insert2db_single('RC_DERIVT', res, ptf_code=self.ptf_codes)

    def run_future(self):
        self._define_contract_type()
        self.formatFuture()  # 全量期货持仓
        self.calcEquivalentDuration()  # 国债期货等价久期


class Derivt_CRMW(RiskIndicators):
    def __init__(self, t, ptf_codes, holdings):
        self.basedate = t
        self.ptf_codes = ptf_codes
        self.holdings = holdings

    def _deal_crmw_market(self, data):
        '''交易所上市的CRMW在trep_valuation表中的市场字段有误，需要重新从聚源的主表中取出准确的上市市场字段'''
        data_orig = data.copy()
        sec_code = data_orig['code'].unique().tolist()
        self.db_jy = JYDB_Query()
        data_jy = self.db_jy.sec_query('jy_crmw_market', sec_code)
        data_jy.columns = [x.lower() for x in data_jy.columns]
        data_jy['market'] = data_jy['secumarket'].map({83: '.SH', 90: '.SZ', 89: '.IB'}).fillna('.IB')
        data_orig['secucode'] = [x.split('.')[0] for x in data_orig['code']]
        data_res = pd.merge(data_orig, data_jy[['secucode', 'market']], on='secucode', how='left').fillna({'market': '.IB'})
        data_res['code'] = data_res[['secucode', 'market']].sum(axis=1)
        return data_res.drop(columns=['secucode', 'market'])

    def formatCRMW(self):
        data = self.holdings.copy()
        sec_code = data['code'].unique().tolist()

        if len(sec_code) == 0:
            logger.info('%s - 无CRMW持仓。' % self.basedate)
            return

        data = self._deal_crmw_market(data)
        sec_code = data['code'].unique().tolist()
        crmw_info = self.getCRMW_info(sec_code, self.basedate).rename(columns={'DIRTY_CNBD': 'DIRTY_CNBD_CRMW'})
        crmw_info['CRM_STARTINGPRICE'] = crmw_info['CRM_STARTINGPRICE'].astype(float)
        crmw_info['DIRTY_CNBD_CRMW'] = crmw_info['DIRTY_CNBD_CRMW'].astype(float)
        data = pd.merge(data, crmw_info, on=['code'], how='left')
        data['CRM_SUBJECTCODE'] = [x + '.IB' if isinstance(x, str) else None for x in data['CRM_SUBJECTCODE'] ]

        if 'C_STOCKCODE_BSH' not in data.columns:
            data['C_STOCKCODE_BSH'] = None

        self.holdings = data.copy()
        self.insert2db_single('dpe_portfoliocrmw', self.holdings, ptf_code=self.ptf_codes)

    def getCRMW_info(self, sec_list, t):
        '''
        从wind拉取crmw的合约要素，如标的实体交易代码crm_subjectcode, 创设价格crm_startingprice, 当前估值全价dirty_cnbd\n
        :param sec_list: list, crmw合约代码清单
        :param t: string, yyyy-mm-dd格式, 计算基期
        :return: DataFrame
        '''
        if len(sec_list) == 0:
            return pd.DataFrame(columns=['code', 'CRM_SUBJECTCODE', 'CRM_STARTINGPRICE', 'DIRTY_CNBD'])
        secs = ','.join(x for x in sec_list)
        tday = t.replace('-', '')

        wind_temp = w.wss(secs, "crm_subjectcode,crm_startingprice,dirty_cnbd", "tradeDate=%s;credibility=1" % tday)
        data_wind = pd.DataFrame(wind_temp.Data, columns=wind_temp.Codes, index=wind_temp.Fields).T
        data_wind = data_wind.reset_index().rename(columns={'index': 'code'})

        return data_wind

    def run_crmw(self):
        self.formatCRMW()
