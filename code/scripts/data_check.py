# @Time : 2021/7/13 17:25 
# @Author : for wangp
# @File : data_check.py 
# @Software: PyCharm
import numpy as np
import pandas as pd

from .settings import config
from .utils.log_utils import logger
from .db import OracleDB, sqls_config

from .utils_ri.RiskIndicators import RiskIndicators

class DataCheck(RiskIndicators):
    def __init__(self, t, ptf_codes=None):
        self.basedate = t
        self._format_ptf_codes(ptf_codes)
        self._loadFile()
        self._loadNavData()

    def _loadNavData(self):
        '''
        仅加载专户的单位净值数据
        :return: None
        '''
        self.prods_sma = self.val.loc[(self.val['L_FUNDTYPE'] == '3'), 'C_FUNDNAME'].drop_duplicates().tolist()
        self.prods_mf = self.val.loc[self.val['L_FUNDTYPE'].isin(['1','13']), 'C_FUNDNAME'].drop_duplicates().tolist()
        self.prods = self.val.loc[:, 'C_FUNDNAME'].drop_duplicates().tolist()
        prods_str = ','.join('\'' + x + '\'' for x in self.prods)

        sql_nav = sqls_config['nav_data']['Sql']%prods_str
        res = self.db_risk.read_sql(sql_nav)
        res.columns = [x.upper() for x in res.columns]
        res = res.rename(columns={'NAV':'NAV_orig', 'NAV_ADJ': 'NAV', 'RETURN':'ret'})
        self.nav_f = res[res['D_DATE'] <= pd.to_datetime(self.basedate)].dropna(subset=['NAV']).sort_values(by=['C_FUNDNAME', 'D_DATE']).drop_duplicates()

    def deal_ret_data(self):
        q = sqls_config['dws_ptfval_ret']['Sql']
        res = self.db_risk.read_sql(q % self.basedate).rename(columns={'d_date': 'D_DATE', 'c_fundname': 'C_FUNDNAME'})
        res['C_FUNDNAME'] = res['C_FUNDNAME'].fillna('')
        res['ret_daily_val'] = res.apply(lambda x: 0 if '货币' in x.C_FUNDNAME else x.ret_daily_val, axis=1)
        return res

    def ReturnRate(self, netvalue_df=0):
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        returnRate = netvalue_df['NAV'].iloc[-1] / netvalue_df['NAV'].iloc[0] - 1

        return returnRate

    def check_nav(self):
        '''
        检查当日累计单位净值-1是否等于产品自成立至今的累计收益率。仅根据第一个和最后一个nav数据。
        通常可以排查出期初是否缺失数据
        :return:
        '''
        ret_cum = self.nav_f[self.nav_f['C_FUNDNAME'].isin(self.prods_sma)].set_index(['C_FUNDNAME', 'D_DATE'])
        ret_cum = ret_cum.groupby(['C_FUNDNAME']).apply(self.ReturnRate).reset_index().rename(columns={0: 'ReturnRate'})
        ret_nav = self.nav_f[(self.nav_f['D_DATE'] == self.basedate) & self.nav_f['C_FUNDNAME'].isin(self.prods_sma)].copy()
        ret_nav['ret_cum'] = ret_nav['NAV'] - 1

        check = pd.merge(ret_cum, ret_nav, on=['C_FUNDNAME'], how='right')
        check['check'] = check['ret_cum'] - check['ReturnRate']
        check_res = check[check['check'] != 0].copy()

        return check_res

    def check_ret_cum(self):
        '''
        检查全量单位净值表，逻辑为：用数据库的单日ret做累计得到计算的累计收益率，与累计净值-1相比。
        :return:check_res_data为单日差异绝对值超过0.1bp的详细数据，check_res_prod为最后一日的产品清单及对应区间差异
        '''
        ret_calc = self.nav_f[self.nav_f['C_FUNDNAME'].isin(self.prods_sma)].sort_values(by=['C_FUNDNAME', 'D_DATE'])
        ret_calc['ret_cum_calc'] = ret_calc.groupby(['C_FUNDNAME'])['ret'].apply(lambda x: (1+x).cumprod()-1)
        ret_nav = self.nav_f[['C_FUNDNAME', 'D_DATE', 'NAV']].copy()
        ret_nav['ret_cum'] = ret_nav['NAV'] - 1

        check = pd.merge(ret_calc, ret_nav, on=['C_FUNDNAME', 'D_DATE'], how='right')
        check['check'] = check['ret_cum'] - check['ret_cum_calc']
        check_res_data = check[np.abs(check['check']) > 0.00001].copy()
        check_res_prod = check[np.abs(check['check']) > 0.00001].drop_duplicates(subset=['C_FUNDNAME'])

        return check_res_data, check_res_prod

    def check_ret_daily(self):
        '''
        比对单日模型计算的ret与估值表中披露的单日ret
        :return:单日return差异绝对值超过0.1bp的数据
        '''
        ret_daily = self.deal_ret_data()
        ret_nav = self.nav_f[self.nav_f['D_DATE'] == self.basedate].copy()

        check = pd.merge(ret_daily, ret_nav, on=['C_FUNDNAME', 'D_DATE'], how='right')
        check['check'] = check['ret'] - check['ret_daily_val']
        check_res = check[(np.abs(check['check']) >= 0.00001) | check['check'].isna()].copy()

        self.insert2db_single('dpe_nav_check', check_res, t=self.basedate)

        return check_res
