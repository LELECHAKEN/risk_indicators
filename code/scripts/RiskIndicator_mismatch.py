# @Time : 2021/8/18 15:13 
# @Author : for wangp
# @File : RiskIndicator_mismatch.py 
# @Software: PyCharm
import numpy as np
import pandas as pd

from .MarketIndicators import MarketIndicators
from .utils.log_utils import logger

class RiskIndicator_mismch(MarketIndicators):
    '''期限错配的风险指标'''
    def __init__(self, t, save_path, ptf_codes=None):
        self.basedate = t
        self.save_path = save_path
        self._format_ptf_codes(ptf_codes)

        self._loadFile()  # 取估值相关的持仓数据:self.val（进行产品筛选）
        self._loadCoding()
        self._loadHoldings(self.save_path)
        self._fundNameMapping()

    def _loadCoding(self):
        self.data_prod = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\产品一览表.xlsx', engine='openpyxl')
        self._fundName = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl').rename(columns={'产品名称': 'C_FULLNAME', '估值系统简称': 'C_FUNDNAME', 'O32产品名称': 'C_FUNDNAME_o32'})
        self.data_prod = pd.merge(self.data_prod, self._fundName[['C_FULLNAME', 'C_FUNDNAME', 'C_FUNDNAME_o32']], left_on=['产品名称'], right_on=['C_FULLNAME'], how='left')
        self._bchPeriod = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\存量专户产品业绩比较基准.xlsx', engine='openpyxl')

    def filter_overdue_secs(self):
        '''清洗持仓券的错配期限，指行权剩余期限大于组合到期日的程度'''
        data = self.bond_holdings[(self.bond_holdings['L_FUNDTYPE'] == 3) & ~self.bond_holdings['WINDL2TYPE'].isin(['可转债', '可交换债', '可分离转债存债'])].copy()
        if data.empty:
            add_cols = ['下一回售日', '债券到期日', '产品到期日', '错配期限', '错配期限_w', 'dura_w', 'overdue_gp']
            self.holdings_overdue = pd.DataFrame(columns=list(data.columns) + add_cols)
            return None

        data['下一回售日'] = pd.to_datetime(data['REPURCHASEDATE_wind'])
        data.loc[data['下一回售日'] <= pd.to_datetime(self.basedate), '下一回售日'] = np.nan
        data['债券到期日'] = data['下一回售日'].fillna(value=data['到期日期'])

        data1 = pd.merge(data, self.data_prod[['C_FUNDNAME', '产品到期日']], on='C_FUNDNAME', how='left')
        data1['错配期限'] = [x.days for x in (data1['债券到期日'] - data1['产品到期日'])]
        data1['错配期限_w'] = data1['错配期限'] * data1['F_ASSETRATIO'] / (100*365)
        data1['dura_w'] = data1['F_ASSETRATIO'] * data1['MODIDURA_CNBD'].fillna(0) / 100
        data1['overdue_gp'] = pd.cut(data1['错配期限']/365, bins=[0, 0.5, 1, np.inf], labels=['6M', '6Mto1Y', '1Yabove'])

        self.holdings_overdue = data1[data1['错配期限'] > 0].copy()

    def calc_overdue_duration(self):
        '''计算各组合的久期错配及占比情况，区分组合、利率债和信用债各类的整体错配久期'''
        if 'holdings_overdue' not in dir(self):
            self.filter_overdue_secs()
        data = self.holdings_overdue[~self.holdings_overdue['WINDL1TYPE'].isin(['资产支持证券'])].copy()

        index_cols = ['C_FUNDNAME', 'D_DATE']
        value_cols = ['dura_overdue', 'dura_overdue_cr', 'dura_overdue_ir', 'ratio_overdue_cr', 'ratio_overdue_ir']
        if data.empty:
            return pd.DataFrame(columns=index_cols + value_cols + ['portfolio_code'])

        res_1 = data.groupby(index_cols).apply(lambda x: x['dura_w'].sum() * 100 / x['F_ASSETRATIO'].sum()).rename('dura_overdue')
        res_dura = pd.pivot_table(data, values='dura_w', index=index_cols, columns='利率or信用', aggfunc='sum')
        res_ratio = pd.pivot_table(data, values='F_ASSETRATIO', index=index_cols, columns='利率or信用', aggfunc='sum')/100
        res_2 = res_dura / res_ratio
        res_2 = res_2.rename(columns={'利率债': 'dura_overdue_ir', '信用债': 'dura_overdue_cr'})
        res_3 = res_ratio.rename(columns={'利率债': 'ratio_overdue_ir', '信用债': 'ratio_overdue_cr'})

        res = pd.merge(res_1, res_2, left_index=True, right_index=True, how='left')
        res = pd.merge(res, res_3, left_index=True, right_index=True, how='left')
        res = res.reset_index().reindex(columns=index_cols + value_cols)
        res['portfolio_code'] = res['C_FUNDNAME'].map(self.fundname_to_code)
        return res

    def calc_overdue_dist(self):
        '''计算各组合在各标准错配期限上的持仓分布情况'''
        if 'holdings_overdue' not in dir(self):
            self.filter_overdue_secs()
        data = self.holdings_overdue.copy()

        index_cols = ['C_FUNDNAME', 'D_DATE']
        value_cols = ['6M', '6Mto1Y', '1Yabove']
        if data.empty:
            return pd.DataFrame(columns=index_cols + value_cols + ['portfolio_code'])

        res = pd.pivot_table(data, index=index_cols, columns='overdue_gp', values='F_ASSETRATIO', aggfunc='sum')/100
        res.columns = res.columns.tolist()
        res = res.reset_index().reindex(columns=index_cols + value_cols)
        res['portfolio_code'] = res['C_FUNDNAME'].map(self.fundname_to_code)
        return res

    def calc_overdue_expo(self):
        '''计算各组合的错配敞口情况'''
        data_holding = self.bond_holdings.copy()
        data_holding = data_holding[data_holding["利率or信用"] != "可转债"].copy()  # 剔除可转债
        data_holding['dura_w'] = data_holding['F_ASSETRATIO'] * data_holding['MODIDURA_CNBD'] / 100

        ptm_port = self.data_prod[['C_FUNDNAME', '产品到期日']].dropna(subset=['产品到期日'])
        ptm_port['ptm_port'] = [(x - pd.to_datetime(self.basedate)).days / 365 for x in ptm_port['产品到期日']]
        ptm_port = ptm_port[ptm_port['ptm_port'] >= 0]

        index_cols = ['C_FUNDNAME', 'D_DATE']
        value_cols = ['dura_port', 'dura_cr', 'dura_ir', 'ptm_port', 'overdue_expo_port', 'overdue_expo_ir', 'overdue_expo_cr']
        if data_holding.empty:
            return pd.DataFrame(columns=index_cols + value_cols + ['portfolio_code'])

        dura_port = data_holding.groupby(index_cols)['dura_w'].sum().rename('dura_port').dropna()
        res_dura = pd.pivot_table(data_holding, values='dura_w', index=index_cols, columns='利率or信用', aggfunc='sum')
        res_ratio = pd.pivot_table(data_holding, values='F_ASSETRATIO', index=index_cols, columns='利率or信用', aggfunc='sum')/100
        dura_bond = res_dura / res_ratio
        dura_bond = dura_bond.rename(columns={'利率债': 'dura_ir', '信用债': 'dura_cr'}).reindex(columns=['dura_ir', 'dura_cr'])

        res_all = pd.merge(dura_port, dura_bond, left_index=True, right_index=True, how='left').reset_index()
        res_all = pd.merge(res_all, ptm_port[['C_FUNDNAME', 'ptm_port']], on='C_FUNDNAME', how='right')
        res_all = res_all.dropna(subset=['dura_port'])
        res_all['overdue_expo_port'] = np.where(res_all['dura_port'] > res_all['ptm_port'], res_all['dura_port'] - res_all['ptm_port'],0)
        res_all['overdue_expo_ir'] = np.where(res_all['dura_ir'] > res_all['ptm_port'], res_all['dura_ir'] - res_all['ptm_port'], 0)
        res_all['overdue_expo_cr'] = np.where(res_all['dura_cr'] > res_all['ptm_port'], res_all['dura_cr'] - res_all['ptm_port'], 0)
        res_all = res_all.replace(0, np.nan).reindex(columns=index_cols + value_cols)
        res_all['portfolio_code'] = res_all['C_FUNDNAME'].map(self.fundname_to_code)

        return res_all

    def calc_keyyear_ratio(self, label=''):
        '''计算关键期限占比'''
        if 'dura_asset' not in dir(self):
            self.duraP_holdings_ex, self.dura_asset = self.dealKey_duration()
        self.data_keyRatio = self.dealKey_year(self.duraP_holdings_ex)

    def calc_all(self):
        self.overdue_dist = self.calc_overdue_dist()
        self.overdue_dura = self.calc_overdue_duration()
        self.overdue_expo = self.calc_overdue_expo()

    def insert2db(self):
        table_list = ['rc_mismch_dist', 'rc_mismch_dura', 'rc_mismch_expo']
        data_list = [self.overdue_dist, self.overdue_dura, self.overdue_expo]
        for table, data in zip(table_list, data_list):
            if data.empty:
                logger.info('%s -- %s 无新增数据' % (self.basedate, table))
                continue
            self.insert2db_single(table, data, t=self.basedate, ptf_code=self.ptf_codes, code_colname='portfolio_code')