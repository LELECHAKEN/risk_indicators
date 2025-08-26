'''
Description: Derived from Liquidity Indicators, focusing on Hierarchical Asset and Liability Liquidity.
Author: Wangp
Date: 2020-08-20 10:15:50
LastEditTime: 2021-06-15 15:58:43
LastEditors: Wangp
'''
import os
import numpy as np
import pandas as pd
from datetime import datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta

from .settings import config
from .db import OracleDB, column, sqls_config
from .db.db_utils import JYDB_Query
from .utils.log_utils import logger
from .LiquidityIndicators import LiquidityIndicators
from .Liquidity_Asset import Liquidity_Asset, Liquidity_IRBond

class LiquidityIndicators_core(LiquidityIndicators):
    '''各类资产的分级核心流动性情况'''
    def __init__(self, t, save_path, ptf_codes=None):
        LiquidityIndicators.__init__(self, t, save_path, ptf_codes)
        self.save_path = save_path
        self.asset_list = ['现金', '逆回购到期', '利率债', '同业存单', '信用债', 'ABS', '可转债', '股票']
        self.part = 0.3    # 信用债的市场参与度
        self._loadTableColumns()

    def get_offset_month(self, t="", n=1):
        '''
        月度日期偏移函数
        :param t: str or datetime，当前日期
        :param n: int, 偏移量，可正可负
        :return: str
        '''
        t = self.basedate if t == "" else t
        adj_day = 1 if n < 0 else -1
        date_t = datetime.strptime(t, "%Y-%m-%d") if isinstance(t, str) else t
        past_m = (int(date_t.year) * 12 + int(date_t.month) + n) % 12
        past_m = 12 if past_m == 0 else past_m
        past_y = int((int(date_t.year) * 12 + int(date_t.month) + n) / 12)
        past_d = monthrange(past_y, past_m)[1] if date_t.day > monthrange(past_y, past_m)[1] else date_t.day
        offset_t = datetime(past_y, past_m, past_d) + relativedelta(days=adj_day)
        return offset_t.strftime("%Y-%m-%d")
    
    def OverDueSecs(self):
        '''拉取专户到期前可变现or不可变现的券明细'''
        if 'ttm' not in dir(self):
            self.ttm = self.getTTM_port(self.fund_list)
        if 'liq_time' not in dir(self):
            self.liq_time = self.calcLiquidTime()

        funds = self.ttm.loc[self.ttm['剩余到期天数'].notna(), 'C_FUNDNAME'].tolist()
        if len(funds) == 0:
            return pd.DataFrame(columns=self.column_dict['变现个券明细']), pd.DataFrame(columns=self.column_dict['变现个券明细'])

        data = self.liq_holdings[self.liq_holdings['C_FUNDNAME'].isin(funds)].copy()
        if data.empty:
            return pd.DataFrame(), pd.DataFrame()
        data = pd.merge(data, self.ttm, on=['C_FUNDNAME', 'D_DATE'], how='left')
        data['liq_end_date'] = pd.to_datetime(data['REPURCHASEDATE_wind']).fillna(value=data['到期日期'])
        # 个券到期日是否产品到期日
        data['cpr_end_date'] = data.apply(lambda x: x['liq_end_date'] > x['产品到期日'] if isinstance(['liq_end_date'], datetime) else False, axis=1)

        data_over1 = data[(data['变现天数'] > data['剩余到期天数']) & (data['cpr_end_date'])].\
            sort_values(by=['C_FUNDNAME', 'D_DATE', '变现天数'], ascending=False).drop(columns='cpr_end_date')
        data_over2 = data[(data['变现天数'] <= data['剩余到期天数']) | (~data['cpr_end_date'])].\
            sort_values(by=['C_FUNDNAME', 'D_DATE', '变现天数'], ascending=False).drop(columns='cpr_end_date')

        return data_over1, data_over2

    def _SimpleStat_OverDue(self, data):
        if data.empty:
            cnt = 0
            ratio = 0
        else:
            cnt = data['code'].nunique()
            ratio = data['F_ASSETRATIO'].sum() / 100
        res = pd.DataFrame([cnt, ratio], index=['cnt', 'ratio']).T

        return res

    def calcAllLiquidation(self):
        '''统计到期前能够变现&不能变现的资产，包括证券只数和市值占净资产比'''
        self.data_over1, self.data_over2 = self.OverDueSecs()

        index_cols = ['C_FUNDNAME', 'D_DATE']
        unable_cols = ['不可变现_证券只数', '不可变现_市值占净资产比']
        able_cols = ['可变现_证券只数', '可变现_市值占净资产比']
        if self.data_over1.empty:
            self.res_unable = pd.DataFrame(columns=index_cols + unable_cols)
        else:
            res_unable = self.data_over1.groupby(index_cols).apply(self._SimpleStat_OverDue).rename(columns={'cnt': '不可变现_证券只数', 'ratio': '不可变现_市值占净资产比'}).reset_index()
            self.res_unable = res_unable.reindex(columns=index_cols + unable_cols)
        if self.data_over2.empty:
            self.res_able = pd.DataFrame(columns=index_cols + able_cols)
        else:
            res_able = self.data_over2.groupby(index_cols).apply(self._SimpleStat_OverDue).rename(columns={'cnt': '可变现_证券只数', 'ratio': '可变现_市值占净资产比'}).reset_index().drop(columns='level_2')
            self.res_able = res_able.reindex(columns=index_cols + able_cols)
        if self.ttm.empty:
            cols = list(self.ttm.columns) + able_cols + unable_cols + ['PORTFOLIO_CODE']
            return pd.DataFrame(columns=cols)

        res = pd.merge(self.ttm, self.res_able, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res = pd.merge(res, self.res_unable, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res['PORTFOLIO_CODE'] = res['C_FUNDNAME'].map(self.fundname_to_code)
        return res

    def getLiquidityAsset_t(self, interval=1):
        '''计算T日可变现的资产规模：证券持仓及现金、逆回购等'''
        data = self.liq_holdings.copy()

        # 计算t日可变现的证券资产规模(金额)
        if 'liq_%d' % interval not in dir(self):
            liq_t = pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'code', 'liq_amt%d'%interval, '%d日可变现_张'%interval])
            for t in self.date_list:
                if self.liq_holdings_t[t].empty:
                    continue
                cal = self.LiqAsset_dict[t]
                liq_t_temp = cal.calc_liquidity_Tday(interval)
                liq_t = pd.concat([liq_t, liq_t_temp], sort=False)

            data = pd.merge(data, liq_t, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')
            data['%d日可变现'%interval] = data['%d日可变现_张'%interval] * data['F_PRICE'].astype(float)

        # 计算t日可变现的现金+逆回购规模
        cash_t = pd.DataFrame()
        for t in self.date_list:
            cash_temp = self.getCashRepoLiq(t, interval=interval)
            cash_t = pd.concat([cash_t, cash_temp], sort=False)

        # 合并证券资产与现金资产的变现能力至res_liq
        if data.empty:
            res_liq = cash_t.reindex(columns=list(cash_t.columns) + ['%d日可变现'%interval])
        else:
            res_liq = data.groupby(['C_FUNDNAME', 'D_DATE'])['%d日可变现'%interval].sum().reset_index()
            res_liq = pd.merge(res_liq, cash_t, on=['D_DATE', 'C_FUNDNAME'], how='right')

        self.liq_holdings = data.drop_duplicates()

        return res_liq, data

    def calcLiquidity_pivot(self, interval_list=[1, 2, 3, 5]):
        '''计算组合T日可变现：1\2\3\5'''
        cash = pd.DataFrame()  # 组合层t日可变现总规模汇总
        asset1 = pd.DataFrame()  # 持仓资产的t日可变现规模(纵向堆叠)
        asset2 = self.liq_holdings.reindex(columns=['C_FUNDNAME', 'D_DATE', 'LiqType', 'code']).drop_duplicates() # 持仓资产的t日可变现规模(横向合并)
        for interval in interval_list:
            print('Calculating: %s 日' % str(interval))
            cash_temp, data = self.getLiquidityAsset_t(interval)

            cols_0 = ['C_FUNDNAME', 'D_DATE', '%d日可变现'%interval, 'Deposit', '%d日逆回购'%interval]
            cols_map = {'%d日可变现'%interval: 'T日可变现总规模', 'Deposit': '现金', '%d日逆回购'%interval: '逆回购到期'}
            cash_temp = cash_temp.reindex(columns=cols_0).rename(columns=cols_map)
            cash_temp['T日可变现总规模'] = cash_temp[['现金', '逆回购到期', 'T日可变现总规模']].sum(axis=1)
            cash_temp['变现能力'] = str(interval) + '日可变现'
            cash = pd.concat([cash, cash_temp], sort=False)

            cols_1 = ['C_FUNDNAME', 'D_DATE', 'LiqType', 'code', '%d日可变现'%interval]
            asset_temp1 = data.reindex(columns=cols_1).sort_values(by=['C_FUNDNAME', 'D_DATE', 'LiqType'])
            asset_temp1 = asset_temp1.rename(columns={'%d日可变现'%interval: 'T日可变现'}).drop_duplicates()
            asset_temp1['变现能力'] = str(interval) + '日可变现'
            asset1 = pd.concat([asset1, asset_temp1], sort=False)

            cols_2 = ['C_FUNDNAME', 'D_DATE', 'LiqType', 'code', 'liq_amt%d'%interval, '%d日可变现_张'%interval, '%d日可变现'%interval]
            asset_temp2 = data.reindex(columns=cols_2).drop_duplicates()
            asset2 = pd.merge(asset2, asset_temp2, on=['C_FUNDNAME', 'D_DATE', 'LiqType', 'code'], how='left')
            asset2 = asset2.drop_duplicates().sort_values(by=['C_FUNDNAME', 'D_DATE', 'LiqType'])
            print('Done.')

        index_cols = ['C_FUNDNAME', 'D_DATE', '变现能力']
        res_pivot = pd.pivot_table(asset1, values='T日可变现', index=index_cols, columns='LiqType', aggfunc='sum').reset_index()
        res_pivot = res_pivot.reindex(columns=index_cols + ['ABS', '信用债', '利率债', '可转债', '同业存单', '股票'])
        res = pd.merge(cash, res_pivot, on=index_cols, how='left')
        data_fund = self.data_fund[['C_FUNDNAME', 'D_DATE', 'TotalAsset']].rename(columns={'TotalAsset': '总资产'})
        res = pd.merge(res, data_fund, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res['CONC'] = res[['C_FUNDNAME','变现能力']].sum(axis=1)
        res = res.sort_values(by=['C_FUNDNAME', 'D_DATE', '变现能力'])

        res['PORTFOLIO_CODE'] = res['C_FUNDNAME'].map(self.fundname_to_code)
        return res, asset1, asset2

    def _dictCoreLevel(self):
        '''定义核心流动性等级'''
        self.dict_core = {'高流动性': '5级核心', 
                        '不活跃券': '4级核心', 
                        '次活跃券': '3级核心', 
                        '活跃券': '2级核心', 
                        '10年国开活跃券': '1级核心', 
                        '10年国开次活跃券': '1级核心'}

    def _defineIRCoreLevel(self, t, data):
        '''按照核心流动性划分利率债活跃度，主要将10年国开区分出来'''
        liq_ir = Liquidity_IRBond(t)
        liq_ir.loadCoeffients()
        liq_ir.loadActivityBond()
#        data1 = liq_ir.DefineActivity(data)
        data_act = liq_ir.data_act.rename(columns={'ptmyear': '剩余期限', 'issuer': '发行人', 'activity': '是否活跃券'}
                                          ).reindex(columns=['code', '剩余期限', '发行人', '是否活跃券']
                                                    ).dropna(subset=['剩余期限'])
        data_act.loc[data_act[['剩余期限', '发行人']].sum(axis=1) == '10年国开', '是否活跃券'] = data_act.loc[data_act[['剩余期限', '发行人']].sum(axis=1) == '10年国开', ['剩余期限', '发行人', '是否活跃券']].sum(axis=1)
        
        data1 = pd.merge(data, data_act[['code', '是否活跃券']], on='code', how='left')
        data1['活跃度'] = data1['是否活跃券'].fillna('不活跃券')
        data1.loc[~data1['WINDL2TYPE'].isin(['政策银行债', '国债']), '活跃度'] = np.nan

        return data1

    def classifyCoreLiquidity(self):
        '''划分持仓资产的各核心流动性等级'''
        if 'dict_core' not in dir(self):
            self._dictCoreLevel()

        data = self.liq_holdings.copy()
        if data.empty:
            self.liq_holdings = data.reindex(columns=list(data.columns) + ['coreClass'])
            self.core_holdings = self.liq_holdings.copy()
            return None

        # 定义利率债活跃度和信用债高流动性
        data1 = pd.DataFrame(columns=data.columns)
        for t in self.date_list:
            data_temp = data.loc[data['D_DATE'] == t, :].copy()
            
            # 利率债活跃度
            data_temp1 = self._defineIRCoreLevel(t, data_temp)
            # 信用债高流动性
            liq = Liquidity_Asset(t, data_temp1)
            sec_list = liq.calcHighLiqBond()
            data_temp1.loc[data_temp1['code'].isin(sec_list), '是否高流动性债券'] = '是'
            
            data1 = pd.concat([data1, data_temp1], sort=False)

        data1['coreClass'] = data1['活跃度'].fillna(value=data1['是否高流动性债券'].replace('是', '高流动性')).map(self.dict_core)
        self.liq_holdings = data1.copy()
        self.core_holdings = data1[data1['coreClass'].notna()].sort_values(by=['C_FUNDNAME', 'D_DATE', 'coreClass', '变现天数'])


    def calcCoreLiquidity(self):
        '''计算各核心流动性的资产规模'''
        res_cols = ['C_FUNDNAME', 'D_DATE', '1级核心', '2级核心', '3级核心', '4级核心', '5级核心', '总资产']
        if 'coreClass' not in self.liq_holdings.columns:
            self.classifyCoreLiquidity()

        if self.liq_holdings.empty:
            return pd.DataFrame(columns=res_cols)

        data = self.liq_holdings.copy()
        res = pd.pivot_table(data, values='F_ASSET', index=['C_FUNDNAME', 'D_DATE'], columns='coreClass', aggfunc='sum').fillna(0).reset_index()
        
        data_fund = self.data_fund[['C_FUNDNAME', 'D_DATE', 'TotalAsset']].rename(columns={'TotalAsset': '总资产'})
        res = pd.merge(res, data_fund, on=['C_FUNDNAME', 'D_DATE'], how='left').reindex(columns=res_cols)

        self.res_core = res.copy()
        return res

    def _getCoreSecs_level(self, x):
        return ';'.join(i for i in x)

    def calcCoreLiq_secs(self):
        '''统计各核心流动性的证券清单'''
        res_cols = ['C_FUNDNAME', 'D_DATE', '1级核心', '2级核心', '3级核心', '4级核心', '5级核心', 'PORTFOLIO_CODE']
        if self.liq_holdings.empty:
            return pd.DataFrame(columns=res_cols)

        if 'coreClass' not in self.liq_holdings.columns:
            self.classifyCoreLiquidity()
        data = self.liq_holdings.copy()
        res = pd.pivot_table(data, values='C_SUBNAME_BSH', index=['C_FUNDNAME', 'D_DATE'], columns='coreClass', aggfunc=self._getCoreSecs_level).reset_index()
        if not res.empty:
            res['PORTFOLIO_CODE'] = res['C_FUNDNAME'].map(self.fundname_to_code)

        return res.reindex(columns=res_cols)

    def _calcPortDuration(self, data, col_weight='F_ASSETRATIO'):
        dura_port = (data[col_weight] * data['MODIDURA_CNBD'].fillna(value=data['Duration'])).sum() / 100
        return dura_port

    def calcCoreLiq_dura(self):
        '''计算若全部卖出核心流动性的资产，组合久期的下降幅度'''
        res_cols = ['C_FUNDNAME', 'D_DATE', '1级核心', '2级核心', '3级核心', '4级核心', '5级核心', 'PortDuration', 'PORTFOLIO_CODE']
        if self.liq_holdings.empty:
            return pd.DataFrame(columns=res_cols)

        if 'coreClass' not in self.liq_holdings.columns:
            self.classifyCoreLiquidity()

        data = self.liq_holdings.copy()
        res = pd.DataFrame(columns=res_cols)
        if any(data['LiqType'].isin(['债券'])):
            res_core = data.groupby(['C_FUNDNAME', 'D_DATE', 'coreClass']).apply(self._calcPortDuration).rename('CoreDuration').unstack('coreClass').reset_index()
            res_dura = data.groupby(['C_FUNDNAME', 'D_DATE']).apply(self._calcPortDuration).rename('PortDuration').reset_index()
            res = pd.merge(res_core, res_dura, on=['C_FUNDNAME', 'D_DATE'], how='left')
            res['PORTFOLIO_CODE'] = res['C_FUNDNAME'].map(self.fundname_to_code)
        return res.reindex(columns=res_cols)

    def calcCoreLiq_duraTday(self, interval=1):
        '''计算T日可变现的各核心流动性资产规模及卖出后久期的可能下降幅度'''
        if '%d日可变现'%interval not in self.liq_holdings.columns:
            self.getLiquidityAsset_t(interval)
        data = self.liq_holdings.copy()

        data = data[data['coreClass'].notna() & (data['L_STOCKTYPE'] == '2') & (data['F_MOUNT'] > 0)].copy()
        data['NewRatio'] = data['%d日可变现_张'%interval] * data['F_ASSETRATIO'] / data['F_MOUNT']
        res_core_dura = data.groupby(['C_FUNDNAME', 'D_DATE', 'coreClass']).apply(self._calcPortDuration, 'NewRatio').rename('CoreDuration_%d日'%interval).reset_index()
        res_core_ratio = data.groupby(['C_FUNDNAME', 'D_DATE', 'coreClass'])['%d日可变现'%interval].sum().rename('CoreRatio_%d日'%interval).reset_index()

        res = pd.merge(res_core_ratio, res_core_dura, on=['C_FUNDNAME', 'D_DATE', 'coreClass'], how='left')
        return res

    def getAllCoreLiq_dura(self, interval_list=[1, 2, 3, 5]):
        '''整合各T日可变现的各核心流动性资产规模及卖出后久期的可能下降幅度'''

        core_cols = []
        for i in interval_list:
            core_cols += ['CoreRatio_%d日' % i, 'CoreDuration_%d日' % i]
        res_cols = ['C_FUNDNAME', 'D_DATE', 'coreClass'] + core_cols + ['总资产', 'CONC', 'PORTFOLIO_CODE']

        if self.liq_holdings.empty:
            return pd.DataFrame(columns=res_cols)

        if 'coreClass' not in self.liq_holdings.columns:
            self.classifyCoreLiquidity()

        data = self.liq_holdings.dropna(subset=['coreClass'])
        if data.empty:
            return pd.DataFrame(columns=res_cols)

        res_core = data[['C_FUNDNAME', 'D_DATE', 'coreClass']].drop_duplicates()
        for interval in interval_list:
            res_temp = self.calcCoreLiq_duraTday(interval)
            res_core = pd.merge(res_core, res_temp, on=['C_FUNDNAME', 'D_DATE', 'coreClass'], how='left')

        data_fund = self.data_fund[['C_FUNDNAME', 'D_DATE', 'TotalAsset']].rename(columns={'TotalAsset': '总资产'})
        res_core = pd.merge(res_core, data_fund, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res_core['CONC'] = res_core[['C_FUNDNAME','coreClass']].sum(axis=1)
        res_core = res_core.sort_values(by=['C_FUNDNAME', 'D_DATE', 'coreClass'])
        res_core['PORTFOLIO_CODE'] = res_core['C_FUNDNAME'].map(self.fundname_to_code)

        return res_core.reindex(columns=res_cols)

    def getCertainIdx(self, interval=2):
        '''专门开给风险指标，包括T+2可变现和各级核心流动性规模'''
        if 'liq_time' not in dir(self):
            self.liq_time = self.calcLiquidTime()
        res_tDay, asset1, asset2 = self.calcLiquidity_pivot([interval])
        res_core = self.calcCoreLiquidity()

        data_fund = self.data_fund.reindex(columns=['PORTFOLIO_CODE', 'C_FUNDNAME', 'D_DATE', 'TotalAsset', 'NetAsset']).rename(columns={'TotalAsset': '总资产', 'NetAsset': '净资产'})
        res_2Day = res_tDay[['C_FUNDNAME', 'D_DATE', 'T日可变现总规模']].rename(columns={'T日可变现总规模': '%d日可变现'%interval})
        res = pd.merge(res_2Day, data_fund, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res = pd.merge(res, res_core.drop(columns='总资产'), on=['C_FUNDNAME', 'D_DATE'], how='left')
        cols_ratio = res.set_index(['C_FUNDNAME', 'D_DATE', '总资产', '净资产', 'PORTFOLIO_CODE']).columns
        for col in cols_ratio:
            res[col + '_Ratio'] = res[col] / res['净资产']
        
        if interval == 2:
            self.liq_2 = res_2Day

        return res

    def CalculateAll(self):
        self.calcLiqCore()
        self.insert2db_all()
        self.integrate_lr_holding()
        self.calc_liq_factor()

    def calcLiqCore(self):
        self.res_liq = self.calcAllLiquidation()
        self.res_tDay, self.asset1, self.asset2 = self.calcLiquidity_pivot()

        if 'res_core' not in dir(self):  # 在getCertainIdx函数中已经计算了该指标
            self.res_core = self.calcCoreLiquidity()
        self.res_secs = self.calcCoreLiq_secs()
        self.core_dura = self.calcCoreLiq_dura()
        self.core_levelAll = self.getAllCoreLiq_dura()
    
    def save_all(self, save_path, t):
        data_dict = {'到期前变现': self.res_liq, '变现个券明细': self.data_over1, '持仓明细': self.liq_holdings,
                     '非核心流动性1': self.res_tDay, '非核心流动性2': self.asset1, '非核心流动性3': self.asset2,
                     '核心流动性_规模': self.res_core, '核心流动性_证券': self.res_secs, '核心流动性_久期': self.core_dura,
                     '核心流动性_各级': self.core_levelAll, '核心流动性_个券': self.core_holdings}

        self.save_excel(folder_path=save_path, file_name='%s_LiquidityIndicators_core.xlsx'%t.replace('-', ''),
                        data_dict=data_dict, key_col='C_FUNDNAME', update=True, cover=False)

    def insert2db_all(self):
        self.insert2db('LiquidityIndicators_core', 'rc_lr_core_maturity', '到期前变现', self.res_liq, ptf_codes=self.ptf_codes)
        new_dataover = self.data_over1.reindex(columns=['C_FUNDNAME', 'D_DATE', 'code', 'C_SUBNAME_BSH', '变现天数'])
        new_dataover['PORTFOLIO_CODE'] = new_dataover['C_FUNDNAME'].map(self.fundname_to_code)
        self.insert2db('LiquidityIndicators_core', 'rc_lr_core_overduesecs', '变现个券明细', new_dataover, ptf_codes=self.ptf_codes)
        self.insert2db('LiquidityIndicators_core', 'rc_lr_core_asset', '非核心流动性1', self.res_tDay, ptf_codes=self.ptf_codes)
        self.insert2db('LiquidityIndicators_core', 'rc_lr_core_secs', '核心流动性_证券', self.res_secs, ptf_codes=self.ptf_codes)
        self.insert2db('LiquidityIndicators_core', 'rc_lr_core_durachg', '核心流动性_久期', self.core_dura, ptf_codes=self.ptf_codes)
        self.insert2db('LiquidityIndicators_core', 'rc_lr_core_all', '核心流动性_各级', self.core_levelAll, ptf_codes=self.ptf_codes)

    def integrate_lr_holding(self):
        '''流动性持仓明细'''
        self.calcCollateralRatio()
        data = self.liq_holdings.copy()
        if data.empty:
            logger.info('%s -- %s 无新增数据' % (self.basedate, 'dpe_lr_holdings'))
            return None
        data.loc[data['WINDL1TYPE'] == '金融债', 'indType'] = '金融债'
        data['indType'] = data['indType'].map({'产业债': '产业', '城投债': '城投', '金融债': '金融'})

        cols_keep = ['D_DATE', 'C_FULLNAME', 'C_FUNDNAME', 'C_SUBNAME_BSH', 'C_STOPINFO', 'code', 'F_MOUNT', 'F_ASSET',
                     'F_ASSETRATIO', '剩余期限', 'RATE_LATESTMIR_CNBD', 'market', 'LiqType', 'irbond_type', 'tfi',
                     'issMethod', 'indType', 'pmtSeq', 'corpAttr', 'clause', '1日可变现_张', '2日可变现_张', '3日可变现_张',
                     '5日可变现_张', '1日可变现', '2日可变现', '3日可变现', '5日可变现', '质押量_张', '质押市值', '受限数量',
                     '变现天数', '是否高流动性债券', '活跃度', 'coreClass']
        cols_map = {'剩余期限': 'ptm_year', 'RATE_LATESTMIR_CNBD': 'im_rating', '1日可变现_张': 'liq_amount_1',
                    '2日可变现_张': 'liq_amount_2', '3日可变现_张': 'liq_amount_3', '5日可变现_张': 'liq_amount_5',
                    '1日可变现': 'liq_asset_1', '2日可变现': 'liq_asset_2', '3日可变现': 'liq_asset_3',
                    '5日可变现': 'liq_asset_5', '质押量_张': 'collateral_amount', '质押市值': 'collateral_asset',
                    '受限数量': 'restr_amount', '变现天数': 'liq_days', '是否高流动性债券': 'hig_liq', '活跃度': 'activity',
                    'coreClass': 'core_class'}

        data = data.reindex(columns=cols_keep).rename(columns=cols_map)
        data['market'] = data['code'].apply(lambda x: x[-2:]).map({'IB': '银行间'}).fillna('交易所')
        data['corpAttr'] = data['corpAttr'].map(
            {'地方国有企业': '国企', '中央国有企业': '央企', '国有企业': '国企', '民营企业': '民企', '集体企业': '民企',
             '公众企业': '上市企业', '中外合资企业': '外资', '外商独资企业': '外资', '外资企业': '外资'})
        data['pmtSeq'] = data['pmtSeq'].replace('普通', '正常')
        data['clause'] = data['clause'].replace('普通', '正常')
        data['liq_days'] = data['liq_days'].astype(float)

        # 匹配产品代码
        data['portfolio_code'] = data['C_FUNDNAME'].map(self.fundname_to_code)

        # 债券：补充交易量及中债流动性评分数据
        bond_holdings = data[data['LiqType'] != '股票'].copy()
        bond_codes = bond_holdings['code'].unique().tolist()
        result = data.copy()
        if len(bond_codes) > 0:
            trans_cols = ['tdvolume_1m', 'tdvalue_1m', 'tdprice_1m', 'tdvolume_1m_i', 'tdvalue_1m_i']
            lqscore_cols = ['lqscore_cnbd', 'lqpct_cnbd']

            # transaction = self._get_bond_transaction(bond_holdings=bond_holdings, new_cols=trans_cols)
            lq_score = self._liq_score_cnbd(bond_codes=bond_codes, new_cols=lqscore_cols)
            result = data.merge(lq_score, how='left', on='code')

        result['F_MOUNT'] = result['F_MOUNT'].astype('str')
        self.insert2db_single('dpe_lr_holding', result, ptf_code=self.ptf_codes, code_colname='portfolio_code')

    # def _get_bond_transaction(self, bond_holdings: list, new_cols: list):
    #     '''个券及主体过去1个月的成交量'''
    #     jy_db = JYDB_Query()
    #     ib_bonds = bond_holdings[bond_holdings['market'] == '银行间']['code'].unique().tolist()
    #     ex_bonds = bond_holdings[bond_holdings['market'] == '交易所']['code'].unique().tolist()
    #     ib_bonds_jy = [i[:i.find(".")] if '.' in i else i for i in ib_bonds]
    #     ex_bonds_jy = [i[:i.find(".")] if '.' in i else i for i in ex_bonds]
    #
    #     # 找到持仓券主体发行的全部证券
    #     issuer_secs = jy_db.sec_query(sqls_config['bond_transaction']['bond_issuer'], sec_list=bond_codes)
    #     all_secs = issuer_secs['bond_code'].to_list()   # 持仓券主体发行的全部证券
    #     bond_info = wind_dq.sec_query(sqls_config['bond_transaction']['bond_info'], sec_list=all_secs)  # 获取债券相关信息
    #
    #     # 取数：近一月债券成交量、成交金额（全价）
    #     bg_date = self.get_offset_month(self.basedate, -1)
    #     sql_raw = sqls_config['bond_transaction']['period_transaction']
    #     trade_1m = wind_dq.sec_query(sql_raw, sec_list=all_secs, t0=bg_date.replace("-", ""), t1=self.basedate.replace("-", ""))
    #     trade_1m = trade_1m.merge(issuer_secs, how='left', on='bond_code').merge(bond_info, how='left', on='bond_code')
    #     # 原始数据是成交手数，交易所1手是10张，银行间1手是100张
    #     trade_1m['volume'] = trade_1m.apply(lambda x: x.volume * (10 if x.bond_mkt in ['SSE', 'SZSE'] else 100), axis=1)
    #
    #     # 纯债&转债：个券过去1个月的成交量、成交价格、成交均价
    #     cols_b = {'volume': 'tdvolume_1m', 'value': 'tdvalue_1m'}
    #     res = trade_1m.groupby('bond_code')[['volume', 'value']].sum().reset_index().rename(columns=cols_b)
    #     res['tdprice_1m'] = res.apply(lambda x: x.tdvalue_1m/x.tdvolume_1m if x.tdvolume_1m > 0 else np.nan, axis=1)
    #
    #     # 针对纯债计算：相同主体过去1个月的成交量、成交价格
    #     trade_1m_i = trade_1m[~trade_1m['windl1type'].isin(['可转债', '可交换债', '可分离转债存债', '资产支持证券'])].copy()
    #     cols_i = {'volume': 'tdvolume_1m_i', 'value': 'tdvalue_1m_i'}
    #     issuer_res = trade_1m_i.groupby('issuer_code')[['volume', 'value']].sum().reset_index().rename(columns=cols_i)
    #
    #     bond_to_issuer = issuer_secs.set_index('bond_code')['issuer_code'].to_dict()
    #     res['issuer_code'] = res.apply(lambda x: bond_to_issuer[x.code] if x.code in bond_to_issuer.keys() else None, axis=1)
    #     res = res.merge(issuer_res, how='left', on='issuer_code').drop('issuer_code', axis=1)
    #
    #     res = res.reindex(columns=['bond_code'] + new_cols).rename(columns={'bond_code': 'code'})
    #     return res

    def _liq_score_cnbd(self, bond_codes: list, new_cols: list):
        '''中债流动性指标'''
        jydq = JYDB_Query()
        sec_digit = [i[0:i.find(".")] for i in bond_codes]

        lq_score = jydq.sec_query('liquidity_score_cnbd', sec_digit, self.basedate)
        lq_score.columns = [i.lower() for i in lq_score]

        lq_score['mkt'] = lq_score['secumarket'].map({89: '.IB', 83: '.SH', 90: '.SZ', '72': '.HK'})
        lq_score['code'] = lq_score['secucode'] + lq_score['mkt']
        lq_score = lq_score[lq_score['code'].isin(bond_codes)].copy()

        return lq_score.reindex(columns=['code'] + new_cols)

    def _get_cr_high_volume(self):
        '''计算信用债全市场高流动性债券的成交量阈值，暂定8%'''
        coef_path = os.path.join(config['shared_drive_data']['liquidity_coef']['path'], 'CreditBond')
        t = self.basedate

        def __findLatestCoefs(coef_path, t):
            date_list = os.listdir(coef_path)
            date_list = [x for x in date_list if ('.' not in x) and ('Coef' not in x)]
            delta_list = np.array([(pd.to_datetime(x) - pd.to_datetime(t)).days for x in date_list])
            t = np.array(date_list)[delta_list <= 0][-1]
            return t

        t_target = __findLatestCoefs(coef_path, t)
        t_path = os.path.join(coef_path, t_target)
        vol_t = pd.read_excel(os.path.join(t_path, 'Volume_Threshold.xlsx'), engine='openpyxl')['Volume_Threshold(8%)(张)'].values[0] * self.part
        return vol_t

    def calc_lr_factor_cr(self, data):
        '''计算组合层面信用债在各个流动性因子上的排序情况，市值加权'''
        data_credit = data[data['LiqType'].isin(['ABS', '信用债'])].copy()
        if data_credit.empty:
            return None
        data_credit['im_rating_w'] = data_credit['im_rating'].map(
            {'AAA+': 1, 'AAA': 2, 'AAA-': 3, 'AA+': 4, 'AA': 5, 'AA(2)': 6, 'AA-': 7, 'A+': 8, 'A': 9, 'A-': 10})
        data_credit['ptm_w'] = data_credit['ptm_year'].copy()
        data_credit['corpAttr_w'] = data_credit['corpAttr'].map({'国企': 1, '央企': 2, '民企': 3, '外资': 4, '上市企业': 5})
        data_credit['indType_w'] = data_credit['indType'].map({'产业': 1, '城投': 2, '金融': 3})

        credit_gp = data_credit.set_index(['d_date', 'c_fundname'])
        credit_gp = credit_gp.iloc[:, -4:] * credit_gp['f_assetratio'].values.reshape(credit_gp.shape[0], 1) / 100

        self.res_credit = credit_gp.groupby(['d_date', 'c_fundname']).sum().reset_index()
        self.res_credit['portfolio_code'] = self.res_credit['c_fundname'].map(self.fundname_to_code)
        self.insert2db_single('rc_lr_port_cr', self.res_credit, ptf_code=self.ptf_codes, code_colname='portfolio_code')

    def calc_lr_factor_port(self, data):
        '''计算组合层面各资产的整体流动性情况，以1日可变现资产市值占该资产总仓位的比例来衡量'''
        data_other = data[~data['LiqType'].isin(['ABS', '信用债'])].copy()
        other_cols = ['c_fundname', 'd_date', 'ir_factor', 'cbond_factor', 'cd_factor', 'stock_factor']
        if data_other.empty:
            self.res_other = pd.DataFrame(columns=other_cols)
        else:
            data_other['f_mount'] = data_other['f_mount'].astype(float)
            res_other = data_other.groupby(['c_fundname', 'd_date', 'LiqType']).apply(
                lambda x: ((x['f_assetratio'] * x['liq_amount_1'] / x['f_mount']) / x['f_assetratio'].sum()).sum())
            res_other = res_other.unstack(level='LiqType').rename(columns={'利率债': 'ir_factor', '可转债': 'cbond_factor', '同业存单': 'cd_factor', '股票': 'stock_factor'})
            self.res_other = res_other.reset_index().reindex(columns=other_cols)

        vol_high = self._get_cr_high_volume()
        data_cr = data[data['LiqType'].isin(['ABS', '信用债'])].copy()
        cr_cols = ['c_fundname', 'd_date', 'cr_factor']
        if data_cr.empty:
            self.res_cr = pd.DataFrame(columns=cr_cols)
        else:
            data_cr['liq_w'] = (data_cr['liq_amount_1'] / vol_high).apply(lambda x: min(1, x))
            res_cr = data_cr.groupby(['c_fundname', 'd_date']).apply(lambda x: (x['f_assetratio'] * x['liq_w']).sum() / x['f_assetratio'].sum())
            self.res_cr = res_cr.reset_index().rename(columns={0: 'cr_factor'}).reindex(columns=cr_cols)

        self.res_factor = pd.merge(self.res_other, self.res_cr, on=['c_fundname', 'd_date'], how='outer')
        self.res_factor['portfolio_code'] = self.res_factor['c_fundname'].map(self.fundname_to_code)
        self.insert2db_single('rc_lr_factor_port', self.res_factor, ptf_code=self.ptf_codes, code_colname='portfolio_code')

    def calc_liq_factor(self):
        '''计算组合层面各资产及信用债各因子的流动性情况'''
        q = sqls_config['lr_holding']['Sql']%self.basedate
        data = self.db_risk.read_sql(q)

        if self.ptf_codes is not None:
            data = data[data['portfolio_code'].isin(self.ptf_codes)].copy()
        if data.empty:
            return None

        self.calc_lr_factor_cr(data)
        self.calc_lr_factor_port(data)


# if __name__ == '__main__':
#     t = '2020-09-24'
#     data_path = r'E:\RiskQuant\风险指标\Valuation\\'
#     save_path = r'E:\RiskQuant\风险指标\DailyIndicators\2020Q3\%s\\'%t.replace('-', '')
#     file_val = 'valuation' + t.replace('-', '') + '.json'
#     LiqIdx_core = LiquidityIndicators_core(data_path, file_val, save_path)
#     res_liq = LiqIdx_core.calcAllLiquidation()
#     res_tDay, asset1, asset2 = LiqIdx_core.calcLiquidity_pivot()
#     res_core = LiqIdx_core.calcCoreLiquidity()
#     res_secs = LiqIdx_core.calcCoreLiq_secs()
#     core_dura = LiqIdx_core.calcCoreLiq_dura()
#     core_levelAll = LiqIdx_core.getAllCoreLiq_dura()

#     writer = pd.ExcelWriter(save_path + '%s_LiquidityIndicators_core.xlsx'%t.replace('-', ''))
#     res_liq.to_excel(writer, sheet_name='到期前变现', index=False)
#     LiqIdx_core.data_over1.to_excel(writer, sheet_name='变现个券明细', index=False)
#     LiqIdx_core.holdings.to_excel(writer, sheet_name='持仓明细', index=False)
#     res_tDay.to_excel(writer, sheet_name='非核心流动性1', index=False)
#     asset1.to_excel(writer, sheet_name='非核心流动性2', index=False)
#     asset2.to_excel(writer, sheet_name='非核心流动性3', index=False)
#     res_core.to_excel(writer, sheet_name='核心流动性_规模', index=False)
#     res_secs.to_excel(writer, sheet_name='核心流动性_证券', index=False)
#     core_dura.to_excel(writer, sheet_name='核心流动性_久期', index=False)
#     core_levelAll.to_excel(writer, sheet_name='核心流动性_各级', index=False)
#     LiqIdx_core.core_holdings.to_excel(writer, sheet_name='核心流动性_个券', index=False)
#     writer.save()