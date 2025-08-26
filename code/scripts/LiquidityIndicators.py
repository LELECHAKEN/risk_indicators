'''
@Description: 流动性风险指标
@Author: Wangp
@Date: 2020-03-11 15:24:47
LastEditTime: 2021-06-15 15:57:45
LastEditors: Wangp
'''
import os
import numpy as np
import pandas as pd
from WindPy import w

from .utils_ri.RiskIndicators import RiskIndicators
from .settings import config, DIR_OF_MAIN_PROG
from .utils.log_utils import logger
from .db import OracleDB, column, sqls_config

import sys
sys.path.append(config['pkgs_custom']['path_1'])
sys.path.append(config['pkgs_custom']['path_2'])

from .Liquidity_Asset import Liquidity_Asset, Liquidity_collaterals


class LiquidityIndicators(RiskIndicators):
    def __init__(self, t, save_path, ptf_codes=None):
        self.basedate = t
        self.save_path = save_path
        self._format_ptf_codes(ptf_codes)
        self._InterestBond = ['政策银行债', '国债', '央行票据', '地方政府债', '政府支持机构债']

        self._loadFile()   # 取估值表数据
        self._load_portfolio_info()  # 产品相关信息

        self._loadHoldings(self.save_path)   # 加载持仓明细
        self._deal_stock_holdings()  # 处理股票持仓
        self._loadLiqHoldings()
        self._loadTableColumns()

    def _deal_stock_holdings(self):
        '''
        合并股票的持仓，基于同时持有正常交易和限售股票的组合会有两条同一只股票的持仓记录
        # TODO: 转融通也会有两条持仓记录，其中一条记在借出股票的科目下，尚未处理
        :return:
        '''
        data = self.stock_holdings.copy()
        if data.empty:
            logger.info('产品无股票持仓')
            return None

        def _simple_stat(x):
            amt = x['F_MOUNT'].sum()
            price = x[['F_PRICE', 'F_MOUNT']].product(axis=1).sum() / x['F_MOUNT'].sum()
            asset = x['F_ASSET'].sum()
            asset_r = x['F_ASSETRATIO'].sum()
            net_cost = x[['F_NETCOST', 'F_MOUNT']].product(axis=1).sum() / x['F_MOUNT'].sum()
            cost = x['F_COST'].sum()
            cost_r = x['F_COSTRATIO'].sum()
            stop_info = x['C_STOPINFO'].sum()
            return pd.DataFrame([amt, price, asset, asset_r, net_cost, cost, cost_r, stop_info], index=['F_MOUNT', 'F_PRICE', 'F_ASSET', 'F_ASSETRATIO', 'F_NETCOST', 'F_COST', 'F_COSTRATIO', 'C_STOPINFO']).T

        data_chg = data.groupby(['C_FUNDNAME', 'D_DATE', 'code']).apply(_simple_stat)
        data_unchg = data.reindex(columns=[x for x in data.columns if x not in data_chg.columns]).drop_duplicates()
        data_all = pd.merge(data_chg.reset_index(), data_unchg, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')

        self.stock_holdings = data_all.reindex(columns=data.columns)

    def _load_portfolio_info(self):
        # 赎回压力映射表相关信息
        redeem_path = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\赎回压力映射表.xlsx'
        self.redeem_map = pd.read_excel(redeem_path, sheet_name='赎回指标', engine='openpyxl')
        data_type = pd.read_excel(redeem_path, sheet_name='基金分类', engine='openpyxl').reindex(columns=['基金名称', '基金分类'])

        self._fundNameMapping()  # 产品基础信息：portfolio_type
        data_prod = self.ptf_info[self.ptf_info['type_l0'] == '专户'].copy()   # 专户产品
        data_prod.columns = [i.upper() for i in data_prod.columns]

        self._o32NameMapping()  # 获取产品的O32系统简称
        name_df = self._fundName.reindex(columns=['C_FULLNAME', 'C_FUNDNAME', 'C_FUNDNAME_o32'])
        self.data_prod = pd.merge(data_prod, name_df, how='left', on=['C_FULLNAME', 'C_FUNDNAME'])
        self.data_type = pd.merge(data_type, name_df, how='left', left_on=['基金名称'], right_on=['C_FUNDNAME_o32'])

    def _o32NameMapping(self):
        code_path = r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx'
        col_dict = {'产品名称': 'C_FULLNAME', '估值系统简称': 'C_FUNDNAME', 'O32产品名称': 'C_FUNDNAME_o32'}
        self._fundName = pd.read_excel(code_path, sheet_name='产品基础信息', engine='openpyxl').rename(columns=col_dict)
        self.dict_name = self._fundName.set_index('C_FUNDNAME')['C_FUNDNAME_o32'].to_dict()

    def _loadLiqHoldings(self):
        '''属性LiqAsset_d为字典：key为日期，value为Liquidity_Asset类'''
        self.liq_holdings = pd.concat([self.bond_holdings, self.stock_holdings], ignore_index=True)
        for col in ['F_ASSET', 'F_ASSETRATIO']:
            self.liq_holdings[col] = self.liq_holdings[col].astype('float')

        self.liq_holdings_t = {}
        self.LiqAsset_dict = {}

        self.liq_holdings = self._dealLiqAssetType(self.liq_holdings)  # 将持仓内各资产的分类规范至流动性模型划分的标准分类中
        for t in self.date_list:
            self.liq_holdings_t[t] = self.liq_holdings.loc[self.liq_holdings['D_DATE'] == t, :].copy()
            if self.liq_holdings_t[t].empty:
                logger.info('-- %s 产品无债券与股票持仓，暂不支持证券流动性情况计算' % t)
                continue
            self.LiqAsset_dict[t] = Liquidity_Asset(t, self.liq_holdings_t[t])
            self.LiqAsset_dict[t].calc_liquidity()    # 一次性计算各证券资产的流动性情况
            self.LiqAsset_dict[t].calc_collateral(ptf_codes=self.ptf_codes)    # 一次性计算证券资产的质押情况
            logger.info('-- %s liquidity Asset Initial Calculation Done.' % t)

    def getTimetoMaturity_s(self, fund, t):
        '''
        拉取给定专户的到期日及剩余到期天数字段\n
        :param fund: string, 产品的估值系统简称
        :param t: string, yyyy-mm-dd, 基期
        :return: DataFrame
        '''
        if type(fund) != list:
            fund = [fund]
        t = pd.to_datetime(t)
        maturityDate = self.data_prod.loc[self.data_prod['C_FUNDNAME'].isin(fund), ['C_FUNDNAME', 'END_DATE']]
        ttm = [(x - t).days for x in maturityDate['END_DATE']]
        maturityDate['剩余到期天数'] = ttm

        return maturityDate.rename(columns={'END_DATE': '产品到期日'})

    def _dealLiqAssetType(self, liq_holdings):
        '''将持仓内各资产的分类规范至流动性模型划分的标准分类中'''
        data = liq_holdings.copy()
        if data.empty:
            return data

        # 处理资产大类分类
        data['L_STOCKTYPE'] = data['L_STOCKTYPE'].astype(str)
        data['LiqType'] = data['L_STOCKTYPE'].map({'1': '股票', '2': '信用债'})
        data.loc[data['WINDL2TYPE'].isin(self._InterestBond), 'LiqType'] = '利率债'
        data.loc[data['WINDL2TYPE'].isin(['可转债', '可交换债', '可分离转债存债']), 'LiqType'] = '可转债'
        data.loc[data['WINDL1TYPE'] == '同业存单', 'LiqType'] = '同业存单'
        data.loc[data['WINDL1TYPE'] == '资产支持证券', 'LiqType'] = 'ABS'

        # 处理隐含评级
        data['RATE_LATESTMIR_CNBD'] = data['RATE_LATESTMIR_CNBD'].replace('评级_利率债', 'NR').replace('无评级', 'NR')

        # 处理铁道债和地方政府债：当作信用债，主评视为AAA+
        data.loc[data['WINDL1TYPE'].isin(['政府支持机构债', '地方政府债']), 'LiqType'] = '信用债'
        data.loc[data['WINDL1TYPE'].isin(['政府支持机构债', '地方政府债']), 'RATE_LATESTMIR_CNBD'] = 'AAA+'

        # 处理利率债分类，按发行主体分为：国债/国开/农发/进出
        dict_ir_issuer = {'中华人民共和国财政部': '国债', '中国进出口银行': '进出', '中国农业发展银行': '农发', '国家开发银行': '国开'}
        data['irbond_type'] = data['发行人名称'].map(dict_ir_issuer)
        data['tfi'] = data.apply(lambda x: (x['D_DATE'] - x['起息日期']).days/365 if x['起息日期'] is not np.nan else 0, axis=1)
        # data['tfi'] = [x.days / 365 for x in data['D_DATE'] - data['起息日期']]

        # 处理发行方式、产业or城投、偿还次序、含权条款
        data['issMethod'] = data['ISSUE_ISSUEMETHOD'].map({'公募': '公募', '私募': '私募'}).fillna('公募')
        data['indType'] = data['MUNICIPALBOND'].map({'是': '城投债'}).fillna('产业债')
        data['pmtSeq'] = data['是否次级债'].map({'是': '次级'}).fillna('普通')
        data['corpAttr'] = data['NATURE1'].fillna('其他企业')
        data['clause'] = data['永续'].fillna('普通')

        return data

    def getTTM_port(self, fund):
        '''遍历所有专户拉取实际到期日及剩余到期天数字段'''
        if type(fund) != list:
            fund = [fund]

        res = pd.DataFrame()
        for t in self.date_list:
            temp = self.getTimetoMaturity_s(fund, t)
            temp['D_DATE'] = t
            res = pd.concat([res, temp], sort=False)

        return res

    def getRedemptionIndice(self, fund):
        '''赎回监控值、赎回压力值'''
        if type(fund) != list:
            fund = [fund]
        temp = self.data_type.loc[self.data_type['C_FUNDNAME'].isin(fund), ['C_FUNDNAME', '基金分类']].copy()
        res0 = pd.merge(temp, self.redeem_map, on=['基金分类'], how='left').drop(columns=['基金分类'])

        # 处理货基的赎回监控值和赎回压力值，本来为30e和50e绝对值，需按资产净值转换为百分比
        monetaryFund = self.val.loc[self.val['L_FUNDKIND2'] == '5', 'C_FUNDNAME'].unique().tolist()
        data_fund = self.data_fund[['C_FUNDNAME', 'D_DATE', 'NetAsset']].copy()
        res1 = pd.merge(res0, data_fund, on=['C_FUNDNAME'], how='left')
        res1['是否货基'] = res1['C_FUNDNAME'].isin(monetaryFund).astype(int)
        res1['赎回监控值'] = res1['赎回监控值'] * (1 - res1['是否货基']) + res1['赎回监控值'] * 1e8 * res1['是否货基']/ res1['NetAsset']
        res1['赎回压力值'] = res1['赎回压力值'] * (1 - res1['是否货基']) + res1['赎回压力值'] * 1e8 * res1['是否货基']/ res1['NetAsset']

        res = res1.drop(columns=['NetAsset', '是否货基'])

        return res

    def getLeverageDist(self):
        '''按照原始期限计算的融资分布'''
        repo_all = self._repo_all.copy()
        if repo_all.shape[0] == 0:
            return pd.DataFrame(columns=['C_FUNDNAME', '1日', '2-7日', '7-14日', '14日以上', '1日_Ratio', '2-7日_Ratio', '7-14日_Ratio', '14日以上_Ratio'])

        repo_all['balance_real'] = repo_all['direction_num'] * repo_all['amount']
        repo_all['repo_days_gp'] = pd.cut(repo_all['repo_days'], bins=[0, 1, 7, 14, np.inf], labels=['1日', '2-7日', '7-14日', '14日以上'], right=True)
        res = pd.pivot_table(repo_all, index='C_FUNDNAME', columns='repo_days_gp', values='balance_real', aggfunc='sum', margins=True)

        for col in res.columns[:-1]:
            res[col+'_Ratio'] = res[col] / res['All']
        res = res.drop(columns=['All']).reset_index()
        res = res[res['C_FUNDNAME'] != 'All'].copy()

        return res

    def getLeverageDist_ttm(self):
        '''按照剩余期限计算的融资分布'''
        repo_all = self._repo_all.copy()
        if repo_all.shape[0] == 0:
            return pd.DataFrame(
                columns=['D_DATE', 'C_FUNDNAME', '1日', '2-7日', '7-14日', '14日以上', '1日_Ratio', '2-7日_Ratio',
                         '7-14日_Ratio', '14日以上_Ratio'])

        # 计算剩余期限
        repo_all['repo_days'] = [x.days for x in repo_all['buyback_date_real'] - pd.to_datetime(self.basedate)]
        repo_all['balance_real'] = repo_all['direction_num'] * repo_all['amount']
        repo_all['repo_days_gp'] = pd.cut(repo_all['repo_days'], bins=[0, 1, 7, 14, np.inf],
                                     labels=['repo_ttm_1d', 'repo_ttm_2to7', 'repo_ttm_7to14', 'repo_ttm_14'], right=True)
        res = pd.pivot_table(repo_all, index='C_FUNDNAME', columns='repo_days_gp', values='balance_real', aggfunc='sum',
                             margins=True).reindex(columns=['repo_ttm_1d', 'repo_ttm_2to7', 'repo_ttm_7to14', 'repo_ttm_14', 'All'])

        for col in res.columns[:-1]:
            res[col + '_Ratio'] = res[col] / res['All']
        res = res.drop(columns=['All']).reset_index()

        res = res[res['C_FUNDNAME'] != 'All'].copy()
        # 分期限计算融资成本
        repo_df = repo_all[repo_all['direction_num'] == -1].copy()
        if not repo_df.empty:
            bins = [0, 1, 7, 14, np.inf]
            labels = ['repo_rate_1d', 'repo_rate_2to7', 'repo_rate_7to14', 'repo_rate_14']
            repo_df['repo_rate_gp'] = pd.cut(repo_df['repo_days'], bins=bins, labels=labels, right=True).astype('str')
            rate_res = repo_df.groupby(['C_FUNDNAME', 'repo_rate_gp']).apply(
                lambda x: np.average(x['interest_rate'], weights=x['balance_real'])).reset_index()
            res2 = pd.pivot_table(rate_res, index='C_FUNDNAME', columns='repo_rate_gp', values=0).reset_index()
            res = res.merge(res2, how='left', on='C_FUNDNAME')

        res['D_DATE'] = pd.to_datetime(self.basedate)
        res['PORTFOLIO_CODE'] = res['C_FUNDNAME'].map(self.fundname_to_code)
        return res

    def getCashRepoLiq(self, t, interval=1):
        '''
        计算T日可变现的逆回购和现金资产\n
        :param t: string, yyyy-mm-dd, 基期
        :param interval: int, 日期偏移量，如1日可变现、2日可变现等
        :return: DataFrame
        '''
        data_cash = self.data_fund[['D_DATE', 'C_FUNDNAME', 'Deposit', 'TotalAsset']].copy()
        data_cash['Deposit'] = data_cash['Deposit'] * data_cash['TotalAsset']

        data_repo = self._repo_all.copy()
        if data_repo.empty:
            repo_amt = pd.DataFrame(columns=['C_FUNDNAME', '%d日逆回购'%interval, 'D_DATE'])
        else:
            t1 = pd.to_datetime(self.get_offset_tradeday(t, interval))  # 获取未来第interval个交易日
            repo_amt = data_repo[(data_repo['buyback_date_legal'] <= t1) & (data_repo['direction'] == '融券回购')].groupby(['C_FUNDNAME'])['amount'].sum() * 100
            repo_amt = repo_amt.rename('%d日逆回购'%interval).reset_index()
            repo_amt['D_DATE'] = t

        res = pd.merge(data_cash[['D_DATE', 'C_FUNDNAME', 'Deposit']], repo_amt, on=['D_DATE', 'C_FUNDNAME'], how='left')
        res['%d日逆回购+现金'%interval] = res[['Deposit', '%d日逆回购'%interval]].sum(axis=1)

        return res

    def calcPortLiquidity(self):
        '''计算组合层面的整体1日、5日流动性情况'''
        data = self.liq_holdings.copy()

        cash_1 = pd.DataFrame()
        cash_5 = pd.DataFrame()
        liq_1 = pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'code', 'liq_amt1', '1日可变现_张'])
        liq_5 = pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'code', 'liq_amt5', '5日可变现_张'])
        for t in self.date_list:
            # 计算1日、5日可变现的逆回购+现金规模
            cash1_temp = self.getCashRepoLiq(t, interval=1)
            cash5_temp = self.getCashRepoLiq(t, interval=5)
            cash_1 = pd.concat([cash_1, cash1_temp], sort=False)
            cash_5 = pd.concat([cash_5, cash5_temp], sort=False)

            # 计算持仓资产的1日、5日可变现规模
            if self.liq_holdings_t[t].empty:
                continue
            cal = self.LiqAsset_dict[t]
            liq_1_temp = cal.calc_liquidity_Tday(t=1)
            liq_5_temp = cal.calc_liquidity_Tday(t=5)
            liq_1 = pd.concat([liq_1, liq_1_temp], sort=False)
            liq_5 = pd.concat([liq_5, liq_5_temp], sort=False)

        data = pd.merge(data, liq_1, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')
        data = pd.merge(data, liq_5, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')
        data['1日可变现'] = data['1日可变现_张'] * data['F_PRICE'].astype(float)
        data['5日可变现'] = data['5日可变现_张'] * data['F_PRICE'].astype(float)

        self.liq_holdings = data.copy()

        if data.empty:
            res_1 = cash_1.rename(columns={'1日逆回购+现金': '1日可变现'}).drop(columns=['Deposit', '1日逆回购'])
            res_5 = cash_5.rename(columns={'5日逆回购+现金': '5日可变现'}).drop(columns=['Deposit', '5日逆回购'])
            return res_1, res_5

        grouped = data.groupby(['C_FUNDNAME', 'D_DATE'])
        res_1 = grouped['1日可变现'].sum().reset_index()
        res_5 = grouped['5日可变现'].sum().reset_index()

        res_1 = pd.merge(res_1, cash_1, on=['D_DATE', 'C_FUNDNAME'], how='right')
        res_5 = pd.merge(res_5, cash_5, on=['D_DATE', 'C_FUNDNAME'], how='right')
        res_1['1日可变现'] = res_1[['1日可变现', '1日逆回购+现金']].sum(axis=1)
        res_5['5日可变现'] = res_5[['5日可变现', '5日逆回购+现金']].sum(axis=1)

        return res_1.drop(columns=['Deposit', '1日逆回购', '1日逆回购+现金']), res_5.drop(columns=['Deposit', '5日逆回购', '5日逆回购+现金'])

    def calcCollateralRatio(self):
        '''计算质押券占比'''
        data = self.liq_holdings.copy()
        if data.empty:
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', '质押市值'])

        liq_col = pd.DataFrame()
        for t in self.date_list:
            if self.liq_holdings_t[t].empty:
                continue
            cal = self.LiqAsset_dict[t]
            col_temp = cal.calc_collateral()
            col_temp['D_DATE'] = t
            liq_col = pd.concat([liq_col, col_temp], sort=False)

        data = pd.merge(data, liq_col, left_on=['C_FUNDNAME', 'D_DATE', 'code'], right_on=['C_FUNDNAME', 'D_DATE', '证券代码'], how='left')
        data['质押量_张'] = data['受限数量'].fillna(0)
        data['质押市值'] = data['质押量_张'] * data['F_PRICE'].astype(float)
        res = data.groupby(['C_FUNDNAME', 'D_DATE'])['质押市值'].sum().reset_index()

        self.liq_holdings = data.copy()

        return res

    def calcLiquidTime(self):
        '''计算组合预计清仓时间，即变现天数，不考虑券的质押情况'''
        if self.liq_holdings.empty:
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', '变现天数'])

        if 'liq_amt1' not in self.liq_holdings.columns:
            self.liq_1, self.liq_5 = self.calcPortLiquidity()

        data = self.liq_holdings.copy()
        data['单日变现量（不考虑质押）_张'] = data['liq_amt1'].replace(0, np.nan)
        data['变现天数'] = data['F_MOUNT'] / data['单日变现量（不考虑质押）_张']

        # 处理新股的变现天数
        data.loc[(data['C_STOPINFO'] == '【无行情】') & (data['L_STOCKTYPE'] == '1'), '变现天数'] = 20
        data.loc[(data['单日变现量（不考虑质押）_张'] == 0) & (data['L_STOCKTYPE'] == '2'), '变现天数'] = 7

        res = data.groupby(['C_FUNDNAME', 'D_DATE'])['变现天数'].max().reset_index()
        self.liq_holdings = data.copy()

        return res

    def getOverdueSecs(self):
        '''计算到期日小于90天的专户，拉出个券变现天数大于产品到期日的个券名单'''
        if 'ttm' not in dir(self):
            self.ttm = self.getTTM_port(self.fund_list)
        if 'liq_time' not in dir(self):
            self.liq_time = self.calcLiquidTime()

        funds = self.ttm.loc[self.ttm['剩余到期天数'] <= 90, 'C_FUNDNAME'].tolist()
        data = self.liq_holdings[self.liq_holdings['C_FUNDNAME'].isin(funds)].copy()
        if data.empty:
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'Overdue_Secs'])
        data = pd.merge(data, self.ttm, on=['C_FUNDNAME', 'D_DATE'], how='left')
        data['liq_end_date'] = pd.to_datetime(data['REPURCHASEDATE_wind']).fillna(value=data['到期日期'])

        res = data[(data['变现天数'] > data['剩余到期天数']) & (data['liq_end_date'] > data['产品到期日'])].groupby(['C_FUNDNAME', 'D_DATE'])['C_SUBNAME_BSH'].apply(lambda x: ','.join(list(x))).rename('Overdue_Secs').reset_index()
        return res

    def calcAssetLeverageSpace(self, mod, data):
        '''计算个券可加杠杆空间，单位为张数'''
        fundname = self.dict_name[data['C_FUNDNAME']]
        assetname = data['C_SUBNAME_BSH']
        asset_amt = data['F_MOUNT']
        impRat = data['RATE_LATESTMIR_CNBD']

        lev_spc = mod.calc_leverage_space(fundname,assetname,asset_amt,impRat)

        return lev_spc

    def calcLeverageSpace(self):
        '''计算组合预计可加杠杆空间'''
        if self.liq_holdings.empty:
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', '可加杠杆_市值'])

        lev_spc = pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'code', '受限数量', '可加杠杆_张'])
        for t in self.date_list:
            if self.liq_holdings_t[t].empty:
                continue
            mod = self.LiqAsset_dict[t]
            lev_spc_temp = mod.calc_leverage_space()
            lev_spc = pd.concat([lev_spc, lev_spc_temp], sort=False)

        data = self.liq_holdings.drop(columns=['受限数量'])if '受限数量' in self.liq_holdings.columns else self.liq_holdings.copy()
        data = pd.merge(data, lev_spc, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')
        data['可加杠杆_市值'] = data['可加杠杆_张'] * data['F_PRICE'].astype(float)
        res = data.groupby(['C_FUNDNAME', 'D_DATE'])['可加杠杆_市值'].sum().reset_index()

        self.liq_holdings = data.copy()

        return res

    def calcHighLiqBond(self):
        '''计算高流动性资产占比'''
        data = self.liq_holdings.copy()
        if data.empty:
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'HighLiqBond'])

        t = data['D_DATE'].iloc[0]
        cal = self.LiqAsset_dict[t]
        sec_list = cal.calcHighLiqBond()

        data.loc[data['code'].isin(sec_list), '是否高流动性债券'] = '是'
        res = data[data['是否高流动性债券'] == '是'].groupby(['C_FUNDNAME', 'D_DATE'])['F_ASSETRATIO'].sum().rename('HighLiqBond') / 100
        res = res.reset_index()

        self.liq_holdings = data.copy()

        return res

    def CalculateAll(self):
        # 组合层指标，与持仓明细无关
        self.basic = self.data_fund.reindex(columns=['PORTFOLIO_CODE', 'C_FUNDNAME', 'D_DATE', 'NetAsset', 'TotalAsset'])
        self.ttm = self.getTTM_port(self.fund_list)
        self.redeem = self.getRedemptionIndice(self.fund_list)
        self.repo_dist = self.getLeverageDist()

        self.liq_1, self.liq_5 = self.calcPortLiquidity()
        self.colRatio = self.calcCollateralRatio()
        self.liq_time = self.calcLiquidTime()
        self.lev_spc = self.calcLeverageSpace()
        self.over_sec = self.getOverdueSecs()

        temp = self.val[['C_FUNDNAME', 'L_FUNDTYPE']].drop_duplicates()
        temp['杠杆上限'] = temp['L_FUNDTYPE'].map({'1': 1.4, '3': 2})

        res = self.IntegrateAll()
        res['杠杆率'] = res['TotalAsset'] / res['NetAsset']
        res = pd.merge(res, temp[['C_FUNDNAME', '杠杆上限']], on=['C_FUNDNAME'], how='left')
        res['剩余杠杆空间'] = res['杠杆上限'] - res['杠杆率']

        cols = ['1日可变现', '5日可变现', '质押市值', '可加杠杆_市值']
        for col in cols:
            if col in res.columns:
                res[col+'_Ratio'] = res[col] / res['NetAsset']

        if '可加杠杆_市值_Ratio' in res.columns:
            res['预估可加杠杆空间'] = res[['剩余杠杆空间', '可加杠杆_市值_Ratio']].min(axis=1)
        else:
            res['预估可加杠杆空间'] = res['剩余杠杆空间']
        res = res.drop(columns=['NetAsset', 'TotalAsset'])

        self.highLiqBond = self.calcHighLiqBond()
        res = pd.merge(res, self.highLiqBond, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res = res.reindex(columns=self.column_dict['liquidity'])

        self.insert_repo_ttm()

        return res

    def insert_repo_ttm(self):
        self.repo_ttm_dist = self.getLeverageDist_ttm()
        if self.repo_ttm_dist.empty:
            logger.info('%s -- rc_lr_repottm 无新增数据' % self.basedate)
            return None
        self.insert2db_single('rc_lr_repottm', self.repo_ttm_dist, ptf_code=self.ptf_codes, code_colname='PORTFOLIO_CODE')

    def supply_for_rc_liquidity(self):
        repo_all = self._repo_all.copy()
        repo_df = repo_all[repo_all['direction_num'] == -1].copy()  # 融资回购
        if repo_df.empty:
            return None

        # 分期限计算融资成本
        repo_df['balance_real'] = repo_df['direction_num'] * repo_df['amount']
        bins = [0, 1, 7, 14, np.inf]
        labels = ['repo_rate_1d', 'repo_rate_2to7', 'repo_rate_7to14', 'repo_rate_14']
        repo_df['repo_rate_gp'] = pd.cut(repo_df['repo_days'], bins=bins, labels=labels, right=True).astype('str')
        rate_res = repo_df.groupby(['C_FUNDNAME', 'repo_rate_gp']).apply(
            lambda x: np.average(x['interest_rate'], weights=x['balance_real'])).reset_index()
        res = pd.pivot_table(rate_res, index='C_FUNDNAME', columns='repo_rate_gp', values=0)
        res = res.reindex(columns=labels).reset_index().rename(columns={'C_FUNDNAME': 'c_fundname'})

        q = "select * from rc_liquidity where d_date='{t}'".format(t=self.basedate)
        data = self.db_risk.read_sql(q).drop(columns=['insert_time'] + labels)
        if self.ptf_codes is not None:
            data = data[data['portfolio_code'].isin(self.ptf_codes)].copy()
        res = data.merge(res, how='left', on='c_fundname')
        res['portfolio_code'] = res['c_fundname'].map(self.fundname_to_code)
        self.insert2db_single('rc_liquidity', res, t=self.basedate, ptf_code=self.ptf_codes)

# if __name__ == "__main__":
#     data_path = r'E:\RiskQuant\风险指标\\'
#     file_val = 'valuation3.json'
#     LiqIdx = LiquidityIndicators(data_path, file_val)
#     basic = LiqIdx.getBasicData()
#     ttm = LiqIdx.getTTM_port(LiqIdx.fund_list)
#     redeem = LiqIdx.getRedemptionIndice(LiqIdx.fund_list)
#     liq_1, liq_5 = LiqIdx.calcPortLiquidity()
#     colRatio = LiqIdx.calcCollateralRatio()
#     liq_time = LiqIdx.calcLiquidTime()
#     lev_spc = LiqIdx.calcLeverageSpace()
    
#     res = LiqIdx.CalculateAll()
#     res.to_excel(LiqIdx.data_path + 'LiquidityIndex2019Q3.xlsx', index=False)

#     print('Done.')