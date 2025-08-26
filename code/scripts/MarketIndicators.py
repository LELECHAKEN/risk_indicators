'''
@Description: Market Indicators.
@Author: Wangp
@Date: 2020-03-10 13:41:21
LastEditTime: 2022-12-06 10:56:45
LastEditors: Wangp
'''
import os
import datetime
import numpy as np
import pandas as pd
from WindPy import w
w.start()

import cx_Oracle as cx
from .utils.log_utils import logger
from .utils_ri.KeyYears import calcKeyYearRatio_n
from .utils_ri.Calc_AssetVaR import model_VaR
from .utils_ri.RiskIndicators import RiskIndicators
from .settings import config
from .db import OracleDB, sqls_config, column
from .db.db_utils import JYDB_Query
# from .demo_code import insert_table, delete_table
from .utils_ri.var import calc_var_parameter
from .utils_ri.IndexFund_deviation import IndexFundIndicators
from dateutil.relativedelta import relativedelta


class MarketIndicators(RiskIndicators):
    def __init__(self, t, save_path, ptf_codes=None):
        '''
        市场风险指标\n
        :param t: string, 计算基期
        :param save_path: string, 中间数据的存档路径
        '''
        # RiskIndicators.__init__(self, t)
        self.basedate = t
        self.save_path = save_path
        self._format_ptf_codes(ptf_codes)

        self.key_years = [-1, 0, 0.5, 1, 3, 5, 7, 10, 30, 50, np.inf]
        self._fundName = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl').rename(columns={'产品名称': 'C_FULLNAME', '估值系统简称': 'C_FUNDNAME', 'O32产品名称': 'C_FUNDNAME_o32'})
        # self._fundName = self._fundName[['C_FULLNAME', 'C_FUNDNAME', 'C_FUNDNAME_o32']].copy()

        self._loadFile()  # 取估值相关的持仓数据:self.val（进行产品筛选）
        self._loadHoldings(self.save_path)
        self._loadNavData()  # 取单位净值数据: self.nav_f（基于self.val，结果已进行产品筛选）
        self._loadBenchmarkReturn()
        self.__formatBenchmark() # 将比较基准相关信息合并至self.nav_f
        self._fundNameMapping()

        self.startDate = self.nav_f['D_DATE'].min()
        self.endDate = self.nav_f['D_DATE'].max()
        self.Equity_funds = ['宁私混合一期', '永赢宁私沪港深一期', '永赢基金精选五期']
        self.short_dura_bond_funds = ['永赢迅利中高等级短债', '永赢开泰中高等级中短债']

    def define_key_years(self, key_years):
        self.key_years = key_years

    def _dealNavData(self, startdate):
        '''取单位净值数据：公募取今年以来，专户取当前考核期'''
        if '_bchTarget' not in dir(self):
            self._loadBenchMarkTarget()
        nav = self.nav_f.copy()
        bch_m = self.val.loc[self.val['L_FUNDTYPE'].isin(['1']), ['C_FUNDNAME']].drop_duplicates()   # 公募取今年以来
        bch_m['考核期开始日'] = pd.to_datetime(startdate)
        bch_m['考核期结束日'] = pd.to_datetime(self.basedate)
        bchTarget = self._bchTarget.append(bch_m, sort=False)                                   # 专户取考核期

        data = pd.merge(nav, bchTarget, on='C_FUNDNAME', how='left')
        data = data.loc[(data['D_DATE'] >= data['考核期开始日']) &
                        (data['D_DATE'] <= data['考核期结束日']) &
                        (data['考核期结束日'] >= self.basedate), :].copy()

        return data

    def _loadBenchmarkReturn(self, benchmark_type='自定义'):
        q = sqls_config['benchmark_return']['Sql']
        benchmark_ret = self.db_risk.read_sql(q)[['c_date', 'fund_name', 'benchmark_type', 'return']]
        benchmark_ret.columns = ['D_DATE', 'C_FUNDNAME_o32', 'bch_type', 'ret_b']
        benchmark_ret['D_DATE'] = pd.to_datetime(benchmark_ret['D_DATE'])

        self._benchmark_ret = benchmark_ret[benchmark_ret['bch_type'] == benchmark_type].copy()

    def __formatBenchmark(self):
        data = pd.merge(self.nav_f, self._fundName[['C_FULLNAME', 'C_FUNDNAME', 'C_FUNDNAME_o32']], on=['C_FUNDNAME'], how='left')
        # data['bch_code'] = data['bch_code'].fillna('bond_general')
        data = pd.merge(data, self._benchmark_ret, on=['D_DATE', 'C_FUNDNAME_o32'], how='left')

        self.nav_f = data.copy()
        
    def _getBenchmarkReturn(self, benchmarkCode='000001.SH', startDate='', endDate=''):
        # benchmarkCode可含市场后缀，也可不含；可为string，也可为list
        if startDate == '':
            startDate = self.startDate
        if endDate == '':
            endDate = self.endDate

        temp = w.wsd(benchmarkCode, "close,pre_close", startDate, endDate, "PriceAdj=F")
        y = pd.DataFrame(temp.Data, index=temp.Fields, columns=temp.Times).T.reset_index().rename(columns={'index': 'D_DATE'})
        y['D_DATE'] = pd.to_datetime(y['D_DATE'])
        y['ret_idx'] = y['CLOSE'] / y['PRE_CLOSE'] - 1

        return y

    def _loadIndexFundsFile(self, t):
        '''指数基金跟踪偏离度'''
        indexfund = IndexFundIndicators(t=t)
        res = indexfund.index_fund_deviation()
        result = res[["fund_name", "a_ann_dev", "c_ann_dev", "a_avg_dev_abs", "c_avg_dev_abs"]].copy()
        result.columns = ['C_FUNDNAME', 'A份额年化跟踪误差', 'C份额年化跟踪误差', 'A份额日均跟踪偏离度绝对值', 'C份额日均跟踪偏离度绝对值']
        result.index = list(range(len(result)))
        return result

    def previous_tradeday(self, t):
        '''取当前日期的前一个交易日'''
        sql_raw = sqls_config['previous_trade_date']['Sql']
        prev_td = self.db_risk.read_sql(sql_raw.format(t=t))['c_date'].iloc[0]
        return prev_td

    def Indexfund_BchPerformance(self, t=""):
        # 基础表: DPE_indexfund_bchpf

        # 指数基金基础信息
        t = self.basedate if t == "" else t
        date_t = datetime.datetime.strptime(t, "%Y-%m-%d")
        bch_info = self.db_risk.read_sql(sqls_config["index_fund_bch"]["sql"])
        bch_info['if_95'] = bch_info.apply(lambda x: False if x.c_fundtype == '债券指数' or ('ETF' in x.c_fundname and '联接' not in x.c_fundname) else True, axis=1)
        bch_info["c_date"] = t

        # 获取基准指数的净值数据
        bg_oneyear = (datetime.datetime(date_t.year - 1, date_t.month, date_t.day) + relativedelta(days=1)).strftime("%Y-%m-%d")
        t0 = self.previous_tradeday(bg_oneyear)
        code_list = list(set(bch_info["bch_wealth_code"].to_list()))
        indexfund = IndexFundIndicators(t=t)
        nav_df = indexfund.bm_index_close(code_list=code_list, t0=t0, t1=t)
        nav_df_oneyear = nav_df.sort_index()
        td_eoy = self.get_latest_tradeday(str(int(t[:4])-1) + "-12-31")
        nav_df_thisyear = nav_df_oneyear[nav_df_oneyear.index >= td_eoy].copy()

        for idx, info in bch_info.iterrows():
            bch_code = info["bch_wealth_code"]

            # 今年以来（新成立产品取成立以来）
            bg_date = self.get_offset_tradeday(info['pos_set_end'] if pd.notna(info['pos_set_end']) else info['setup_date'], -1)
            nav_ty = nav_df_thisyear[nav_df_thisyear.index >= bg_date][bch_code].to_frame(name="NAV")
            if len(nav_ty)==0:
                continue
            bch_info.loc[idx, "crtdd_thisyear"] = self.DrawDown(nav_ty)
            bch_info.loc[idx, "maxdd_thisyear"] = self.MaxDrawDown(nav_ty)
            nav_ty['ret_daily'] = nav_ty["NAV"].pct_change()
            bch_info.loc[idx, "return_daily"] = nav_ty["ret_daily"].iloc[-1]
            if info["if_95"]:
                bch_info.loc[idx, "return_thisyear"] = (nav_ty['ret_daily'] * 0.95 + 1).cumprod().iloc[-1] - 1
            else:
                bch_info.loc[idx, "return_thisyear"] = nav_ty["NAV"].iloc[-1] / nav_ty["NAV"].iloc[0] - 1

            # 过去1年
            nav_oneyear = nav_df_oneyear[bch_code].to_frame(name="NAV")
            bch_info.loc[idx, "return_oneyear"] = nav_oneyear["NAV"].iloc[-1] / nav_oneyear["NAV"].iloc[0] - 1
            bch_info.loc[idx, "crtdd_oneyear"] = self.DrawDown(nav_oneyear)
            bch_info.loc[idx, "maxdd_oneyear"] = self.MaxDrawDown(nav_oneyear)

        self.insert2db_single('DPE_INDEXFUND_BCHPF', bch_info.drop(columns=['setup_date', 'if_95','pos_set_end']), t=t, t_colname='c_date')

    def Leverage(self):
        '''杠杆水平'''
        res = self.data_fund[['D_DATE', 'C_FUNDNAME', '卖出回购', 'NetAsset', 'TotalAsset']].copy()
        res['杠杆率'] = res['卖出回购'].fillna(0) * res['TotalAsset'] / res['NetAsset'] + 1
        res = res.drop(columns=['卖出回购', 'NetAsset', 'TotalAsset'])
        
        return res

    def LeverageCost(self):
        '''加权杠杆成本'''
        if self._lev_all.empty:
            return pd.DataFrame(columns=['D_DATE', 'C_FUNDNAME', 'Lev_cost'])

        res = self._lev_all.reindex(columns=['D_DATE', 'C_FUNDNAME', 'Lev_cost'])
        res['Lev_cost'] = res['Lev_cost'] * 100
        return res

    def avgYield_static(self, data_holding):
        '''组合加权平均静态收益率'''
        if data_holding.empty:
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'avgYield_static_orig', 'avgYield_static'])
        res1 = data_holding.groupby(['C_FUNDNAME', 'D_DATE']).apply(lambda x: (x['YIELD_CNBD'] * x['F_ASSETRATIO']).sum() / 100).rename('avgYield_static_orig').reset_index()
        res2 = self.Leverage()
        res3 = self.LeverageCost()

        res = pd.merge(res1, res2, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res = pd.merge(res, res3, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res['avgYield_static'] = res['avgYield_static_orig'] - res['Lev_cost'].fillna(0) * (res['杠杆率'] - 1)
        res = res.drop(columns=['Lev_cost', '杠杆率'])

        return res

    def avgCoupon_bond(self, data_holding):
        '''债券部分平均票息，浮息债取当期票息，可转债有票息'''
        res = (data_holding['COUPONRATE3'] * data_holding['F_ASSET']).sum() / data_holding['F_ASSET'].sum()

        return res
    
    def avgDuration_bond(self, data_holding):
        '''债券平均久期，可转债无久期'''
        res = (data_holding['Duration'] * data_holding['F_ASSET']).sum() / data_holding['F_ASSET'].sum()

        return res

    def avg_ptm_bond(self, data_holding):
        '''债券部分平均剩余期限，单位为年'''
        res = data_holding.groupby(['D_DATE', 'C_FUNDNAME']).apply(
            lambda x: (x['F_ASSET']*x['PTMYEAR']).sum()/x['F_ASSET'].sum()).rename('avgptm_bond')
        return res

    def avg_ptm_port(self, data_holding):
        '''组合平均剩余期限，以市值占净值比为权重，可转债无剩余期限'''
        ptm_bond = self.avg_ptm_bond(data_holding)
        res = pd.merge(ptm_bond.reset_index(), self.data_fund, on=['D_DATE', 'C_FUNDNAME'], how='left')
        res['avgptm_port'] = res['avgptm_bond'] * res['债券'].fillna(0) * res['TotalAsset'] / res['NetAsset']
        return res.reindex(columns=['D_DATE', 'C_FUNDNAME', 'avgptm_bond', 'avgptm_port'])

    def avgCoupon_port(self, data_holding):
        '''组合平均票息，以市值占净值比为权重，可转债有票息'''
        idx = data_holding.index[0]
        data_fund = self.data_fund.copy()
        data_fund['债券'] = data_fund['债券'] * data_fund['TotalAsset'] / data_fund['NetAsset']
        data_fund['可转债'] = data_fund['可转债'] * data_fund['TotalAsset'] / data_fund['NetAsset']

        lev = data_fund.set_index(['C_FUNDNAME', 'D_DATE']).loc[idx, ['债券', '可转债']].sum()
        res = self.avgCoupon_bond(data_holding) * lev

        return res
    
    def avgDuration_port(self, data_holding):
        '''组合平均久期，以市值占净值比为权重，可转债无久期'''
        idx = data_holding.index[0]
        data_fund = self.data_fund.copy()
        data_fund['债券'] = data_fund['债券'].fillna(0) * data_fund['TotalAsset'] / data_fund['NetAsset']
        # data_fund['可转债'] = data_fund['可转债'] * data_fund['TotalAsset'] / data_fund['NetAsset']
        
        lev = data_fund.set_index(['C_FUNDNAME', 'D_DATE']).loc[idx, ['债券']].sum()
        res = self.avgDuration_bond(data_holding) * lev

        return res

    def _dealAssetVaR(self, data_holding=''):
        '''匹配各资产的VaR阈值'''
        if type(data_holding) == str:
            data_holding = self.bond_holdings.copy()
            
        def _dealAssetType(data_holding):
            data_holding['AssetType_VaR'] = data_holding['利率or信用'].copy()
            # data_holding.loc[data_holding['L_STOCKTYPE'] == '1', 'AssetType_VaR'] = '股票'
            data_holding.loc[data_holding['WINDL1TYPE'] == '资产支持证券', 'AssetType_VaR'] = 'ABS'
            data_holding.loc[data_holding['WINDL1TYPE'].isin(['可转债', '可交换债', '可分离转债存债']), 'AssetType_VaR'] = '可转债'
            data_holding = data_holding.dropna(subset=['AssetType_VaR'])

            return data_holding
        
        data_holding1 = _dealAssetType(data_holding).copy()
        # TODO: 无评级债券暂取做AA
        data_holding1['RATE_LATESTMIR_CNBD'] = data_holding1['RATE_LATESTMIR_CNBD'].replace('无评级', 'AA')
        data_holding1['PTMYEAR'] = data_holding1['PTMYEAR'].fillna(0)
        # 删除无明确分类的数据
        data_holding1 = data_holding1[~data_holding1['AssetType_VaR'].isin(['无分类'])].copy()

        model=model_VaR()   # 创建VaR对象
        data_holding1['AssetVaR_basic'] = [model.calc_VaR(item['AssetType_VaR'], item['D_DATE'], item['PTMYEAR'], item['RATE_LATESTMIR_CNBD']) for idx, item in data_holding1.iterrows()]
        data_holding1['AssetVaR'] = data_holding1['AssetVaR_basic'] * data_holding1['MODIDURA_CNBD'].fillna(0) - data_holding1['YIELD_CNBD'].fillna(0)/12

        data_holding2 = self._dealHoldingVaR(data_holding1)

        return data_holding2

    def _dealHoldingVaR(self, data):
        '''债券到期日早于专户到期日的，无需设置VaR阈值'''
        data = pd.merge(data, self._fundName[['C_FUNDNAME', '产品到期日']], on=['C_FUNDNAME'], how='left')
        data.loc[data['到期日期'] <= data['产品到期日'], 'AssetVaR'] = 0
        
        return data

    def PortVaR(self, data):
        '''计算组合VaR'''
        res = (data['AssetVaR'] * data['F_ASSETRATIO'] / 100).sum()
        res = min(res / -100,0)

        return res

    def ReturnRate(self, netvalue_df=0):
        '''区间收益率'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        returnRate = netvalue_df['NAV'].iloc[-1] / netvalue_df['NAV'].iloc[0] - 1
        
        return returnRate

    def ReturnRate_yr(self, netvalue_df=0):
        '''区间收益率(年化)'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f.copy()
        else:
            netvalue_df = netvalue_df.reset_index(level=['D_DATE'])
        delta = (netvalue_df['D_DATE'].max() - netvalue_df['D_DATE'].min()).days
        returnRate = self.ReturnRate(netvalue_df)
        returnRate_yr = (1+returnRate)**(365/delta) - 1

        return returnRate_yr

    def Beta(self,netvalue_df=0,benchmark=0):
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        if type(benchmark)==int:
            benchmark = netvalue_df.reset_index(level=['D_DATE'])[['D_DATE', 'ret_b']].copy()
        ret_f, ret_b = self.timeAlign(netvalue_df[['ret', 'NAV']], benchmark)
        cov_l = np.cov(ret_f['ret'], ret_b['ret_b'])
        return cov_l[0, 1] / cov_l[1, 1]

    def Volatility(self,netvalue_df=0):
        '''波动率(日频)'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        return netvalue_df['ret'].std(ddof=0)

    def Volatility_up(self,netvalue_df=0):
        '''上行波动率(日频)'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        return netvalue_df.loc[netvalue_df['ret'] > 0, 'ret'].std(ddof=0)

    def Volatility_down(self,netvalue_df=0):
        '''下行波动率(月频)'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        return netvalue_df.loc[netvalue_df['ret'] < 0, 'ret'].std(ddof=0) * np.sqrt(20)

    def Volatility_excess(self,netvalue_df=0,benchmark=0):
        '''超额波动率(月频)：sigma(rp) - sigma(rb)'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        if type(benchmark)==int:
            benchmark = netvalue_df.reset_index(level=['D_DATE'])[['D_DATE', 'ret_b']].copy()
        ret_f, ret_b = self.timeAlign(netvalue_df[['ret', 'NAV']], benchmark)
        vol_ex = ret_f['ret'].std(ddof=0) - ret_b['ret_b'].std(ddof=0)
        return vol_ex * np.sqrt(20)

    def TrackingError(self, netvalue_df=0, benchmark=0):
        '''跟踪误差(月频)：sigma(rp-rb)'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        if type(benchmark)==int:
            benchmark = netvalue_df.reset_index(level=['D_DATE'])[['D_DATE', 'ret_b']].copy()
        ret_f, ret_b = self.timeAlign(netvalue_df[['ret', 'NAV']], benchmark)
        tra_err = (ret_f['ret'] - ret_b['ret_b']).std(ddof=0)
        return tra_err * np.sqrt(20)
        
    def _calcHWM(self, x):
        '''前期高点High Water Mark'''
        # 传入净值序列
        y = x.diff().fillna(0)
        if x[(y >= 0) & (y.shift(-1) < 0)].shape[0] == 0:            # 即净值一路上行
            return np.nan
        else:
            idx_hwm = x[(y >= 0) & (y.shift(-1) < 0)].idxmax()
            hwm = x.loc[idx_hwm]
            
            return hwm

    def DrawDown(self, netvalue_df=0):
        '''回撤DrawDown：若为0则说明回撤HWM已恢复or净值一路上行
            DrawDown = S(t) / HWM(:t) - 1'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        hwm = self._calcHWM(netvalue_df['NAV'])
        if hwm is np.nan:
            return 0
        else:
            return min(netvalue_df['NAV'].iloc[-1] / hwm - 1, 0)
    
    def DrawDown_dura(self, netvalue_df=0):
        '''回撤持续天数'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        hwm = self._calcHWM(netvalue_df['NAV'])
        if hwm is np.nan:
            return np.nan
        elif netvalue_df['NAV'].iloc[-1] >= hwm:
            return np.nan
        else:
            idx_hwm = np.where(netvalue_df['NAV'].values == hwm)[0][0]
            return netvalue_df['NAV'].shape[0] - 1 - idx_hwm

    def MaxDrawDown(self, netvalue_df=0):
        '''最大回撤'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        dd = [self.DrawDown(netvalue_df[['NAV']].iloc[:i+1]) for i in range(len(netvalue_df['NAV']))]
        max_dd = min(dd)

        return max_dd
    
    # 最大回撤久期，即从最高点回落到最大回撤处经历多长时间，以估值表时间间隔为单位，目前为自然日
    def MaxDrawDown_Dura(self, netvalue_df=0):
        '''最大回撤持续时间(交易日)，即从最高点回落到最大回撤处经历多长时间'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        dd = [self.DrawDown(netvalue_df[['NAV']].iloc[:i+1]) for i in range(len(netvalue_df['NAV']))]
        idx_maxdd = np.argmin(dd)                                     # 最大回撤处index
        hwm = self._calcHWM(netvalue_df['NAV'].iloc[:idx_maxdd+1])    # 最大回撤对应的hwm
        idx_hwm = np.where(netvalue_df['NAV'].values == hwm)[0]       # 最大回撤HWM对应的index

        if len(idx_hwm) == 0:
            return np.nan
            
        dura = idx_maxdd - idx_hwm[0]
        return dura
    
    def MaxDrawDown_rec(self, netvalue_df=0):
        '''最大回撤恢复时间(交易日)，即从最高点回落至最大回撤之后，能否恢复至最高点，能则返回第一次恢复时间，不能则返回空值'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        dd = [self.DrawDown(netvalue_df[['NAV']].iloc[:i+1]) for i in range(len(netvalue_df['NAV']))]
        idx_maxdd = np.argmin(dd)                                     # 最大回撤处index
        hwm = self._calcHWM(netvalue_df['NAV'].iloc[:idx_maxdd+1])    # 最大回撤对应的hwm
        idx_hwm = np.where(netvalue_df['NAV'].values == hwm)[0]       # 最大回撤HWM对应的index
        max_list = np.where(netvalue_df['NAV'].iloc[idx_maxdd:].values >= hwm)[0]

        if (len(max_list) == 0) or (len(idx_hwm) == 0):        # 即在最大回撤点之后未能恢复至最高点或未发生定义的回撤
            return np.nan

        return max_list[0]

    def Calmar_index(self, netvalue_df=0, risk_free_list=0):
        '''Calmar比率，区间超额收益率/最大回撤'''
        # risk_free为一个数而非series
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        if type(risk_free_list)==int:
            risk_free_list = pd.DataFrame([0 for index in range(netvalue_df.shape[0])])[0]

        ret_f, ret_free = self.timeAlign(netvalue_df[['ret', 'NAV']], risk_free_list)
        return1 = self.ReturnRate(ret_f)
        maxdraw = self.MaxDrawDown(ret_f)
        risk_free = (ret_free['risk_free'] + 1).product() - 1
        if maxdraw == 0:
            # print("产品无回撤", netvalue_df.iloc[0, :3])
            return np.nan
        else:
            return (return1 - risk_free) / abs(maxdraw)
    
    # TODO: 改算法
    def pain_index(self,netvalue_df=0):
        '''痛苦指数：单位时间的回撤幅度'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        dd = [self.DrawDown(netvalue_df[['NAV']].iloc[:i+1]) for i in range(len(netvalue_df['NAV']))]
        max_dd = [min(dd[:i+1]) for i in range(len(dd))]                  # 求出每个最大回撤幅度
        return -np.unique(max_dd).sum()/len(netvalue_df)

    def Treynor(self, netvalue_df=0, risk_free_list=0, benchmark=0):
        '''Treynor比率，区间超额收益率/beta'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        if type(benchmark)==int:
            benchmark = netvalue_df.reset_index(level=['D_DATE'])[['D_DATE', 'ret_b']].copy()
        if type(risk_free_list)==int:
            risk_free_list = pd.DataFrame([0 for index in range(netvalue_df.shape[0])])[0]
        ret_f, ret_free = self.timeAlign(netvalue_df[['ret', 'NAV']], risk_free_list)
        ret_f, ret_b = self.timeAlign(netvalue_df[['ret', 'NAV']], benchmark)
        
        try:
            risk_free = (ret_free['risk_free'].iloc[1:] + 1).product() - 1                 # 第一天即起始日不应有收益率，故剔除第一天的收益率数据
            beta2 = self.Beta(ret_f, ret_b)
            return (self.ReturnRate(ret_f) - risk_free) / beta2
        except:
            # print(netvalue_df.index[0], ret_f.shape, ret_free.shape, ret_b.shape)
            return np.nan

    def Sharpe(self,netvalue_df=0,risk_free_list=0):
        '''Sharpe比率，区间超额收益率/波动率(波动率未年化)'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        if type(risk_free_list)==int:
            risk_free_list = pd.DataFrame([0 for index in range(netvalue_df.shape[0])])[0]
        ret_f, ret_free = self.timeAlign(netvalue_df[['ret', 'NAV']], risk_free_list)
        risk_free = (ret_free['risk_free'].iloc[1:] + 1).product() - 1                 # 第一天即起始日不应有收益率，故剔除第一天的收益率数据
        return (self.ReturnRate(ret_f) - risk_free) / ret_f['ret'].std(ddof=0)

    def AdjustedSharpe(self,netvalue_df=0,risk_free_list=0):
        '''调整Sharpe比率'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        if type(risk_free_list)==int:
            risk_free_list = pd.DataFrame([0 for index in range(netvalue_df.shape[0])])[0]
        ret_f, ret_free = self.timeAlign(netvalue_df[['ret', 'NAV']], risk_free_list)
        SR = self.Sharpe(ret_f, ret_free)
        sk = ret_f['ret'].skew()
        ku = ret_f['ret'].kurt()
        return SR * (1 + sk * SR / 6 - ku * SR * SR / 24)

    def Jenson(self,netvalue_df=0, risk_free_list=0, benchmark=0):
        '''Jenson比率，CAPM超额收益'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        if type(benchmark)==int:
            benchmark = netvalue_df.reset_index(level=['D_DATE'])[['D_DATE', 'ret_b']].copy()
        if type(risk_free_list)==int:
            risk_free_list = pd.DataFrame([0 for index in range(netvalue_df.shape[0])])[0]
        ret_f, ret_free = self.timeAlign(netvalue_df[['ret', 'NAV']], risk_free_list)
        ret_f, ret_b = self.timeAlign(netvalue_df[['ret', 'NAV']], benchmark)

        try:
            beta1 = self.Beta(ret_f, ret_b)
            risk_free = (ret_free['risk_free'].iloc[1:] + 1).product() - 1
            return_market = (ret_b['ret_b'].iloc[1:] + 1).product() - 1        # 第一天即起始日不应有收益率，故剔除第一天的收益率数据
            return self.ReturnRate(ret_f) - (risk_free + beta1 * (return_market - risk_free))
        except:
            # print(netvalue_df.index[0], ret_f.shape, ret_free.shape, ret_b.shape)
            return np.nan

    def Sortino(self, netvalue_df=0, risk_free_list=0):
        '''Sortino比率，区间超额收益率/下行波动率'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        ####Sortino比率
        if type(risk_free_list)==int:
            risk_free_list = pd.DataFrame([0 for index in range(netvalue_df.shape[0])])[0]
        ret_f, ret_free = self.timeAlign(netvalue_df[['ret', 'NAV']], risk_free_list)
        risk_free = (ret_free['risk_free'].iloc[1:] + 1).product() - 1      # 第一天即起始日不应有收益率，故剔除第一天的收益率数据

        down_std = self.Volatility_down(ret_f)
        return (self.ReturnRate(ret_f) - risk_free) / down_std

    # TODO: not tested!
    def Upside_Potential(self, netvalue_df=0, return_required=0):
        '''Upside Potential比率(not evaluated and not used)'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        return_ret_day = pow(return_required, 1 / (len(netvalue_df)-1))
        netvalue_df["abv"] = netvalue_df['ret'] - return_ret_day
        netvalue_df["abv2"] = netvalue_df["abv"].copy()
        netvalue_df["label"] = pd.cut(netvalue_df["abv2"], [-10, -0.00000001, 10], labels=[-1, 1])
        up_std = self.Volatility_up(netvalue_df)
        result=netvalue_df[netvalue_df["label"] == 1].mean() / up_std
        return result["abv"]

    def Information_Ratio(self, netvalue_df=0, risk_free_list=0, benchmark=0):
        '''信息比率：Jenson/Tracking_error'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        if type(benchmark)==int:
            benchmark = netvalue_df.reset_index(level=['D_DATE'])[['D_DATE', 'ret_b']].copy()
        if type(risk_free_list)==int:
            risk_free_list = pd.DataFrame([0 for index in range(netvalue_df.shape[0])])[0]

        try:
            alf = self.Jenson(netvalue_df, risk_free_list, benchmark)
            vol_b = self.TrackingError(netvalue_df, benchmark)
            return alf / vol_b
        except:
            # print(netvalue_df.index[0])
            return np.nan

    def PainRatio(self, netvalue_df=0):
        '''收益痛苦比：区间收益率/痛苦指数'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        pain_idx = self.pain_index(netvalue_df)
        if pain_idx == 0:
            return np.nan
        else:
            return self.ReturnRate(netvalue_df) / pain_idx

    def hit_rate(self, netvalue_df=0):
        '''命中率：收益率大于0的天数占比'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        return netvalue_df[netvalue_df['ret'] >= 0].shape[0] / netvalue_df['ret'].count()

    def GPR(self,netvalue_df=0):
        '''Gain to Pain Ratio: 上行累计收益/下行累计收益'''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        ####Gain to Pain Ratio
        sum_up = netvalue_df.loc[netvalue_df['ret'] > 0, 'ret'].sum()
        sum_down = netvalue_df.loc[netvalue_df['ret'] < 0, 'ret'].sum()
        if sum_down == 0:
            return np.nan
        else:
            return -(sum_up / sum_down)

    def _duration_type(self, data):
        '''识别久期结构的形态'''
        max1 = 0
        max_index = 0
        order2 = 0
        sum2 = 0
        order3 = 0
        ave_diff3 = 0
        type2 = "0"
        data["dura_index"] = pd.cut(data["MODIDURA_CNBD"], [0, 1, 3, 5, 10], labels=[1, 3, 5, 10])
        hist = pd.pivot_table(data, index=["dura_index"], values=["F_ASSETRATIO"], aggfunc="sum").reset_index()
        hist2 = hist.sort_values("F_ASSETRATIO", ascending=False).reset_index()
        hist2["diff"] = -hist2["F_ASSETRATIO"].diff().fillna(0)
        hist2["percent"] = hist2["F_ASSETRATIO"] / sum(hist2["F_ASSETRATIO"])
        hist2["sumpro"] = hist2["percent"].cumsum()
        # 最大久期集中区
        max1 = hist2.loc[0, "percent"]
        max_index = hist2.loc[0, "index"]

        if hist2.loc[2, "sumpro"] >= 0.9999:
            delta = 0.70
        else:
            delta = 0.675
        # 前两大久期集中区
        if max1 <= delta:
            order2 = hist2.loc[1, "index"] - hist2.loc[0, "index"]
            sum2 = hist2.loc[1, "sumpro"]
            if sum2 < 0.95:
                order3 = (hist2.loc[2, "index"] - hist2.loc[1, "index"]) * (hist2.loc[1, "index"] - hist2.loc[0, "index"])
                ave_diff3 = (hist2.loc[0, "percent"] - hist2.loc[2, "percent"]) / 2
        else:
            type2 = "子弹型"
        if type2 != "子弹型":
            if order3 > 0:
                if max_index > 1:
                    type2 = "正梯形"
                else:
                    type2 = "倒梯形"
            elif order2 > 1 and sum2 >= 0.9:
                type2 = "哑铃型"
            elif order2 > 1 and sum2 < 0.9:
                type2 = "蝴蝶型"
            elif sum2 > 0.9:
                type2 = "子弹型"
            elif ave_diff3 < 0.05 and ave_diff3 > 0:
                type2 = "均匀型"
            else:
                type2 = "陀螺型"
                
        return type2

    def _dealCode(self, data):
        data0 = data.copy()
        data0['mkt'] = data0['SECUMARKET'].map({89: '.IB', 83: '.SH', 90: '.SZ'})
        data0['code'] = data0['SECUCODE'] + data0['mkt']

        return data0.drop(columns=['SECUCODE', 'SECUMARKET', 'mkt'])

    def _idOptionEmbedded(self, baseDay):
        '''
        按照给定基期处理持仓债券中推荐行权的含权债清单，回售优先于调整票面利率\n
        :param baseDay: datetime/Timestamp, 基期
        :return: (list, list), codes0为推荐票面利率调整的债券清单, codes1为推荐回售/赎回的债券清单
        '''
        data0 = self.bond_holdings.loc[self.bond_holdings['CLAUSE'].notna(), ['code', 'CLAUSE', 'REPURCHASEDATE', 'CALLDATE']].drop_duplicates()
        data1 = self.data_jy.loc[self.data_jy['YIELDCODE'] == 2, ['D_DATE', 'code']].drop_duplicates()
        data0['D_DATE'] = baseDay
        # data = pd.merge(data0, data1, on=['D_DATE', 'code'], how='left')
        data = data0.loc[~data0['code'].isin(data1['code'].unique()),:].copy()
        if data.shape[0] == 0:
            return [], []
        
        data['票面利率调整'] = [1 if '调整票面利率' in x else 0 for x in data['CLAUSE']]
        data['回售or赎回'] = [1 if ('回售' in x ) or ('赎回' in x) else 0 for x in data['CLAUSE']]
        codes0 = data.loc[data['票面利率调整'] == 1, 'code'].unique().tolist()
        codes1 = data.loc[(data['回售or赎回'] == 1) & (data['REPURCHASEDATE'].notna()|data['CALLDATE'].notna()), 'code'].unique().tolist()
        codes0 = list(set(codes0) - set(codes1))    # 回售优先于调整票面利率

        return codes0, codes1

    def _dealInterestChg(self, wind_code):
        '''
        处理发生过票面利率调整的债券现金流信息\n
        :param wind_code: list, 债券代码清单
        :return: DataFrame
        '''
        dq = JYDB_Query()
        jy = dq.sec_query('bond_interest_chg', wind_code)
        jy = self._dealCode(jy)
        
        # 处理发生过票面利率调整的债券现金流信息
        jy['CASHFLOW_orig'] = jy['CASHFLOW'].copy()
        jy['CASHFLOW'] = jy['INTERESTTHISYEAR'] + jy['PAYMENTPER']

        return jy

    def _dealOptionDate(self, date_put, baseDay):
        '''
        依据基期，取含权债的最近一个行权日\n
        :param date_put: string, 取自wind的含权债行权日期, 格式如"2022-01-04, 2023-01-04"
        :param baseDay: string, 基期
        :return: datetime, 最近一个行权日
        '''
        date_list = sorted(date_put.replace(' ','').split(','))
        baseDay = pd.to_datetime(baseDay)
        date_list = pd.to_datetime(date_list)
        if len(date_list[date_list >= baseDay]) == 0:
            return pd.to_datetime('2099-01-01')
        date_target = date_list[date_list >= baseDay][0]   # 选择距离baseDay最近的一个行权日

        return date_target

    def _dealPuttableCash(self, x):
        '''
        处理推荐行使回售/赎回权的债券现金流信息\n
        :param x: DataFrame, 取自聚源的债券现金流信息
        :return: DataFrame, 经含权调整后的现金流信息
        '''
        total_cash = x.loc[x['PAYMENTDATE'] >= x['date_target'], 'PAYMENTPER'].sum()
        date_target = x['date_target'].iloc[0]

        res = pd.DataFrame([date_target, total_cash], index=['PAYMENTDATE', 'CASHFLOW']).T
        res['CASHFLOWTYPE'] = 4
        res['CASHFLOWTYPEDESC'] = '回售行权'

        return res

    def _deal2Option(self, wind_code, tb, baseDay):
        '''
        一条龙处理所有含权债的行权日、现金流等信息\n
        :param wind_code: string/list, 需处理的债券代码清单
        :param tb: DataFrame, 债券现金流信息表
        :param baseDay: string, 基期
        :return: DataFrame, 处理后的含权债现金流信息
        '''
        if type(wind_code) != list:
            wind_code = [wind_code]
        date_option = self.bond_holdings.loc[self.bond_holdings['code'].isin(wind_code) & (self.bond_holdings['CALLDATE'].notna() | self.bond_holdings['REPURCHASEDATE'].notna()), ['code', 'REPURCHASEDATE', 'CALLDATE']].drop_duplicates()
        date_option['rep_target'] = [self._dealOptionDate(x, baseDay) if x != '*' else np.nan for x in date_option['REPURCHASEDATE'].fillna('*')]
        date_option['call_target'] = [self._dealOptionDate(x, baseDay) if x != '*' else np.nan for x in date_option['CALLDATE'].fillna('*')]
        date_option['date_target'] = date_option[['rep_target', 'call_target']].min(axis=1)
        
        tb_1 = tb.loc[tb['code'].isin(wind_code), :].copy()
        tb_1 = pd.merge(tb_1, date_option, on=['code'], how='left')
        tb_2 = tb_1.groupby('code').apply(self._dealPuttableCash).reset_index()        # 回售日后所有现金流归集到回售日当天
        tb_f = tb_1.loc[tb_1['PAYMENTDATE'] < tb_1['date_target'], :].append(tb_2, sort=False)
        tb_f = tb_f.sort_values(['code', 'PAYMENTDATE'])

        return tb_f

    def _map_key_year_t1(self, key_years, year):
        idx = key_years.index(year)
        year_t1 = key_years[idx+1]

        return year_t1
        
    def _calc_key_cash(self, tb):
        '''
        汇总各关键期限上的现金流\n
        :param tb: DataFrame, 债券现金流信息
        :return: DataFrame, 关键期限上的现金流信息
        '''
        key_cash_t0 = tb.groupby(['code', 'key_year'])['cash_t'].sum()
        key_cash_t1 = tb.groupby(['code', 'key_year_t1'])['cash_t1'].sum()
        key_cash = key_cash_t0.append(key_cash_t1).reset_index().rename(columns={0: 'cash'})
        key_cash = key_cash.groupby(['code', 'key_year'])['cash'].sum().reset_index()

        return key_cash

    def _pivot_key_cash(self, key_cash):
        '''
        转置债券的关键久期数据\n
        :param key_cash: DataFrame, 各债券的关键久期数据
        :return: DataFrame
        '''
        data = key_cash.drop_duplicates()
        data.loc[:, 'new_col'] = [str(x) + 'Y_Duration' for x in data['key_year']]
        res = data.pivot(values='key_duration', columns='new_col', index='code')
        res['D_DATE'] = key_cash['baseDay'].values[0]

        # cols = [str(x) + 'Y_Duration' for x in self.key_years[1:]] + ['D_DATE']
        return res

    def _key_duration(self, wind_code, baseDay):
        '''
        计算给定债券在给定基期的关键久期\n
        :param wind_code: list, 债券代码清单
        :param baseDay: string, 基期
        :return: DataFrame, 含各关键期限的现金流及久期的债券信息表
        '''
        dq = JYDB_Query()
        tb = dq.sec_query('bond_cash_flow', wind_code)
        tb = self._dealCode(tb)

        key_years = self.key_years
        delta = 10  # bp
        baseDay = pd.to_datetime(baseDay)

        # 处理含权债的现金流：回售、赎回和调整票面利率
        codes0, codes1 = self._idOptionEmbedded(baseDay)
        if len(codes0) > 0:
            tb_0 = self._dealInterestChg(codes0)
        else:
            tb_0 = pd.DataFrame()
        if len(codes1) > 0:
            tb_1 = self._deal2Option(codes1, tb, baseDay)
        else:
            tb_1 = pd.DataFrame()
        tb = tb.loc[~tb['code'].isin(codes0+codes1), :].copy()
        tb = tb.append(tb_0, sort=False)
        tb = tb.append(tb_1, sort=False)

        tb = tb[tb['PAYMENTDATE'] >= baseDay].copy()
        tb['baseDay'] = baseDay
        tb['days'] = tb['PAYMENTDATE'] - baseDay
        tb['year'] = [x.days/365 for x in tb['days']]
        tb['key_year'] = pd.cut(tb['year'], bins=key_years, labels=key_years[:-1]).astype(float)
        tb['key_year_t1'] = [self._map_key_year_t1(key_years, x) for x in tb['key_year']]
        tb['cash_t'] = (1 - (tb['year'] - tb['key_year'])/(tb['key_year_t1'] - tb['key_year'])) * tb['CASHFLOW']    # 当期按比例划转的现金流
        tb['cash_t1'] = tb['CASHFLOW'] - tb['cash_t']

        # 按关键久期归总现金流，可能涉及到一个关键期限有多条现金流
        key_cash = self._calc_key_cash(tb)
        ytm = dq.sec_query('bond_yield', wind_code, baseDay.strftime('%Y-%m-%d'))
        ytm = self._dealCode(ytm)
        key_cash = pd.merge(key_cash, ytm[['code', 'VPYIELD']], on=['code'], how='left').rename(columns={'VPYIELD': 'yield'})
        key_cash['baseDay'] = baseDay
        if key_cash['yield'].count() == 0:
            print(baseDay)
            return None
        
        key_cash['yield'] = key_cash['yield'].fillna(0)   # 若无YTM则暂取0
        key_cash = key_cash[(key_cash['key_year'] >= 0) & (key_cash['key_year'] < np.inf)].copy()  # 0Y会占权重
        
        key_cash["price0"]=[x["cash"]/pow((1+x["yield"]/100),x["key_year"]) for idx, x in key_cash.iterrows()]
        key_cash["price-"]=[x["cash"]/pow((1+(x["yield"]-delta/100)/100),x["key_year"]) for idx, x in key_cash.iterrows()]
        key_cash["price+"]=[x["cash"]/pow((1+(x["yield"]+delta/100)/100),x["key_year"]) for idx, x in key_cash.iterrows()]
        key_cash["key_duration0"]=(key_cash["price-"]-key_cash["price+"])/(2*delta/10000*key_cash["price0"])
        key_cash["weight"] = key_cash.groupby(['code'])['price0'].apply(lambda x: x/x.sum())
        key_cash["key_duration"]=key_cash["key_duration0"]*key_cash["weight"]

        return key_cash

    def dealKey_duration(self):
        '''
        计算持仓债券的关键久期、关键期限现金流等各项信息\n
        :return: (DataFrame, DataFrame). res_duraP为转置后各债券的关键久期，res_dura为转置前各债券横向现金流及久期表
        '''
        if 'bond_holdings' not in dir(self) or self.bond_holdings.empty:
            self._loadHoldings(self.save_path)

        data = self.bond_holdings.copy()
        if data.empty:
            res_duraP = pd.DataFrame(columns=['0.0Y_Duration', '0.5Y_Duration', '1.0Y_Duration', '3.0Y_Duration', '5.0Y_Duration', '7.0Y_Duration', '10.0Y_Duration', '30.0Y_Duration', '50.0Y_Duration', 'D_DATE'])
            res_duraP.index.name = 'code'
            res_dura = pd.DataFrame(columns=['code', 'key_year', 'cash', 'yield', 'D_DATE', 'price0', 'price-', 'price+', 'key_duration0', 'weight', 'key_duration'])
            return res_duraP, res_dura

        date_list = data['D_DATE'].drop_duplicates().tolist()
        res_dura_list = []
        res_duraP_list = []
        for date_temp in date_list:
            code_list = data.loc[data['D_DATE'] == date_temp, 'code'].drop_duplicates().tolist()
            if len(code_list) > 0:
                key_dura_temp = self._key_duration(code_list, date_temp)
                if key_dura_temp is None:
                    continue
                dura_pivot_temp = self._pivot_key_cash(key_dura_temp)
                res_dura_list.append(key_dura_temp)
                res_duraP_list.append(dura_pivot_temp)
        
        res_dura = pd.DataFrame(columns=key_dura_temp.columns)
        for x in res_dura_list:
            res_dura = res_dura.append(x, sort=False)
        res_dura = res_dura.rename(columns={'baseDay': 'D_DATE'})
        res_duraP = pd.DataFrame(columns=dura_pivot_temp.columns)
        for x in res_duraP_list:
            res_duraP = res_duraP.append(x, sort=False)
        res_duraP.index.name = 'code'
        return res_duraP, res_dura
    
    def calcKey_Duration(self, data, label=''):
        '''
        计算给定持仓组合的关键久期\n
        :param data: DataFrame, 持仓债券信息
        :param label: string, 计算后的关键久期列名前置标签, 默认无标签则为组合的关键久期
        :return: DataFrame
        '''
        dur_cols = [label + str(x) + 'Y_Duration' for x in sorted(self.key_years) if (x > 0) and (x < np.inf)]
        res_cols = ['C_FUNDNAME', 'D_DATE'] + dur_cols
        if data.empty:
            return pd.DataFrame(columns=res_cols)

        if 'dura_asset' not in dir(self):
            self.duraP_holdings_ex, self.dura_asset = self.dealKey_duration()

        data['weight'] = data['F_ASSETRATIO'] / 100
        data = pd.merge(data, self.dura_asset.drop(columns=['weight']), on=['D_DATE', 'code'], how='left')
        
        res = data.groupby(['C_FUNDNAME', 'D_DATE', 'key_year']).apply(lambda x: (x['weight'] * x['key_duration']).sum()).rename('key_duration').reset_index()
        res['new_col'] = [label + str(x).replace('.0', '') + 'Y_Duration' for x in res['key_year']]
        res_pivot = pd.pivot_table(res, index=['C_FUNDNAME', 'D_DATE'], columns='new_col', values='key_duration').reset_index()
        res_pivot = res_pivot.reindex(columns=res_cols)

        return res_pivot

    def calcKey_duration_P(self, data_holding):
        '''计算各组合的关键久期'''
        self.duraP_holdings_ex, self.dura_asset = self.dealKey_duration()

        data = data_holding.loc[~data_holding['WINDL1TYPE'].isin(['可转债', '可交换债']), ['C_FUNDNAME', 'D_DATE', 'code', 'F_ASSETRATIO']].copy()
        res_pivot = self.calcKey_Duration(data)

        return res_pivot

    def calcKey_duration_ir(self, data_holding):
        '''计算各组合持仓利率债的关键久期'''
        data = data_holding.loc[data_holding['利率or信用'] == '利率债', ['C_FUNDNAME', 'D_DATE', 'code', 'F_ASSETRATIO']].copy()
        res_pivot = self.calcKey_Duration(data, label='利率债_')

        return res_pivot
    
    def calcKey_duration_cr(self, data_holding):
        '''计算各组合持仓信用债的关键久期'''
        data = data_holding.loc[(data_holding['利率or信用'] == '信用债') & ~data_holding['WINDL1TYPE'].isin(['可转债', '可交换债']), ['C_FUNDNAME', 'D_DATE', 'code', 'F_ASSETRATIO']].copy()
        res_pivot = self.calcKey_Duration(data, label='信用债_')

        return res_pivot

    def calcKey_duration_all(self, data_holding):
        '''计算各组合各类债券的关键久期，并计算各关键久期占比'''
        res_pivot_p = self.calcKey_duration_P(data_holding)
        res_pivot_ir = self.calcKey_duration_ir(data_holding)
        res_pivot_cr = self.calcKey_duration_cr(data_holding)

        res = pd.merge(res_pivot_p, res_pivot_ir, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res = pd.merge(res, res_pivot_cr, on=['C_FUNDNAME', 'D_DATE'], how='left')

        cols_all = res.set_index(['C_FUNDNAME', 'D_DATE']).columns.tolist()
        cols_ratio_all = [x + '_Ratio' for x in cols_all]
        cols = [str(x) + 'Y_Duration' for x in sorted(self.key_years) if (x > 0) and (x < np.inf)]

        if res.empty:
            return res.reindex(columns=['C_FUNDNAME', 'D_DATE'] + cols_all + ['KeyDura_sum'] + cols_ratio_all)

        res['KeyDura_sum'] = res[cols].sum(axis=1)
        for col, col_ratio in zip(cols_all, cols_ratio_all):
            res[col_ratio] = res[col] / res['KeyDura_sum']
        return res

    def calc_key_duration_special(self, prods):
        '''计算特殊产品的关键久期数据'''
        self.duraP_holdings_ex, self.dura_asset = self.dealKey_duration()

        data_holding = self.bond_holdings.copy()
        data = data_holding.loc[~data_holding['WINDL1TYPE'].isin(['可转债', '可交换债', '资产支持证券']),
                                ['C_FUNDNAME', 'D_DATE', 'code', 'F_ASSETRATIO']].copy()
        if len(prods) > 0:
            data = data[data['C_FUNDNAME'].isin(prods)].copy()

        res_key_dura = self.calcKey_Duration(data).fillna(0)
        return res_key_dura

    def dealKey_year(self, data_keyDura):
        '''
        计算各组合的关键期限, 即对应的关键久期暴露除以对应的关键期限。如在3年关键期限上的关键久期为1.2, 则关键期限=1.2/3=40%\n
        :param data_keyDura: DataFrame, 各债券的关键久期数据
        :return: DataFrame
        '''
        # keyDura的index为债券代码            
        if 'D_DATE' in data_keyDura.columns:
            data_keyDura = data_keyDura.set_index(['D_DATE'], append=True)
        key_year = data_keyDura.columns.tolist()
        key_year = [float(x.replace('Y_Duration', '')) for x in key_year]

        sort_col = [str(x).replace('.0', '') + 'Y_Ratio' for x in sorted(key_year)]
        if data_keyDura.empty:
            return pd.DataFrame(columns=['code', 'D_DATE'] + sort_col)

        key_dura = data_keyDura.values
        res_keyYear = calcKeyYearRatio_n(key_year, key_dura)
        new_cols = [col.replace('.0', '').replace('Duration', 'Ratio') for col in data_keyDura.columns]
        res_keyYear = pd.DataFrame(res_keyYear, columns=new_cols, index=data_keyDura.index)
        res_keyYear = res_keyYear.reindex(columns=sort_col).reset_index()
        return res_keyYear

    def _calcKeyYear_weighted(self, x):
        '''
        计算加权关键期限\n
        :param x: DataFrame, 含权重weight和关键期限key_year数据的DataFrame
        :return: DataFrame
        '''
        weight = x['weight'].values
        ratios = x.drop(columns=['weight', 'code']).fillna(0).values
        prod = np.dot(ratios.T, weight)
        res = pd.DataFrame(prod, index=x.drop(columns=['weight', 'code']).columns).T

        return res

    def calcKeyYear_Ratio(self, data, label=''):
        '''
        计算给定持仓组合的关键期限\n
        :param data: DataFrame, 持仓债券信息
        :param label: string, 计算后的关键期限列名前置标签, 默认无标签则为组合的关键期限
        :return: DataFrame
        '''
        if 'dura_asset' not in dir(self):
            self.duraP_holdings_ex, self.dura_asset = self.dealKey_duration()
        data_keyRatio = self.dealKey_year(self.duraP_holdings_ex)
        self.data_keyRatio = data_keyRatio.copy()

        key_cols = [label + str(x).replace('.0', '') + 'Y_Ratio' for x in self.key_years[2:-1]]
        res_cols = ['C_FUNDNAME', 'D_DATE'] + key_cols

        if data.empty:
            return pd.DataFrame(columns=res_cols)

        data['weight'] = data['F_ASSETRATIO'] / 100
        data = pd.merge(data[['C_FUNDNAME', 'D_DATE', 'code', 'weight']], data_keyRatio, on=['D_DATE', 'code'], how='left')
        
        res = data.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME', 'D_DATE']).apply(self._calcKeyYear_weighted)
        res.columns = [label + col for col in res.columns]
        if label + '0Y_Ratio' in res.columns:
            next_col = res.columns[[i for i in range(len(res.columns)) if label + '0Y_Ratio' == res.columns[i]][0] + 1]
            res[next_col] = res[label + '0Y_Ratio'] + res[next_col]
        res = res.reset_index().reindex(columns=res_cols)
        
        return res

    def calcKey_ratio_P(self, data_holding):
        '''计算各组合的关键期限'''
        data = data_holding.loc[~data_holding['WINDL1TYPE'].isin(['可转债', '可交换债']), ['C_FUNDNAME', 'D_DATE', 'code', 'F_ASSETRATIO']].copy()
        res_ratio = self.calcKeyYear_Ratio(data)

        return res_ratio

    def calcKey_ratio_ir(self, data_holding):        
        '''计算各组合持仓利率债的关键期限'''
        data = data_holding.loc[data_holding['利率or信用'] == '利率债', ['C_FUNDNAME', 'D_DATE', 'code', 'F_ASSETRATIO']].copy()
        res_ratio = self.calcKeyYear_Ratio(data, label='利率债_')

        return res_ratio
    
    def calcKey_ratio_cr(self, data_holding):
        '''计算各组合持仓信用债的关键期限'''
        data = data_holding.loc[(data_holding['利率or信用'] == '信用债') & ~data_holding['WINDL1TYPE'].isin(['可转债', '可交换债']), ['C_FUNDNAME', 'D_DATE', 'code', 'F_ASSETRATIO']].copy()
        res_ratio = self.calcKeyYear_Ratio(data, label='信用债_')

        return res_ratio

    def calcKey_ratio_all(self, data_holding):
        '''计算各组合各类债券的关键期限'''
        res_ratio_p = self.calcKey_ratio_P(data_holding)
        res_ratio_ir = self.calcKey_ratio_ir(data_holding)
        res_ratio_cr = self.calcKey_ratio_cr(data_holding)

        res = pd.merge(res_ratio_p, res_ratio_ir, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res = pd.merge(res, res_ratio_cr, on=['C_FUNDNAME', 'D_DATE'], how='left')

        return res

    def getPTMDist(self, data, label='组合', col_name='PTM分组'):
        '''
        计算各组合给定分类下的剩余期限分布\n
        :param data: DataFrame, 组合持仓债券信息表(含剩余期限分组信息)
        :param label: string, 计算后的剩余期限分布列名前置标签, 默认无标签则为组合的剩余期限分布
        :param col_name: string, 剩余期限分组依据的列名
        :return: DataFrame
        '''
        col_order = ['1Y以下', '1-3Y', '3-5Y', '5-7Y', '7-10Y', '10-30Y', '30-50Y', '50Y以上']
        new_col = [label + '_' + x for x in ['1Y以下', '1-3Y', '3-5Y', '5-7Y', '7-10Y', '10-30Y', '30-50Y', '50Y以上']]
        if data.empty:
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE'] + new_col + [label + '_10Y以上'])

        res = pd.pivot_table(data, values='F_ASSETRATIO', columns=col_name, index=['C_FUNDNAME', 'D_DATE'], aggfunc='sum').reindex(columns=col_order)
        res.columns = new_col
        res = res.reset_index()
        # todo: 系统升级后可以删除
        res[label + '_10Y以上'] = res[[label + '_10-30Y', label + '_30-50Y', label + '_50Y以上']].sum(axis=1)

        return res

    def _convertTotalAssetRatio(self, x, index_col=['C_FUNDNAME', 'D_DATE']):
        '''
        将占净值比转换为占总资产比\n
        :param x: DataFrame, 需进行转换的数据表
        :param index_col: list, 主键清单, 默认为"组合名称+持仓日期"
        :return: DataFrame
        '''
        if x.empty:
            return x.copy()

        lev = self.data_fund[['C_FUNDNAME', 'D_DATE', 'TotalAsset', 'NetAsset']].copy()
        lev['lev'] = lev['TotalAsset'] / lev['NetAsset']

        new_x = pd.merge(x, lev, on=['C_FUNDNAME', 'D_DATE'], how='left').drop(columns=['TotalAsset', 'NetAsset'])
        all_cols = x.set_index(index_col).columns.tolist()
        for col in all_cols:
            new_x[col] = new_x[col] / new_x['lev']
        
        return new_x.drop(columns=['lev'])

    def _loadBenchMarkTarget(self):
        '''读取专户的业绩比较基准数据'''
        self._bchTarget = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\存量专户产品业绩比较基准.xlsx', engine='openpyxl').rename(columns={'估值系统简称': 'C_FUNDNAME'})
        self._bchTarget['存续期限'] = [x.days / 365 for x in self._bchTarget['产品到期日'] - self._bchTarget['产品设立日期']]
        self._bchTarget = self._bchTarget[['C_FUNDNAME', '绝对收益', '存续期限', '考核期比较基准', '考核期开始日', '考核期结束日']].dropna(subset=['绝对收益'])

    def calcReturnGap(self, data_nav, data_holding):
        '''
        计算专户产品经静态收益率换算后的收益率缺口\n
        :param data_nav: DataFrame, 单位净值数据表
        :param data_holding: DataFrame, 组合加权静态收益率表
        :return: DataFrame, 列ReturnGap_abs表示绝对收益缺口, ReturnGap表示年化收益率缺口
        '''
        if 'avgYield_static_lev' not in dir(self):
            self.avgYield_static_lev = self.avgYield_static(data_holding)
        avgYield_static_lev = self.avgYield_static_lev[['C_FUNDNAME', 'D_DATE', 'avgYield_static']].copy()

        if avgYield_static_lev.empty:
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'ReturnRate_cum', 'days_cum', 'Benchmark',
                                         'TotalMaturity', 'ReturnGap_abs', 'ReturnGap', 'avgYield_static'])

        self._loadBenchMarkTarget()
        grouped = data_nav.groupby(['C_FUNDNAME'])
        ret_cum = grouped.apply(self.ReturnRate).rename('ReturnRate_cum').reset_index()
        day_cum = grouped.apply(lambda x: (x['D_DATE'].max() - x['D_DATE'].min()).days).rename('days_cum').reset_index()

        data_m = pd.merge(ret_cum, day_cum, on=['C_FUNDNAME'], how='left')
        data_m = pd.merge(data_m, self._bchTarget[['C_FUNDNAME', '绝对收益', '存续期限']], on=['C_FUNDNAME'], how='left')
        data_m = pd.merge(data_m, avgYield_static_lev, on=['C_FUNDNAME'], how='left')
        data_m['ReturnGap_abs'] = data_m['绝对收益'] * data_m['存续期限'] - (data_m['ReturnRate_cum'] + data_m['avgYield_static']/100 * (data_m['存续期限'] - data_m['days_cum']/365))
        data_m['ReturnGap'] = data_m['ReturnGap_abs'] / (data_m['存续期限'] - data_m['days_cum']/365) * (data_m['ReturnGap_abs'] > 0).astype(int)
        data_m['D_DATE'] = data_m['D_DATE'].fillna(pd.to_datetime(self.basedate))

        return data_m

    def calc_portytm_component(self):
        '''
        拆分组合中债券资产部分静态收益率的组成
        :return:
        '''
        if self.bond_holdings.empty:
            self._loadHoldings(self.save_path)
        data_holding = self.bond_holdings.loc[~self.bond_holdings['WINDL1TYPE'].isin(
            ['资产支持证券', '可转债', '可交换债', '可分离转债存债'])].copy()

        if data_holding.empty:
            logger.info('-- rc_mr_yield_comp: 无纯债持仓，无需更新数表')
            return None

        data_holding.loc[data_holding['利率or信用'] == '利率债', '曲线代码'] = 195
        data_holding.loc[data_holding['利率or信用'] == '利率债', 'benchmark_rating'] = data_holding.loc[data_holding['利率or信用'] == '利率债', 'benchmark_gk']
        ptm_port = self.avg_ptm_port(data_holding)
        ptm_port['ptm_round'] = [round(x, 1) for x in ptm_port['avgptm_bond']]
        ptm_port['c_date'] = [x.strftime('%Y-%m-%d') for x in ptm_port['D_DATE']]

        data = pd.merge(ptm_port, self._benchmark_gk, left_on=['D_DATE', 'ptm_round'],
                        right_on=['ENDDATE_gk', 'YEARSTOMATURITY_gk'], how='left').drop(columns=['ENDDATE_gk'])
        data.columns = [x.lower() for x in data.columns]
        data = data.rename(columns={'benchmark_gk': 'yield_comp_gk'}).drop(columns=['yearstomaturity_gk'])

        # 确保每个组合每只券持仓的唯一性
        data_holding.columns = [x.lower() for x in data_holding.columns]
        basic_col = ['d_date', 'c_fundname', 'code', 'c_subname_bsh']
        sum_col = ['f_asset', 'f_assetratio']
        avg_col = ['ptmyear', 'yield_cnbd', 'benchmark_gk', 'benchmark_rating']
        agg_f = dict(zip(sum_col+avg_col, ['sum'] * 2 + ['mean'] * 4))
        df = data_holding.groupby(basic_col).agg(agg_f).reset_index()

        # 收益率拆分：期限选择(按债券资产ptm计算的国开收益率); 期限结构(yield_comp_struc); 信用等级(yield_comp_rating); 个券溢价(yield_comp_spread)
        data_holding = pd.merge(df, data, on=['c_fundname', 'd_date'], how='left')
        data_holding['yield_comp_struc'] = data_holding['benchmark_gk'] - data_holding['yield_comp_gk']
        data_holding['yield_comp_rating'] = data_holding['benchmark_rating'] - data_holding['benchmark_gk']
        data_holding['yield_comp_spread'] = data_holding['yield_cnbd'] - data_holding['benchmark_rating']

        # 添加portfolio_code
        data_holding['portfolio_code'] = data_holding['c_fundname'].map(self.fundname_to_code)

        # 存储个券
        col_nd = ['c_date', 'portfolio_code', 'c_fundname', 'sec_code', 'sec_name', 'f_asset', 'f_assetratio',
                  'ptmyear', 'yield_cnbd', 'benchmark_gk', 'benchmark_rating',
                  'yield_comp_gk', 'yield_comp_struc', 'yield_comp_rating', 'yield_comp_spread']
        holding_yield = data_holding.rename(columns={
            'code': 'sec_code', 'c_subname_bsh': 'sec_name'}).reindex(columns=col_nd).fillna(0)
        holding_yield['sec_name'] = [x.replace('(总价)', '') for x in holding_yield['sec_name']]
        self.holding_yield = holding_yield[(holding_yield['c_fundname'] != 0) &
                                           (holding_yield['portfolio_code'] != 0)].copy()
        yield_comp_port = self.holding_yield.groupby(['c_date', 'portfolio_code', 'c_fundname']).apply(lambda x: (
            np.multiply(x[['yield_cnbd', 'yield_comp_gk', 'yield_comp_struc', 'yield_comp_rating', 'yield_comp_spread']],
                        np.mat((x['f_asset']/x['f_asset'].sum()).values.reshape(x.shape[0], 1)))).sum()).reset_index()
        yield_comp_port = pd.merge(yield_comp_port, ptm_port, left_on=['c_date', 'c_fundname'],
                                   right_on=['c_date', 'C_FUNDNAME'], how='left').drop(columns=['D_DATE', 'C_FUNDNAME'])
        self.yield_comp_port = yield_comp_port.rename(columns={'yield_cnbd': 'yield_bond'})

        # 纳入骑乘收益率的测算结果
        self.calc_yield_rolldown()

        self.yield_comp_all = pd.merge(self.yield_comp_port, self.rolldown_port,
                                       on=['c_date', 'portfolio_code', 'c_fundname'], how='left').drop_duplicates()
        self.yield_sec_all = pd.merge(self.holding_yield, self.rolldown_yield.drop(
            columns=['sec_name', 'f_asset', 'f_assetratio', 'ptmyear', 'yield_cnbd', 'benchmark_rating']),
                                      on=['c_date', 'portfolio_code', 'c_fundname', 'sec_code'], how='left').drop_duplicates()

        self.insert2db_single('dpe_mr_yield_comp', self.yield_sec_all, t=self.basedate, t_colname='c_date',
                              ptf_code=self.ptf_codes, code_colname='portfolio_code')
        self.insert2db_single('rc_mr_yield_comp', self.yield_comp_all, t=self.basedate, t_colname='c_date',
                              ptf_code=self.ptf_codes, code_colname='portfolio_code')

    def calc_yield_rolldown(self):
        if 'bond_holdings' not in dir(self):
            self._loadHoldings(self.save_path)
        data_holding = self.bond_holdings.loc[~self.bond_holdings['WINDL1TYPE'].isin(
            ['资产支持证券', '可转债', '可交换债', '可分离转债存债'])].copy()
        data_holding.loc[data_holding['利率or信用'] == '利率债', '曲线代码'] = 195
        data_holding.loc[data_holding['利率or信用'] == '利率债', 'benchmark_rating'] = \
            data_holding.loc[data_holding['利率or信用'] == '利率债', 'benchmark_gk']

        data_holding.columns = [x.lower() for x in data_holding.columns]
        data_holding['ptm_rolldown_3m'] = [round(max(x, 0),1) for x in data_holding['ptmyear']-0.25]
        data_holding['ptm_rolldown_6m'] = [round(max(x, 0),1) for x in data_holding['ptmyear']-0.5]
        data_holding['ptm_rolldown_1y'] = [round(max(x, 0),1) for x in data_holding['ptmyear']-1]

        benchmark_gk = self._benchmark_gk.rename(columns={'ENDDATE_gk': 'ENDDATE',
                                                          'YEARSTOMATURITY_gk': 'YEARSTOMATURITY',
                                                          'benchmark_gk': 'YIELD'})
        benchmark_gk['CURVECODE'] = 195
        benchmark_curve = pd.concat([self._benchmark_rating, benchmark_gk], axis=0, sort=True)

        data_holding = pd.merge(data_holding, benchmark_curve.rename(columns={'YIELD': 'yield_rating_3m'}),
                                left_on=['d_date', '曲线代码', 'ptm_rolldown_3m'], right_on=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY'],
                                how='left').drop(columns=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY'])
        data_holding = pd.merge(data_holding, benchmark_curve.rename(columns={'YIELD': 'yield_rating_6m'}),
                                left_on=['d_date', '曲线代码', 'ptm_rolldown_6m'], right_on=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY'],
                                how='left').drop(columns=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY'])
        data_holding = pd.merge(data_holding, benchmark_curve.rename(columns={'YIELD': 'yield_rating_1y'}),
                                left_on=['d_date', '曲线代码', 'ptm_rolldown_1y'], right_on=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY'],
                                how='left').drop(columns=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY'])
        data_holding['yield_rolldown_3m'] = data_holding['modidura_cnbd'] * (data_holding['benchmark_rating'] - data_holding['yield_rating_3m'])
        data_holding['yield_rolldown_6m'] = data_holding['modidura_cnbd'] * (data_holding['benchmark_rating'] - data_holding['yield_rating_6m'])
        data_holding['yield_rolldown_1y'] = data_holding['modidura_cnbd'] * (data_holding['benchmark_rating'] - data_holding['yield_rating_1y'])
        data_holding['c_date'] = [x.strftime('%Y-%m-%d') for x in data_holding['d_date']]

        # 添加portfolio_code
        q = sqls_config['portfolio_type']['Sql']
        fund_info = self.db_risk.read_sql(q).set_index('c_fundname')['c_fundcode'].to_dict()
        data_holding['portfolio_code'] = data_holding['c_fundname'].map(fund_info)

        # 存储个券
        col_nd = ['c_date', 'portfolio_code', 'c_fundname', 'sec_code', 'sec_name', 'f_asset', 'f_assetratio',
                  'ptmyear', 'yield_cnbd', 'benchmark_rating',
                  'ptm_rolldown_3m', 'ptm_rolldown_6m', 'ptm_rolldown_1y',
                  'yield_rating_3m', 'yield_rating_6m', 'yield_rating_1y',
                  'yield_rolldown_3m', 'yield_rolldown_6m', 'yield_rolldown_1y']
        rolldown_yield = data_holding.rename(columns={
            'code': 'sec_code', 'c_subname_bsh': 'sec_name'}).reindex(columns=col_nd).fillna(0)
        rolldown_yield['sec_name'] = [x.replace('(总价)', '') for x in rolldown_yield['sec_name']]
        self.rolldown_yield = rolldown_yield[rolldown_yield['c_fundname'] != 0 &
                                             (rolldown_yield['portfolio_code'] != 0)].copy()

        rolldown_port = rolldown_yield.groupby(['c_date', 'portfolio_code', 'c_fundname']).apply(
            lambda x: (np.multiply(x[['yield_rolldown_3m', 'yield_rolldown_6m', 'yield_rolldown_1y']],
                                  np.mat((x['f_asset']/x['f_asset'].sum()).values.reshape(x.shape[0], 1)))).sum())
        self.rolldown_port = rolldown_port.reset_index()

    def get_pastyear_date(self, t, n=1):
        '''计算过去n年的日期， 返回字符串'''
        t = t if isinstance(t, str) else t.strftime('%Y-%m-%d')
        year = str(int(t[:4]) - n)
        res = year + t[4:] if not (t[4:] == "-02-29" and n % 4 == 0) else year + "-02-28"
        return res

    def calcNavRelated(self, endDate='', period='Y'):
        '''计算单位净值相关的风险收益指标'''
        if endDate == '':
            endDate = pd.to_datetime(self.endDate)

        startDate = pd.to_datetime('2022-12-31')
        startDate = pd.to_datetime(startDate) if isinstance(startDate, str) else startDate
        self.baseDate = endDate.strftime('%Y-%m-%d')
        self.enddate_lastyear = self.get_pastyear_date(self.baseDate, 1)

        data = self._dealNavData(self.enddate_lastyear)
        self._risk_free_list = self._getBenchmarkReturn(benchmarkCode='SHIBOR3M.IR', startDate=startDate, endDate=endDate).drop(columns=['ret_idx'])
        self._risk_free_list['risk_free'] = self._risk_free_list['CLOSE'] / (100*365)

        func_nav = [self.ReturnRate, self.Volatility, self.Volatility_up, self.Volatility_down]
        name_nav = [x.__name__ for x in func_nav]
        func_nav2 = [self.DrawDown, self.DrawDown_dura, self.MaxDrawDown, self.MaxDrawDown_Dura, self.MaxDrawDown_rec]
        name_nav2 = [x.__name__ for x in func_nav2]
        func_nav3 = [self.pain_index, self.hit_rate, self.GPR, self.PainRatio]
        name_nav3 = [x.__name__ for x in func_nav3]
        func_bch = [self.Beta, self.Volatility_excess, self.TrackingError]
        name_bch = [x.__name__ for x in func_bch]
        func_free = [self.Calmar_index, self.Sharpe, self.AdjustedSharpe, self.Sortino]
        name_free = [x.__name__ for x in func_free]
        func_bch_free = [self.Treynor, self.Jenson, self.Information_Ratio]
        name_bch_free = [x.__name__ for x in func_bch_free]

        res_list = []
        try:
            for func, name in zip(func_nav, name_nav):
                res_temp = data.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME']).apply(func).reset_index().rename(columns={0: name})
                res_list.append(res_temp)
            # todo: 起始日期：回撤相关指标全部改为今年以来的数据（净值自前一年1231开始计算），其他指标暂保持不变
            # q = sqls_config['latest_tradeday']['Sql']%(self.baseDate[:4] + '-01-01')
            q = sqls_config['latest_tradeday']['Sql'] % ('2024-01-01')
            latest_tradeday = self.db_risk.read_sql(q).sort_values(by=['c_date'], ascending=False)['c_date'].iloc[0]
            data_now = self._dealNavData(latest_tradeday)
            for func, name in zip(func_nav2, name_nav2):
                res_temp = data_now.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME']).apply(func).reset_index().rename(columns={0: name})
                res_list.append(res_temp)
            for func, name in zip(func_nav3, name_nav3):
                res_temp = data_now.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME']).apply(func).reset_index().rename(columns={0: name})
                res_list.append(res_temp)

            # 比较基准默认参数为nav_f中包含的ret_b列
            for func, name in zip(func_bch, name_bch):
                res_temp = data.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME']).apply(func).reset_index().rename(columns={0: name})
                res_list.append(res_temp)
            # 下述函数需risk_free_list，为series，会在函数里处理成区间收益risk_free
            for func, name in zip(func_free, name_free):
                res_temp = data.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME']).apply(func, self._risk_free_list).reset_index().rename(columns={0: name})
                res_list.append(res_temp)
            # 下述函数需risk_free，区间收益常量；以及benchmark（默认参数为nav_f中包含的ret_b列）
            for func, name in zip(func_bch_free, name_bch_free):
                res_temp = data.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME']).apply(func, self._risk_free_list).reset_index().rename(columns={0: name})
                res_list.append(res_temp)
        except:
            print(name)

        # 指数基金跟踪偏离度及年化跟踪跟踪误差
        res_trk = self._loadIndexFundsFile(pd.to_datetime(endDate).strftime('%Y-%m-%d'))
        res_list.append(res_trk)

        res = self.nav_f[['C_FUNDNAME']].drop_duplicates().sort_values(by=['C_FUNDNAME'])
        res['D_DATE'] = pd.to_datetime(endDate)
        for x in res_list:
            res = pd.merge(res, x, on='C_FUNDNAME', how='left')

        res['PORTFOLIO_CODE'] = res['C_FUNDNAME'].map(self.fundname_to_code)
        self.res_nav = res.copy()
        self.res_list_nav = res_list.copy()            
        logger.info('——>  Nav Indices Done!')

    def calcHoldingRelated(self):
        '''计算持仓相关的指标'''

        del_types = ['资产支持证券', '可转债', '可交换债']
        data_holding = self.bond_holdings.loc[~self.bond_holdings['WINDL1TYPE'].isin(del_types)].copy()
        func_holding = [self.avgCoupon_bond, self.avgDuration_bond, self.avgCoupon_port, self.avgDuration_port]
        name_holding = [x.__name__ for x in func_holding]

        res_list0 = []
        self.res_lev = self.Leverage()
        self.res_lev_cost = self.LeverageCost()
        self.avgYield_static_lev = self.avgYield_static(data_holding)
        res_list0.append(self.res_lev)
        res_list0.append(self.res_lev_cost)
        res_list0.append(self.avgYield_static_lev)

        if data_holding.empty:
            for name in name_holding:
                res_list0.append(pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', name]))

            self.port_duraType = pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', '久期类型']) # 久期类型
            res_list0.append(self.port_duraType)
        else:
            for func, name in zip(func_holding, name_holding):
                res_temp = data_holding.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME', 'D_DATE']).apply(func).reset_index().rename(columns={0: name})
                res_list0.append(res_temp)

            self.port_duraType = data_holding.groupby(['C_FUNDNAME', 'D_DATE']).apply(self._duration_type).reset_index().rename(columns={0: '久期类型'})
            res_list0.append(self.port_duraType)

        # 关键久期
        self.port_KeyDura = self.calcKey_duration_all(data_holding)
        res_list0.append(self.port_KeyDura)
        
        # 组合剩余期限分布
        data_holding['PTM分组'] = pd.cut(data_holding['PTMYEAR'], bins=[-np.inf, 1, 3, 5, 7, 10, 30, 50, np.inf],
                                       labels=['1Y以下', '1-3Y', '3-5Y', '5-7Y', '7-10Y', '10-30Y', '30-50Y', '50Y以上'])
        data_ir = data_holding[data_holding['利率or信用'] == '利率债'].copy()
        data_cr = data_holding[data_holding['利率or信用'] == '信用债'].copy()
        self.ptmDist = self.getPTMDist(data_holding, '组合')
        self.ptmDist = self._convertTotalAssetRatio(self.ptmDist, index_col=['C_FUNDNAME', 'D_DATE'])
        self.ptmDist_ir = self.getPTMDist(data_ir, '利率债')
        self.ptmDist_ir = self._convertTotalAssetRatio(self.ptmDist_ir, index_col=['C_FUNDNAME', 'D_DATE'])
        self.ptmDist_cr = self.getPTMDist(data_cr, '信用债')
        self.ptmDist_cr = self._convertTotalAssetRatio(self.ptmDist_cr, index_col=['C_FUNDNAME', 'D_DATE'])
        res_list0.append(self.ptmDist)
        res_list0.append(self.ptmDist_ir)
        res_list0.append(self.ptmDist_cr)

        if self.bond_holdings.empty:
            self.port_VaR = pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'VaR_port'])
        else:
            # 组合VaR，95%置信度，涉及股票、债券、ABS等资产？
            calc_var_parameter(self.basedate)  # 更新VAR模型参数
            data_holding1 = self._dealAssetVaR()  # 债券资产(无股票）
            self.port_VaR = data_holding1.groupby(['C_FUNDNAME', 'D_DATE']).apply(self.PortVaR).reset_index().rename(columns={0: 'VaR_port'})
        res_list0.append(self.port_VaR)

        data_nav = self.nav_f.copy()
        self.ReturnGap = self.calcReturnGap(data_nav, data_holding).drop(columns=['avgYield_static'])
        res_list0.append(self.ReturnGap)
        
        # 关键期限
        self.port_KeyYear = self.calcKey_ratio_all(data_holding)
        res_list0.append(self.port_KeyYear)

        # 整合最终结果
        res0 = self.holdings[['C_FUNDNAME', 'D_DATE']].drop_duplicates().sort_values(by=['C_FUNDNAME', 'D_DATE'])
        for x in res_list0:
            if x['D_DATE'].dtypes == object:
                x['D_DATE'] = pd.to_datetime(x['D_DATE'])
            res0 = pd.merge(res0, x, on=['C_FUNDNAME', 'D_DATE'], how='left')

        res0['PORTFOLIO_CODE'] = res0['C_FUNDNAME'].map(self.fundname_to_code)
        self.res_holding = res0.copy()
        self.res_list_holding = res_list0.copy()
        logger.info('——>  Holdings Indices Done!')

    def calc_maxdd_lastyear(self):
        '''
        计算过去一年的最大回撤（滚动一年）
        :return:
        '''
        oneyear_before = self.basedate.replace(self.basedate[:4], str(int(self.basedate[:4]) - 1))
        data = self._dealNavData(oneyear_before)
        prods_mf = self.val.loc[self.val['L_FUNDTYPE'].isin(['1', '13', '14']), 'C_FUNDNAME'].drop_duplicates().tolist()
        data = data[data['C_FUNDNAME'].isin(prods_mf)].copy()

        func_nav = [self.DrawDown, self.DrawDown_dura, self.MaxDrawDown, self.MaxDrawDown_Dura, self.MaxDrawDown_rec]
        name_nav = ['drawdown_oneyear', 'drawdown_dura_oneyear', 'maxdrawdown_oneyear', 'maxdrawdown_dura_oneyear', 'maxdrawdown_rec_oneyear']

        res_list = []
        for func, func_name in zip(func_nav, name_nav):
            maxdd = data.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME']).apply(func).reset_index().rename(columns={0: func_name})
            maxdd['d_date'] = self.basedate
            res_list.append(maxdd)

        res_maxdd = data[['C_FUNDNAME']].drop_duplicates().sort_values(by=['C_FUNDNAME'])
        res_maxdd['d_date'] = self.basedate
        for x in res_list:
            res_maxdd = pd.merge(res_maxdd, x, on=['C_FUNDNAME', 'd_date'], how='left')
        logger.info('过去一年最大回撤计算完毕。')
        self.insert2db_single('rc_mr_maxdd', res_maxdd, t=self.basedate)

        return maxdd

    def calc_winning_ratio_t(self, period=30, day_cvt='calender'):
        '''
        计算过去一年的滚动胜率\n
        :param period: int, 计算收益率的区间, 默认计算30天的累计收益率
        :param day_cvt: string, day count convention, 指按照自然日calender计算还是按照交易日计算period
        :return: DataFrame
        '''
        day_convention_map = {'exchg': '1', 'ib': '2', 'calender': '9'}
        q = sqls_config['calender_day_t']['Sql']%(self.basedate, day_convention_map[day_cvt])
        date_list = self.db_risk.read_sql(q).sort_values(by=['d_date'])
        period_t = pd.to_datetime(date_list['d_date'].iloc[-1 * period - 1].strftime('%Y-%m-%d'))

        q_tradeday = sqls_config['nearest_tradeday']['Sql']%period_t.strftime('%Y-%m-%d')
        period_t_trade = pd.to_datetime(self.db_risk.read_sql(q_tradeday).iloc[0][0].strftime('%Y-%m-%d'))

        oneyear_before = self.basedate.replace(self.basedate[:4], str(int(self.basedate[:4]) - 1))
        data = self._dealNavData(oneyear_before)
        nav = data.loc[data['C_FUNDNAME'].isin(self.short_dura_bond_funds), :].copy()
        nav_n = nav.loc[nav['D_DATE'] == pd.to_datetime(self.basedate), ['C_FUNDNAME', 'D_DATE', 'NAV']].rename(columns={'NAV': 'nav_adj'})
        nav_t = data.loc[data['D_DATE'] == period_t_trade, ['C_FUNDNAME', 'NAV']].rename(columns={'NAV': 'nav_adj_t'})
        nav_m = pd.merge(nav_n, nav_t, on=['C_FUNDNAME'], how='left')
        nav_m['nav_adj_t'] = nav_m['nav_adj_t'].fillna(1)
        nav_m['ret_period'] = nav_m['nav_adj'] / nav_m['nav_adj_t'] - 1
        nav_m['ret_cnt'] = (nav_m['ret_period'] > 0).astype(int)
        nav_m['D_DATE'] = [x.strftime('%Y-%m-%d') for x in nav_m['D_DATE']]
        nav_m = nav_m.rename(columns={'C_FUNDNAME': 'c_fundname', 'D_DATE': 'd_date'})
        nav_m['PORTFOLIO_CODE'] = nav_m['c_fundname'].map(self.fundname_to_code)

        self.insert2db_single('rc_mr_winratio', nav_m, t=self.basedate)
        return nav_m

    def insert2db_dura(self):
        '''将个券的关键久期信息录入数据库'''
        if 'dura_asset' not in dir(self):
            self.duraP_holdings_ex, self.dura_asset = self.dealKey_duration()
        if 'data_keyRatio' not in dir(self):
            self.data_keyRatio = self.dealKey_year(self.duraP_holdings_ex)

        data_key_dura = self.duraP_holdings_ex.copy()
        if data_key_dura.empty:
            return None
        data_key_dura.columns = [x.replace('.0', '').replace('0.5Y', '6M') for x in data_key_dura.columns]
        new_cols = [str(x) + 'Y_Duration' for x in self.key_years if x > 0 and x < np.inf]
        new_cols = [x.replace('0.5Y', '6M') for x in new_cols]
        data_key_dura = data_key_dura.reindex(columns=new_cols + ['D_DATE'])
        data_key_dura = data_key_dura.reset_index()

        data_key_cash = self.dura_asset.copy()
        data_key_ratio = self.data_keyRatio.copy()
        data_key_ratio.columns = [x.replace('.0', '').replace('0.5Y', '6M') for x in data_key_ratio.columns]
        for data, table in zip([data_key_dura, data_key_cash, data_key_ratio], ['dpe_key_duration', 'dpe_key_cash', 'dpe_key_ratio']):
            update_codes = None if self.ptf_codes is None else data['code'].unique().tolist()
            self.insert2db_single(table, data, t=self.basedate, ptf_code=update_codes, code_colname='CODE')

#if __name__ == '__main__':
#    data_path = r'E:\RiskQuant\风险指标\Valuation\\'
#    save_path = r'E:\RiskQuant\风险指标\DailyIndicators\\'
#    t = '2021-04-22'
#
#    file_val = 'valuation' + t.replace('-', '') + '.json'
#    file_nav = '单位净值-基金A&专户.xlsx'
#    data_path_out = save_path + '%s\\'%t.replace('-', '')
#
#    MktIdx = MarketIndicators(data_path, file_nav, file_val, data_path_out)
#    MktIdx.calcNavRelated()