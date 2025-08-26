'''
@Description: Credit Indicators
@Author: Wangp
@Date: 2020-03-10 13:38:57
LastEditTime: 2021-06-15 15:25:50
LastEditors: Wangp
'''
import json
import numpy as np
import pandas as pd

from WindPy import w
w.start()

from .utils_ri.RiskIndicators import RiskIndicators, logger
from .utils_ri.Calc_DefaultRate import credit_transition_matrix
from .db import OracleDB, sqls_config

class CreditIndicators(RiskIndicators):
    def __init__(self, t, save_path, ptf_codes=None):
        # RiskIndicators.__init__(self, data_path, file_val)
        # self.HoldingsCleaning()
        self.basedate = t
        self.save_path = save_path
        self._format_ptf_codes(ptf_codes)

        self.path_masterData = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\MasterData\\'
        self._SectorMap = pd.DataFrame([np.arange(1, 8), ['主板', '中小板', '三板', '其他', '大宗交易系统', '创业板', '科创板']], index=['LISTEDSECTOR', '上市板块']).T
        self._InterestBond = ['政策银行债', '国债', '央行票据', '地方政府债', '政府支持机构债']
        self._CBond = ['可转债', '可交换债']

        self._loadFile()
        self._loadHoldings(self.save_path)
        # 剔除债券持仓中的 可转债，可交债
        self.bond_holdings = self.bond_holdings[~self.bond_holdings['WINDL1TYPE'].isin(self._CBond)].copy()
        self._loadMasterData()
        self._loadSpreadAD()
        self._loadIndustryID()
        self._formatCurveMap()
        self._loadTableColumns()

    def getDistribution(self, cols_name, col_values='F_ASSET', data_given=''):
        '''
        按照不同分类方式获取不同分布数据\n
        :param cols_name: string, 分类方式对应的列名
        :param col_values: string, 统计分类数据时用的市值列名，默认取F_Asset
        :param data_given: DataFrame, 用于计算分布的持仓数据
        :return: DataFrame
        '''
        if type(data_given) == str:
            data = self.bond_holdings.copy()
        else:
            data = data_given.copy()
        res1 = pd.pivot_table(data, values=col_values, index=['D_DATE', 'C_FUNDNAME'], columns=cols_name, aggfunc='sum')
        res2 = res1.sum(axis=1)
        res1 = res1.reset_index(level=['D_DATE', 'C_FUNDNAME'])
        res2 = res2.reset_index(level=['D_DATE', 'C_FUNDNAME']).rename(columns={0: 'Total'})
        
        res = pd.merge(res1, res2, on=['D_DATE', 'C_FUNDNAME'], how='outer').set_index(['D_DATE', 'C_FUNDNAME'])
        for col in res.columns:
            res[col] = res[col] / res['Total']

        res = res.reset_index(level=['D_DATE', 'C_FUNDNAME']).drop(columns=['Total'])
        
        return res

    def calcMirDistribution(self):
        '''持仓债券评级分布'''
        data = self.bond_holdings.copy()
        # CRMW保护的债券实际应为高评级
        crm_df = self.db_risk.read_sql(sqls_config['crwm_protect']['Sql'] % self.basedate)
        if not crm_df.empty:
            data = data.merge(crm_df, how='left', left_on=['C_FULLNAME', 'code'],
                              right_on=['c_fullname', 'crm_subjectcode']).fillna({'crm_mount': 0})
            data['评级分类'] = data.apply(lambda x: '高评级' if x.crm_mount == x.F_MOUNT else x.评级分类, axis=1)

        dist_mir = self.getDistribution('评级分类', data_given=data)

        # 还原为占总资产的比例
        data_fund = self.data_fund.copy()
        data_fund['总债券'] = data_fund[['债券', '可转债', 'ABS']].sum(axis=1)
        data_fund = data_fund.reindex(columns=['C_FUNDNAME', 'D_DATE', '总债券'])

        data_temp = pd.merge(dist_mir, data_fund, on=['C_FUNDNAME', 'D_DATE'], how='left')
        cols = dist_mir.drop(columns=['C_FUNDNAME', 'D_DATE']).columns
        for col in cols:
            data_temp[col] = data_temp[col] * data_temp['总债券']
        dist_mir = data_temp.drop(columns=['总债券'])
        return dist_mir

    def _retrieveR007Data(self, baseDate):
        if type(baseDate) != str:
            baseDate = pd.to_datetime(baseDate).strftime('%Y%m%d')
        else:
            baseDate = baseDate.replace('-', '')

        # 取R007的加权平均价格，而非收盘价，单位为%
        wind_temp = w.edb("M0041653", baseDate, baseDate,"Fill=Previous")
        
        return wind_temp.Data[0][0]

    def _dealAssetDefaultRate(self, data_holding=''):
        '''利用违约转移矩阵计算组合内各债券对应的预估违约率水平'''
        if type(data_holding) == str:
            data_holding = self.bond_holdings.copy()
        
        if 'DefaultRate' not in data_holding.columns:
            ctm = credit_transition_matrix()   # 创建违约率对象
            data_holding['RATE_LATESTMIR_CNBD'] = data_holding['RATE_LATESTMIR_CNBD'].fillna('NR').replace('无评级', 'NR')
            data_holding['PTMYEAR'] = data_holding['PTMYEAR'].fillna(0)
            data_holding['DefaultRate'] = [ctm.calc_defaultRate(x['RATE_LATESTMIR_CNBD'], x['PTMYEAR'])[1] for idx, x in data_holding.iterrows()]
        
        return data_holding

    def calcStaticReturn(self, data_holding=''):
        '''计算静态收益率ytm，不含利率债'''
        if type(data_holding) == str:
            data_holding = self.bond_holdings.copy()

        data_holding = data_holding[data_holding['利率or信用'] == '信用债'].copy()
        res = data_holding.groupby(['C_FUNDNAME', 'D_DATE']).apply(lambda x: (x['YIELD_CNBD'] * x['F_ASSETRATIO'] / 100).sum()).reset_index().rename(columns={0: 'StaticReturn'})
        
        return res

    def calcStaticReturn_adj(self, LGD=0.4, data_holding=''):
        '''
        计算经风险调整(违约率)、扣除杠杆成本后的静态收益率\n
        :param LGD: float, Loss Given Default, 默认0.4
        :param data_holding: DataFrame, 持仓证券表，默认取全量债券持仓
        :return: DataFrame
        '''
        if type(data_holding) == str:
            data_holding = self.bond_holdings.copy()

            def Leverage(self):
                res = self.data_fund[['D_DATE', 'C_FUNDNAME', '卖出回购', 'NetAsset', 'TotalAsset']].copy()
                res['杠杆率'] = res['卖出回购'].fillna(0) * res['TotalAsset'] / res['NetAsset'] + 1
                res = res.drop(columns=['卖出回购', 'NetAsset', 'TotalAsset'])
                
                return res

            def LeverageCost(self):
                if not '_lev_all' in dir(self):
                    res = pd.DataFrame(columns=['D_DATE', 'C_FUNDNAME', 'Lev_cost'])
                else:
                    res = self._lev_all[['D_DATE', 'C_FUNDNAME', 'Lev_cost']].copy()
                    res['Lev_cost'] = res['Lev_cost'] * 100
                
                return res

        data_holding.loc[data_holding['利率or信用'] == '利率债', 'RATE_LATESTMIR_CNBD'] = '无评级'     # 利率债评级改为无评级
        data_holding = self._dealAssetDefaultRate(data_holding)
        data_holding.loc[data_holding['利率or信用'] == '利率债', 'DefaultRate'] = 0     # 利率债违约率为0
        # data_holding = data_holding[data_holding['利率or信用'] == '信用债'].copy()
        # baseDate = data_holding['D_DATE'].iloc[0]
        # r007 = self._retrieveR007Data(baseDate)
        # data_holding['R007'] = r007
        data_holding['StaticReturn_adj'] = data_holding['YIELD_CNBD'] - data_holding['DefaultRate'] * 100 * LGD

        res_init = data_holding.groupby(['C_FUNDNAME', 'D_DATE']).apply(lambda x: (x['StaticReturn_adj'] * x['F_ASSETRATIO'] / 100).sum()).reset_index().rename(columns={0: 'StaticReturn_adj_orig'})
        res_lev1 = Leverage(self)
        res_lev2 = LeverageCost(self)
        res = pd.merge(res_init, res_lev1, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res = pd.merge(res, res_lev2, on=['C_FUNDNAME', 'D_DATE'], how='left')
        res['StaticReturn_adj'] = res['StaticReturn_adj_orig'] - res['Lev_cost'].fillna(0) * (res['杠杆率'] - 1)
        res = res.drop(columns=['Lev_cost', '杠杆率'])
        # res_basic = self.data_fund[['C_FUNDNAME', 'D_DATE', 'NetAsset', 'TotalAsset']].copy()

        # res = pd.merge(res_init, res_basic, on=['C_FUNDNAME', 'D_DATE'], how='left')
        # res['lev'] = res['TotalAsset'] / res['NetAsset']
        # res['StaticReturn_adj'] = res['StaticReturn_adj_orig'] / res['lev']
        # res = res.drop(columns=['NetAsset', 'TotalAsset', 'lev'])
        
        return data_holding, res

    def calcStaticSpread_adj(self, LGD=0.4, data_holding=''):
        '''
        计算经风险调整后的静态利差(与杠杆成本无关)\n
        :param LGD: float, Loss Given Default, 默认0.4
        :param data_holding: DataFrame, 持仓证券表，默认取全量债券持仓
        :return: DataFrame
        '''
        if type(data_holding) == str:
            data_holding = self.bond_holdings.copy()

        data_holding = self._dealAssetDefaultRate(data_holding)
        data_holding = data_holding[data_holding['利率or信用'] == '信用债'].copy()
        data_holding['StaticSpread_adj'] = data_holding['spread_gk'] - data_holding['DefaultRate'] * 100 * LGD

        res = data_holding.groupby(['C_FUNDNAME', 'D_DATE']).apply(lambda x: (x['StaticSpread_adj'] * x['F_ASSETRATIO']).sum() / x['F_ASSETRATIO'].sum()).reset_index().rename(columns={0: 'StaticSpread_adj'})
        
        return data_holding, res        

    def calcDefaultRate_port(self, data_holding=''):
        '''计算组合层面的加权违约率水平'''
        if type(data_holding) == str:
            data_holding = self.bond_holdings.copy()

        data_holding = self._dealAssetDefaultRate(data_holding)
        
        res = data_holding.groupby(['C_FUNDNAME', 'D_DATE']).apply(lambda x: (x['DefaultRate'] * x['F_ASSETRATIO']).sum()/ x['F_ASSETRATIO'].sum()).reset_index().rename(columns={0: 'DefaultRate_lev'})
        
        return data_holding, res
    
    def calcExpectedLossRate(self, LGD=0.4, data_holding=''):
        '''计算组合层面的预期损失率水平，默认LGD为0.4'''
        if type(data_holding) == str:
            data_holding = self.bond_holdings.copy()

        data_holding = self._dealAssetDefaultRate(data_holding)
        res = data_holding.groupby(['C_FUNDNAME', 'D_DATE']).apply(lambda x: (x['DefaultRate'] * LGD * x['F_ASSETRATIO'] / 100).sum()).reset_index().rename(columns={0: 'ExpectedLossRate'})

        return data_holding, res

    # 组合利差N日变化，N指交易日
    def getSpreadDiff_ts(self, data, period=10, col_spread='spread_ad'):
        '''not used'''
        new_diffName = col_spread.replace('_ad', '')+'_diff_t%d'%period
        data1 = data.sort_values(by=['C_FUNDNAME', 'D_DATE'], ascending=True)
        res = data1.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME'])[col_spread].diff(periods=period).rename(new_diffName).reset_index()

        return res

    def __dealIndSpread_t(self, data_t):
        '''计算各持仓券(不含利率债和可转债)相对于行业利差而言的超额利差，记做spread_ind_ad'''
        data_t1 = self.bond_holdings.loc[self.bond_holdings['城投or产业'].isin(['产业债', '金融债', '城投债']) & ~self.bond_holdings['WINDL1TYPE'].isin(['可转债', '可交换债']),
                                        ['C_FUNDNAME', 'D_DATE' , 'code', '行业利差']].rename(columns={'行业利差': 'IndSpread_t1'})
        data_temp = pd.merge(data_t, data_t1, on=['C_FUNDNAME', 'D_DATE' , 'code'], how='left')
        data_temp['行业利差'] = data_temp['行业利差'].fillna(value=data_temp['IndSpread_t1'])    # 若T日以前无行业利差数据，则用当期行业利差填充
        data_temp['spread_ind_ad'] = data_temp['spread_rating'] - data_temp['行业利差']

        return data_temp.drop(columns=['IndSpread_t1'])

    def getSpreadDiff(self, data, period=20, col_spread='spread_ad'):
        '''
        计算组合利差的N日变化，N指交易日。记作列名spread_diff_tn\n
        :param data: DataFrame, 持仓数据
        :param period: int, 计算变化的时间区间, 指交易日
        :param col_spread: string, 用于计算利差变化的基础利差字段名，默认为绝对利差spread_ad
        :return: DataFrame
        '''
        spread_t0 = data.copy()
        date_list = data['D_DATE'].drop_duplicates().tolist()
        new_colName = col_spread.replace('_ad', '_t%d'%period)
        new_diffName = col_spread.replace('_ad', '')+'_diff_t%d'%period
        
        data_t = pd.DataFrame()
        for temp_date in date_list:
            data_temp = self._getTdaysoffsetVal(temp_date, period)      # 取前t日的估值表
            if data_temp.shape[0] == 0:
                return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', new_colName, new_diffName])
            if 'ind' in col_spread:
                data_temp = self.__dealIndSpread_t(data_temp)           # 计算t日估值表的债券持仓对应的利差数据
            data_t = pd.concat([data_t, data_temp])

        data_t1 = data_t.loc[data_t['RATE_LATESTMIR_CNBD'] != '评级_利率债',
                             ['C_FUNDNAME', 'D_DATE', 'code', 'C_SUBNAME_BSH', 'F_ASSET', 'F_ASSETRATIO', col_spread, 'bp_ad']].copy()
        spread_t1 = data_t1[~data_t1[col_spread].isna()].groupby(['C_FUNDNAME', 'D_DATE']).apply(lambda x: (x['F_ASSET']*x[col_spread]).sum()/x['F_ASSET'].sum()).reset_index().rename(columns={0: new_colName})

        spread = pd.merge(spread_t0, spread_t1, on=['C_FUNDNAME', 'D_DATE'], how='left')
        spread[new_diffName] = spread[col_spread] - spread[new_colName]

        return spread[['C_FUNDNAME', 'D_DATE', new_colName, new_diffName]].copy()

    def calcDispersion(self):
        '''计算行业分布的分散度, mean - 2*std'''
        # 剔除利率债
        data1 = self.bond_holdings[self.bond_holdings['INDUSTRY_SW'] != '行业_利率债'].groupby(['D_DATE', 'C_FUNDNAME', 'INDUSTRY_SW'])['F_ASSET'].sum().reset_index()
        data2 = data1.groupby(['D_DATE', 'C_FUNDNAME'])['F_ASSET'].sum().reset_index().rename(columns={'F_ASSET': 'TotalBond'})
        data = pd.merge(data1, data2, on=['D_DATE', 'C_FUNDNAME'], how='left')
        data['IndRatio'] = data['F_ASSET'] / data['TotalBond']
        
        res = data.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: x['IndRatio'].mean() - 2*x['IndRatio'].std(ddof=0)).reset_index().rename(columns={0: 'Dispersion'})
        res['Dispersion'] = abs(res['Dispersion']) * 100

        return res

    def getConcentrationCoV(self, ind_n, col='Industry_N_asset'):
        '''获取组合前n大行业集中度持仓占比的变异系数'''
        res = ind_n.groupby(['C_FUNDNAME'])[col].apply(self.calcCoV).reset_index().rename(columns={col: 'CoV_indN'})

        return res
    
    def getMAE(self, data, col):
        '''
        计算各组合给定统计列的MAE(Median Absolute Error)\n
        :param data: DataFrame
        :param col: string
        :return: DataFrame
        '''
        res = data.groupby(['C_FUNDNAME'])[col].apply(self.calcMAE_basic).reset_index().rename(columns={col: col+'_MAE'})

        return res

    def _getTdaysoffsetVal(self, baseDay, t):
        '''
        获取T日前的估值表并清洗数据\n
        :param baseDay: string/Timestamp, 基期
        :param t: float, 日期偏移量, 以交易日为单位
        :return: DataFrame
        '''
        if type(baseDay) != str:
            baseDay = baseDay.strftime('%Y-%m-%d')
        Tday = w.tdaysoffset(-1*t, baseDay, "").Data[0][0].strftime('%Y-%m-%d')
        data_t = self._loadFile(Tday)
        data_t = self.HoldingsCleaning(data_t)[0]
        data_t['Date_orig'] = data_t['D_DATE'].copy()
        data_t['D_DATE'] = pd.to_datetime(baseDay)     # 记为基期，方便后续merge
        
        return data_t

    def _dealTdayCredit(self, baseDay, t):
        '''
        计算T日的收益率及利差情况\n
        :param baseDay: string, 基期
        :param t: int, 日期偏移量
        :return: DataFrame
        '''
        data_t = self._getTdaysoffsetVal(baseDay, t)
        if data_t.shape[0] == 0:
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', 'code', 'w_%dbefore'%t, 'IR_%dbefore'%t, 'curve_%dbefore'%t, 'curveInd_%dbefore'%t, 'spreadRating_%dbefore'%t])
        
        data1 = data_t[data_t['城投or产业'].isin(['产业债', '金融债', '城投债']) & ~data_t['WINDL1TYPE'].isin(['可转债', '可交换债'])].copy()
        data1['w_%dbefore'%t] = data1.groupby(['C_FUNDNAME'])['F_ASSET'].apply(lambda x: x/x.sum())
        data1['IR_%dbefore'%t] = data1['RATE_LATESTMIR_CNBD'].copy()
        data1['curve_%dbefore'%t] = data1['曲线代码'].copy()
        data1['curveInd_%dbefore'%t] = data1['行业利差ID'].copy()
        data1['yield_%dbefore'%t] = data1['YIELD_CNBD'].copy()
        data1['IndSpread_%dbefore'%t] = data1['行业利差'].copy()
        data1['spreadRating_%dbefore'%t] = data1['spread_rating'].copy()
        data1['spreadInd_%dbefore'%t] = data1['spread_ind_ad'].copy()

        return data1[['C_FUNDNAME', 'D_DATE', 'code', 'w_%dbefore'%t, 'IR_%dbefore'%t, 'curve_%dbefore'%t, 'curveInd_%dbefore'%t, 'yield_%dbefore'%t, 'IndSpread_%dbefore'%t, 'spreadRating_%dbefore'%t, 'spreadInd_%dbefore'%t]].copy()

    def _CreditDown_asset(self, data, t=20):
        '''
        计算个券层面信用利差及行业利差(不变价)的变动，即
            假设维持T日评级、T日行业分类不变，按照市场情况发展至今的利差变动，相当于消除市场利差变动的影响\n
        :param data: DataFrame, 基期的债券持仓表
        :param t: int, 日期偏移量
        :return: DataFrame
        '''
        date_list = data['D_DATE'].drop_duplicates().tolist()
        data_c = data[data['城投or产业'].isin(['产业债', '金融债', '城投债']) & ~data['WINDL1TYPE'].isin(
            ['可转债', '可交换债'])].sort_values(by=['C_FUNDNAME', 'D_DATE'])
        data_c['weight'] = data_c.groupby(['C_FUNDNAME', 'D_DATE'])['F_ASSET'].apply(lambda x: x/x.sum())
        data_t = pd.DataFrame()
        for temp_date in date_list:
            data_temp = self._dealTdayCredit(temp_date, t)
            data_t = pd.concat([data_t, data_temp])
        data_c = pd.merge(data_c, data_t, on=['C_FUNDNAME', 'D_DATE', 'code'])

        # T日前的评级曲线代码 & 当前剩余期限 & 当前曲线行情(benchmark_t)
        data_m = pd.merge(data_c.drop(columns=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY']), self._benchmark_rating,
                          left_on=['D_DATE', 'curve_%dbefore'%t, '剩余期限'],
                          right_on=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY'], how='left')
        # T日前的行业利差代码 & 当前利差行情(IndSpread_t)
        data_m = pd.merge(data_m.drop(columns=['行业利差ID', '行业利差']), self._benchmark_ind,
                          left_on=['D_DATE', 'curveInd_%dbefore'%t], right_on=['D_DATE', '行业利差ID'], how='left')
        data_m = data_m.rename(columns={'YIELD': 'benchmark_t%d'%t, '行业利差': 'IndSpread_t%d'%t})
        data_m['spreadInd_%dbefore'%t] = data_m['spreadInd_%dbefore'%t].fillna(value=data_m['spreadRating_%dbefore'%t] - data_m['IndSpread_t%d'%t])
        data_m['spreadT1_%d'%t] = data_m['YIELD_CNBD'] - data_m['benchmark_t%d'%t]  # T日评级在当前的利差水平
        data_m['spreadIndT1_%d'%t] = data_m['YIELD_CNBD'] - data_m['benchmark_t%d'%t] - data_m['IndSpread_t%d'%t].fillna(value=data_m['IndSpread_%dbefore'%t])
        # 若维持T日评级不变，在当前可能的利差水平与T日实际利差水平的变动
        data_m['creditDown_t%d'%t] = data_m['spreadT1_%d'%t] - data_m['spreadRating_%dbefore'%t]
        # 若维持T日评级及行业不变，在当前可能的利差水平与T日实际利差水平的变动
        data_m['creditDownInd_t%d'%t] = data_m['spreadIndT1_%d'%t] - data_m['spreadInd_%dbefore'%t]

        return data_m[['C_FUNDNAME', 'D_DATE', 'code', 'w_%dbefore'%t, 'IR_%dbefore'%t, 'curve_%dbefore'%t, 'curveInd_%dbefore'%t, 'benchmark_t%d'%t, 
                        'spreadT1_%d'%t, 'spreadIndT1_%d'%t, 'spreadRating_%dbefore'%t, 'spreadInd_%dbefore'%t, 'creditDown_t%d'%t, 'creditDownInd_t%d'%t]].copy()

    def calcPortCreditDown(self, t=20):
        '''
        计算组合层面的信用利差、超额行业利差(不变价, 市场中性)的变动，按个券在信用债总仓位里的权重加权\n
        :param t: int, 日期偏移量
        :return: DataFrame
        '''
        data = self.bond_holdings[self.bond_holdings['城投or产业'].isin(['产业债', '金融债', '城投债']) 
                                    & ~self.bond_holdings['WINDL1TYPE'].isin(['可转债', '可交换债'])].sort_values(by=['C_FUNDNAME', 'D_DATE'])
        data['weight'] = data.groupby(['C_FUNDNAME', 'D_DATE'])['F_ASSET'].apply(lambda x: x/x.sum())
        data_asset = self._CreditDown_asset(data, t)
        data_m = pd.merge(data, data_asset, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')
        
        port_creditDown1 = data_m.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x['weight'] * x['spreadT1_%d'%t]).sum()).reset_index().rename(columns={0: 'creditSpread'})
        port_creditDown2 = data_m.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x['w_%dbefore'%t] * x['spreadRating_%dbefore'%t]).sum()).reset_index().rename(columns={0: 'creditSpread_t%d'%t})
        port_creditDown = pd.merge(port_creditDown1, port_creditDown2, on=['D_DATE', 'C_FUNDNAME'], how='outer')
        # 超额信用利差不变价
        port_creditDown['diff_%d'%t] = port_creditDown['creditSpread'] - port_creditDown['creditSpread_t%d'%t]

        port_creditDown1_ind = data_m.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x['weight'] * x['spreadIndT1_%d'%t]).sum()).reset_index().rename(columns={0: 'creditSpread_ind'})
        port_creditDown2_ind = data_m.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x['w_%dbefore'%t] * x['spreadInd_%dbefore'%t]).sum()).reset_index().rename(columns={0: 'creditSpread_ind_t%d'%t})
        port_creditDown_ind = pd.merge(port_creditDown1_ind, port_creditDown2_ind, on=['D_DATE', 'C_FUNDNAME'], how='outer')
        # 超额行业利差不变价
        port_creditDown_ind['diffInd_%d'%t] = port_creditDown_ind['creditSpread_ind'] - port_creditDown_ind['creditSpread_ind_t%d'%t]

        return data_asset, port_creditDown, port_creditDown_ind

    def _CreditDown_asset_ts(self, data, t=20):
        '''计算个券层面利差(不变价)变动的时间序列，要求持仓数据覆盖的是时间序列'''
        data_c = data[data['城投or产业'].isin(['产业债', '金融债', '城投债']) & ~data['WINDL1TYPE'].isin(['可转债', '可交换债'])].sort_values(by=['C_FUNDNAME', 'D_DATE'])
        data_c['weight'] = data_c.groupby(['C_FUNDNAME', 'D_DATE'])['F_ASSET'].apply(lambda x: x/x.sum())
        data_c['w_%dbefore'%t] = data_c.groupby(['C_FUNDNAME', 'code'])['weight'].shift(t)
        data_c['IR_%dbefore'%t] = data_c.groupby(['C_FUNDNAME', 'code'])['RATE_LATESTMIR_CNBD'].shift(t)
        data_c['curve_%dbefore'%t] = data_c.groupby(['C_FUNDNAME', 'code'])['CURVECODE'].shift(t)
        data_c['curveInd_%dbefore'%t] = data_c.groupby(['C_FUNDNAME', 'code'])['行业利差ID'].shift(t)

        data_m = pd.merge(data_c.drop(columns=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY']), self._benchmark_rating, left_on=['D_DATE', 'curve_%dbefore'%t, '剩余期限'], right_on=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY'], how='left')
        data_m = pd.merge(data_m.drop(columns=['行业利差ID', '行业利差']), self._benchmark_ind, left_on=['D_DATE', 'curveInd_%dbefore'%t], right_on=['D_DATE', '行业利差ID'], how='left')
        data_m = data_m.rename(columns={'YIELD': 'benchmark_t%d'%t, '行业利差': 'IndSpread_t%d'%t})
        data_m['spread_t%d'%t] = data_m['YIELD_CNBD'] - data_m['benchmark_t%d'%t]                          # 已调整溢差
        data_m['spreadInd_t%d'%t] = data_m['YIELD_CNBD'] - data_m['benchmark_t%d'%t] - data_m['IndSpread_t%d'%t]
        data_m['creditDown_t%d'%t] = data_m['spread_rating'] - data_m['spread_t%d'%t]
        data_m['creditDownInd_t%d'%t] = data_m['creditDown_t%d'%t] - data_m['IndSpread_t%d'%t]

        return data_m[['C_FUNDNAME', 'D_DATE', 'code', 'w_%dbefore'%t, 'IR_%dbefore'%t, 'curve_%dbefore'%t, 'benchmark_t%d'%t, 'spread_t%d'%t, 'spreadInd_t%d'%t, 'creditDown_t%d'%t, 'creditDownInd_t%d'%t]].copy()

    def calcPortCreditDown_ts(self, t=20):
        '''计算组合层面利差(不变价)变动的时间序列，要求持仓数据覆盖的是时间序列'''
        data = self.bond_holdings[self.bond_holdings['城投or产业'].isin(['产业债', '金融债', '城投债'])
                                    & ~self.bond_holdings['WINDL1TYPE'].isin(['可转债', '可交换债'])].sort_values(by=['C_FUNDNAME', 'D_DATE'])
        data['weight'] = data.groupby(['C_FUNDNAME', 'D_DATE'])['F_ASSET'].apply(lambda x: x/x.sum())
        data_asset = self._CreditDown_asset_ts(data, t)
        data_m = pd.merge(data, data_asset, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')
        
        port_creditDown1 = data_m.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x['weight'] * x['spread_rating']).sum()).reset_index().rename(columns={0: 'creditSpread'})
        port_creditDown2 = data_m.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x['w_%dbefore'%t] * x['spread_t%d'%t]).sum()).reset_index().rename(columns={0: 'creditSpread_t%d'%t})
        port_creditDown1 = port_creditDown1[port_creditDown1['creditSpread'] != 0].copy()                                    # 组合creditSpread为0说明在该时间点weight*spread均为空，应剔除
        port_creditDown2 = port_creditDown2[port_creditDown2['creditSpread_t%d'%t] != 0].copy()
        port_creditDown = pd.merge(port_creditDown1, port_creditDown2, on=['D_DATE', 'C_FUNDNAME'], how='outer')
        port_creditDown['diff_%d'%t] = port_creditDown['creditSpread'] - port_creditDown['creditSpread_t%d'%t]

        port_creditDown1_ind = data_m.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x['weight'] * x['spread_ind_ad']).sum()).reset_index().rename(columns={0: 'creditSpread_ind'})
        port_creditDown2_ind = data_m.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x['w_%dbefore'%t] * x['spreadInd_t%d'%t]).sum()).reset_index().rename(columns={0: 'creditSpread_t%d_ind'%t})
        port_creditDown_ind = pd.merge(port_creditDown1_ind, port_creditDown2_ind, on=['D_DATE', 'C_FUNDNAME'], how='outer')
        port_creditDown_ind['diffInd_%d'%t] = port_creditDown_ind['creditSpread_ind'] - port_creditDown_ind['creditSpread_t%d_ind'%t]

        return data_asset, port_creditDown, port_creditDown_ind

    def CalculateAll(self):
        if self.bond_holdings.empty:
            self.resAll = pd.DataFrame(columns=self.column_dict['credit'])
            logger.info('产品无债券持仓，无需计算信用风险指标')
            return None

        # 高中低评级分布，占全部债券投资比例，含高评级_利率债
        self.dist_mir2 = self.calcMirDistribution()

        corpbond_holdings = self.bond_holdings[self.bond_holdings['RATE_LATESTMIR_CNBD'] != '评级_利率债'].copy()
        if corpbond_holdings.empty:
            self.resAll = self.dist_mir2.reindex(columns=self.column_dict['credit'])
            self.resAll['PORTFOLIO_CODE'] = self.resAll['C_FUNDNAME'].map(self.fundname_to_code)
            return None

        # 违约率相关指标（不含利率债）
        self.StaticReturn = self.calcStaticReturn()
        self.bond_holdings, self.StaticReturn_adj = self.calcStaticReturn_adj()
        self.bond_holdings, self.StaticSpread_adj = self.calcStaticSpread_adj()
        self.bond_holdings, self.DefaultRate_port = self.calcDefaultRate_port()
        self.bond_holdings, self.ExpectedLossRate = self.calcExpectedLossRate()

        # 利差类 -- 信用债
        self.spread_total = corpbond_holdings.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x['F_ASSET']*x['spread_gk']).sum()/x['F_ASSET'].sum()).reset_index().rename(columns={0: 'spread_ad'})                       # 组合利差
        # 超额利差 -- 产业债
        self.spread_ind_total = self.bond_holdings[self.bond_holdings['城投or产业'].isin(['产业债', '金融债', '城投债']) & ~self.bond_holdings['WINDL1TYPE'].isin(['可转债', '可交换债']) & ~self.bond_holdings['spread_ind_ad'].isna()].groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x['F_ASSET']*x['spread_ind_ad']).sum()/x['F_ASSET'].sum()).reset_index().rename(columns={0: 'spread_ind_ad'})                       # 组合利差
        if self.spread_ind_total.shape[0] == 0:
            self.spread_ind_total = pd.DataFrame(columns=['D_DATE', 'C_FUNDNAME', 'spread_ind_ad'])
            
        # 利差变动
        self.spread_diff_20 = self.getSpreadDiff(self.spread_total, period=20, col_spread='spread_ad').dropna()                  # 日度变化
        self.spread_diff_60 = self.getSpreadDiff(self.spread_total, period=60, col_spread='spread_ad').dropna()                  # 周度变化
        self.spread_diff_120 = self.getSpreadDiff(self.spread_total, period=120, col_spread='spread_ad').dropna()                # 月度变化
    
        # 利差变动 -- 产业债超额利差
        self.spread_ind_diff_20 = self.getSpreadDiff(self.spread_ind_total, period=20, col_spread='spread_ind_ad').dropna()                  # 日度变化
        self.spread_ind_diff_60 = self.getSpreadDiff(self.spread_ind_total, period=60, col_spread='spread_ind_ad').dropna()                  # 周度变化
        self.spread_ind_diff_120 = self.getSpreadDiff(self.spread_ind_total, period=120, col_spread='spread_ind_ad').dropna()                # 月度变化

        # 合并各类spread的N日变化值
        spread_diff = pd.merge(self.spread_total, self.spread_ind_total, on=['D_DATE', 'C_FUNDNAME'], how='left')
        spread_list = [self.spread_diff_20, self.spread_diff_60, self.spread_diff_120, self.spread_ind_diff_20, self.spread_ind_diff_60, self.spread_ind_diff_120]
        for x in spread_list:
            spread_diff = pd.merge(spread_diff, x, on=['D_DATE', 'C_FUNDNAME'], how='left')
        self.spread_diff = spread_diff.copy()

        # 行业分散度指标
        self.dispersion = self.calcDispersion()

        # 信用资质下沉指标，交易日：月度、季度、半年、全年
        self.asset_creditDown_t20, self.port_creditDown_t20, self.port_creditDown_ind_t20 = self.calcPortCreditDown(t=20)
        self.asset_creditDown_t60, self.port_creditDown_t60, self.port_creditDown_ind_t60 = self.calcPortCreditDown(t=60)
        self.asset_creditDown_t120, self.port_creditDown_t120, self.port_creditDown_ind_t120 = self.calcPortCreditDown(t=120)

        self.asset_creditDown = self.asset_creditDown_t20.copy()
        self.port_creditDown = self.port_creditDown_t20.copy()
        self.port_creditDown_ind = self.port_creditDown_ind_t20.copy()
        batch_asset = [(self.asset_creditDown_t60, self.port_creditDown_t60, self.port_creditDown_ind_t60), (self.asset_creditDown_t120, self.port_creditDown_t120, self.port_creditDown_ind_t120)]
        for asset, port, port_ind in batch_asset:
            self.asset_creditDown = pd.merge(self.asset_creditDown, asset, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')
            self.port_creditDown = pd.merge(self.port_creditDown, port.drop(columns=['creditSpread']), on=['C_FUNDNAME', 'D_DATE'], how='left')
            self.port_creditDown_ind = pd.merge(self.port_creditDown_ind, port_ind.drop(columns=['creditSpread_ind']), on=['C_FUNDNAME', 'D_DATE'], how='left')
        
        self.asset_creditDown_holdings_ex = self.asset_creditDown.copy()

        self.resAll = self.IntegrateAll()

        if '无评级' not in self.resAll.columns:
            cols = list(self.resAll.columns)[:4] + ['无评级'] + list(self.resAll.columns)[4:]
            self.resAll = self.resAll.reindex(columns=cols)

# if __name__ == "__main__":
#     data_path = r'E:\RiskQuant\风险指标\\'
#     file_val = 'valuation3.json'
#     start = time.time()
#     CreditIdx = CreditIndicators(data_path, file_val)
#     CreditIdx.CalculateAll()
#     CreditIdx.resAll = CreditIdx.IntegrateAll()
#     CreditIdx.resAll.to_excel(CreditIdx.data_path + 'CreditIndex2019Q3.xlsx', index=False)
# #    holding_all = CreditIdx.IntegrateHoldings()
# #    holding_all.to_excel(CreditIdx.data_path + 'Holdings20191227.xlsx', index=False)
#     print('--'*20)
#     print('Credit Index Done!')
#     end = time.time()
#     print('Total Time Cost:', end - start)