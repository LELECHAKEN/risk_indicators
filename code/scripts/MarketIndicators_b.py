'''
@Description: 
@Author: Wangp
@Date: 2020-04-21 16:32:45
@LastEditTime: 2020-06-12 13:55:48
@LastEditors: Wangp
'''
import numpy as np
import pandas as pd

from .MarketIndicators import MarketIndicators, logger


class BenchmarkRiskReturn(MarketIndicators):
    def __init__(self, t, ptf_codes=None):
        '''
        风险收益指标 - 比较基准\n
        :param t: string, 计算基期
        '''
        self.basedate = t
        self._format_ptf_codes(ptf_codes)
        self._connectingRiskDB()
        self._loadBenchmarkReturn()
        self._fundNameMapping()
        self.nav_f = self._benchmark_ret.drop(columns='bch_type').rename(columns={'ret_b': 'ret'})
        # self.nav_f = pd.read_excel(data_path + file_nav).rename(columns={'基金名称': 'C_FUNDNAME', '日期': 'D_DATE', '收益率': 'ret'})
        self.nav_f = self.nav_f.sort_values(by=['C_FUNDNAME_o32', 'D_DATE']).reset_index(drop=True)
        self.startDate = self.nav_f['D_DATE'].min()   # datetime格式，但后面未用到过
        self.endDate = self.basedate                  # string格式

        self._dealNAV()

    def _restoreNAV(self, x):
        '''将期初单位净值恢复至1'''
        y = x + 1
        y.iloc[0] = 1
        nav = y.cumprod()

        return nav

    def _dealNAV(self):
        data_coding = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl').rename(columns={'O32产品名称': 'C_FUNDNAME_o32'})
        nav_f = self.nav_f.copy()
        nav_f = pd.merge(nav_f, data_coding[['C_FUNDNAME_o32', '产品设立日期', '估值系统简称']], on=['C_FUNDNAME_o32'], how='left')
        nav_f = nav_f[nav_f['D_DATE'] >= nav_f['产品设立日期']].dropna(subset=['C_FUNDNAME_o32'])
        nav_f['NAV'] = nav_f.groupby(['C_FUNDNAME_o32'])['ret'].apply(self._restoreNAV)
        nav_f['PORTFOLIO_CODE'] = nav_f['估值系统简称'].map(self.fundname_to_code)

        if self.ptf_codes is not None:
            nav_f = nav_f[nav_f['PORTFOLIO_CODE'].isin(self.ptf_codes)].copy()

        self.nav_f = nav_f.drop(columns=['C_FUNDNAME_o32']).rename(columns={'估值系统简称': 'C_FUNDNAME'})

    def Treynor(self, netvalue_df=0, risk_free_list=0):
        '''
        Treynor比率, (区间收益率 - 无风险收益率) / beta \n
        :param netvalue_df: DataFrame, 收益率序列
        :param risk_free_list: DataFrame or Series, 无风险利率序列
        :return: float
        '''
        if type(netvalue_df) == int:
            netvalue_df = self.nav_f
        if type(risk_free_list)==int:
            risk_free_list = pd.DataFrame([0 for index in range(netvalue_df.shape[0])])[0]
        ####Treynor比率
        ret_f, ret_free = self.timeAlign(netvalue_df[['ret', 'NAV']], risk_free_list)
        
        try:
            risk_free = (ret_free['risk_free'].iloc[1:] + 1).product() - 1                 # 第一天即起始日不应有收益率，故剔除第一天的收益率数据
            beta2 = 1
            return (self.ReturnRate(ret_f) - risk_free) / beta2
        except:
            # print(netvalue_df.index[0], ret_f.shape, ret_free.shape, ret_b.shape)
            return np.nan

    def calcSingleIndex(self, func, endDate, risk_free_list, period='Y'):
        '''计算某一个指标'''
        if type(endDate) == str:
            endDate = pd.to_datetime(endDate)
        startDate = pd.to_datetime(self.get_pastyear_date(endDate, n=1))
        
        data = self.nav_f.loc[(self.nav_f['D_DATE'] >= startDate) & (self.nav_f['D_DATE'] <= endDate)].drop_duplicates().reset_index(drop=True)
        res = data.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME']).apply(func, risk_free_list).reset_index().rename(columns={0: func.__name__})
        res['D_DATE'] = endDate
        
        return res

    def calcNavRelated(self, endDate='', period='Y'):
        '''计算所有风险收益相关指标，算法继承自MarketIndicators'''
        if endDate == '':
            endDate = pd.to_datetime(self.endDate)
        startDate = pd.to_datetime(self.get_pastyear_date(endDate, n=1))

        data = self.nav_f.loc[(self.nav_f['D_DATE'] >= startDate) & (self.nav_f['D_DATE'] <= endDate)].drop_duplicates().reset_index(drop=True)
        if data.empty:
            logger.error('产品在表 benchmark_bb_fund 中无数据，无法计算基准指标')
            self.res_nav = pd.DataFrame()
            return None

        self._risk_free_list = self._getBenchmarkReturn(benchmarkCode='SHIBOR3M.IR', startDate=startDate, endDate=endDate).drop(columns=['ret_idx'])
        self._risk_free_list['risk_free'] = self._risk_free_list['CLOSE'] / (100*365)

        func_nav = [self.Volatility_down, self.DrawDown, self.MaxDrawDown, self.pain_index, self.hit_rate, self.GPR, self.PainRatio]
        name_nav = [x.__name__ for x in func_nav]
        func_free = [self.Calmar_index, self.Sharpe, self.Sortino, self.Treynor]
        name_free = [x.__name__ for x in func_free]

        res_list = []
        for func, name in zip(func_nav, name_nav):
            res_temp = data.groupby(['C_FUNDNAME']).apply(func).reset_index().rename(columns={0: name})
            res_list.append(res_temp)
        # 下述函数需risk_free_list，为series，会在函数里处理成区间收益risk_free
        for func, name in zip(func_free, name_free):
            res_temp = data.set_index(['C_FUNDNAME', 'D_DATE']).groupby(['C_FUNDNAME']).apply(func, self._risk_free_list).reset_index().rename(columns={0: name})
            res_list.append(res_temp)

        res = data[['C_FUNDNAME']].drop_duplicates().sort_values(by=['C_FUNDNAME'])
        res['D_DATE'] = pd.to_datetime(endDate)
        for x in res_list:
            res = pd.merge(res, x, on='C_FUNDNAME', how='left')
        
        res['Beta'] = 1
        res['Jenson'] = 0
        res['Information_Ratio'] = 0

        res['PORTFOLIO_CODE'] = res['C_FUNDNAME'].map(self.fundname_to_code)
        self.res_nav = res.replace(np.inf, np.nan).replace((-1)*np.inf, np.nan)
        self.res_list_nav = res_list.copy()            
        print('——>  Nav Indices Done!')

    def insert2db(self):
        if 'res_nav' not in dir(self):
            self.calcNavRelated()
        if self.res_nav.empty:
            return None
        table = 'rc_mr_return_bch'
        self.insert2db_single(table, self.res_nav, ptf_code=self.ptf_codes, code_colname='PORTFOLIO_CODE')

def calcTimeSeries(bchIdx, date_list):
    res_navDF = pd.DataFrame()
    
    for t in date_list:
        print(t)
        bchIdx.calcNavRelated(endDate=t)    
        res_navDF = pd.concat([res_navDF, bchIdx.res_nav], sort=False)
#        res_single = bchIdx.calcSingleIndex(bchIdx.Treynor, t, risk_free_list)
#        res_navDF = pd.concat([res_navDF, res_single], sort=False)        
    
    return res_navDF


#%%
if __name__ == "__main__":
    data_path = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\Benchmark\\'
    file_nav = 'benchmark_database_fund.xlsx'

#    date_list = pd.read_excel(r'E:\RiskQuant\风险指标\2020年待补充交易日.xlsx')['TradingDate'].tolist()
#    date_list = [x.strftime('%Y-%m-%d') for x in date_list]
    date_list = ['2021-05-31', '2021-06-01', '2021-06-02', '2021-06-03', '2021-06-04']

    bchIdx = BenchmarkRiskReturn(data_path, file_nav)
    res_navDF = calcTimeSeries(bchIdx, date_list)

    # t = date_list[-1]
    # res_navDF.to_excel(r'E:\RiskQuant\风险指标\WeeklyBenchmark\%s_BchRiskReturn.xlsx'%t.replace('-', ''), index=False)