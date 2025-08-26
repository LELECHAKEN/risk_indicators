'''
@Description: Basic Indicators
@Author: Wangp
@Date: 2020-03-12 10:50:00
LastEditTime: 2021-06-15 15:47:35
LastEditors: Wangp
'''
import pandas as pd

from .db.db_utils import JYDB_Query
from .utils_ri.retrieveAttribution import *
from .utils_ri.RiskIndicators import RiskIndicators

class BasicIndicators(RiskIndicators):
    def __init__(self, t, save_path, ptf_codes=None):
        self.basedate = t
        self.save_path = save_path
        self._format_ptf_codes(ptf_codes)
        
        self._loadFile()
        self._loadHoldings(self.save_path)
        self._dealKCSector()
        self.__loadTypeFile()


    
    def __loadTypeFile(self):
        '''加载各类码表'''
        self._fundName = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl').rename(columns={'产品名称': 'C_FULLNAME', '估值系统简称': 'C_FUNDNAME', 'O32产品名称': 'C_FUNDNAME_o32'})
        self._fundType = pd.read_excel(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\PortfolioType\产品管理类型.xlsx', engine='openpyxl').rename(columns={'基金简称': 'C_FUNDNAME_o32'})
        self._bchPeriod = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\存量专户产品业绩比较基准.xlsx', engine='openpyxl')

    def _dealKCSector(self, data=''):
        '''识别科创板'''
        if type(data) == str:
            data1 = self.stock_holdings.copy()
        else:
            data1 = data.copy()
        data1['是否科创板'] = [x[:3] == '688' for x in data1['code']]
        data1.loc[data1['是否科创板'], '上市板块'] = '科创板'
        data1.loc[data1['是否科创板'], 'LISTEDSECTOR'] = 8
        data1 = data1.drop(columns=['是否科创板'])

        if type(data) == str:
            self.stock_holdings = data1.copy()
        else:
            return data1

    def getFundType(self):
        '''匹配产品管理类型'''
        data_name = self._fundName[['C_FUNDNAME', 'C_FUNDNAME_o32']].copy()
        data_type = self._fundType[['C_FUNDNAME_o32', '管理类型', '一级分类', '二级分类', '三级分类', '特殊分类', '持有人结构', '基金经理']].copy()
        res = pd.merge(data_name, data_type, on=['C_FUNDNAME_o32'], how='outer').drop(columns=['C_FUNDNAME_o32']).dropna(subset=['C_FUNDNAME'])

        return res

    # 按照不同分类方式获取不同分布数据
    def getDistribution(self, cols_name, col_values='F_ASSET', data_given=''):
        '''获取各分类的持仓及占比情况'''
        if type(data_given) == str:
            data = self.bond_holdings.copy()
            data = data[~data['WINDL1TYPE'].isin(['资产支持证券', '可转债', '可交换债'])].copy()
        else:
            data = data_given.copy()

        res1 = pd.pivot_table(data, values=col_values, index=['D_DATE', 'C_FUNDNAME'], columns=cols_name, aggfunc='sum')
        res2 = res1.sum(axis=1)
        res1 = res1.reset_index(level=['D_DATE', 'C_FUNDNAME'])
        res2 = res2.reset_index(level=['D_DATE', 'C_FUNDNAME']).rename(columns={0: 'Total'})

        # 各类债券占债券资产的比例
        res = pd.merge(res1, res2, on=['D_DATE', 'C_FUNDNAME'], how='outer').set_index(['D_DATE', 'C_FUNDNAME'])
        for col in res.columns:
            res[col] = res[col] / res['Total']

        res = res.reset_index(level=['D_DATE', 'C_FUNDNAME']).drop(columns=['Total'])
        
        return res

    def __getMarketIndices(self, baseDate):
        '''基期全市场股票的市值情况'''
        baseDate = pd.to_datetime(baseDate).strftime('%Y-%m-%d')
        dq = JYDB_Query()
        data_market = dq.sec_query('stock_market_mv', [1,2], baseDate)    # 默认只取主板和中小板
        data_market = self._dealCode(data_market)
        data_market = data_market.rename(columns={'TRADINGDAY': 'Date', 'PE': 'PE_TTM'})
        # data_market = pd.merge(data_market, self._SectorMap, on=['LISTEDSECTOR'], how='left')

        return data_market

    def _splitCapType(self, data_market):
        '''
        全市场股票区分大中小盘股\n
        :param data_market: DataFrame, 全市场股票市值数据
        :return: (DataFrame, float, float)含市值划分标签的全市场股票数据; 大盘股阈值; 小盘股阈值
        '''
        data_market = data_market.sort_values(by=['TOTALMV'], ascending=False)
        data_market['CumRatio'] = data_market['TOTALMV'].cumsum() / data_market['TOTALMV'].sum()
        threshold1 = data_market.loc[data_market['CumRatio'] <= 0.7, 'TOTALMV'].iloc[-1]
        threshold2 = data_market.loc[data_market['CumRatio'] <= 0.9, 'TOTALMV'].iloc[-1]
        data_market['CapType'] = pd.cut(data_market['TOTALMV'], bins=[0, threshold2, threshold1, np.inf], labels=['小盘股', '中盘股', '大盘股'])
        
        return data_market, threshold1, threshold2

    def _splitValueGrowth1(self, data_market, ratio=0.5, cols_name='cmpIndice'):
        '''
        全市场股票区分价值/成长(晨星标准, pe和pb各占50%，按大中小盘分组划分)\n
        :param data_market: DataFrame, 全市场股票数据
        :param ratio: float, pe_ttm在划分标准中的占比
        :param cols_name: string, 划分标准的取值列名
        :return: DataFrame, 各分组下区分价值/成长的阈值
        '''
        def splitValueGrowth0(data_all):
            data_all = data_all.sort_values(ascending=False)
            threshold = data_all.iloc[int(0.5*data_all.shape[0])]

            return threshold
            
        data_market[cols_name] = data_market['PE_TTM'] * ratio + data_market['PB'] * (1-ratio)
        res = data_market.groupby(['CapType'])[cols_name].apply(splitValueGrowth0).reset_index()
        res.columns = ['CapType', 'VG_threshold']

        self.data_market = data_market.copy()
        
        return res

    def _dealCapType(self, data=''):
        '''
        区分持仓股票的市值类型\n
        :param data: DataFrame, 全量股票持仓表
        :return: DataFrame, 划分大中小盘后的股票持仓表
        '''
        if type(data) == str:
            data = self.stock_holdings.copy()        

        data_market = self.__getMarketIndices(self.basedate)
        data_market, threshold1, threshold2 = self._splitCapType(data_market)
        self.data_market = data_market.copy()
        
        data['CapType'] = pd.cut(data['TOTALMV'], bins=[0, threshold2, threshold1, np.inf], labels=['小盘股', '中盘股', '大盘股']).astype(str)
        data['CapType'] = data['CapType'].replace('nan', '新股')
        data.loc[data['上市板块'].isin(['创业板', '科创板']), 'CapType'] = data.loc[data['上市板块'].isin(['创业板', '科创板']), '上市板块']

        return data

    def _dealValueGrowth(self, data='', ratio=0.5):
        '''
        区分持仓股票的价值VS成长类型\n
        :param data: DataFrame, 全量股票持仓表
        :param ratio: float, 划分价值成长时pe_ttm指标的权重
        :return: DataFrame, 划分价值/成长标签后的股票持仓表
        '''
        if type(data) == str:
            data = self.stock_holdings.copy()
        
        if 'PE_TTM' not in data.columns:
            print('持仓信息中无PE_TTM数据！')
            return None
        elif 'PB' not in data.columns:
            print('持仓信息中无PB数据！')
            return None
        else:
            data['cmpIndice'] = data['PE_TTM'] * ratio + data['PB'] * (1-ratio)

            VG_threshold = self._splitValueGrowth1(self.data_market, ratio, cols_name='cmpIndice')
            data = pd.merge(data, VG_threshold, on=['CapType'], how='left')
            data.loc[data['cmpIndice'] <= data['VG_threshold'], 'VGType'] = '成长'
            data.loc[data['cmpIndice'] > data['VG_threshold'], 'VGType'] = '价值'

            return data

    def getAttribution(self):
        '''读取各投资组合的大类资产贡献结果，公募取年初以来，专户取当前考核期'''
        baseDate = self.holdings['D_DATE'].max()

        # 处理专户考核期
        dateDF1 = self._bchPeriod[['产品名称', '考核期开始日', '考核期结束日']].dropna(subset=['考核期开始日'])
        dateDF1['截止日期'] = baseDate
        dateDF1['截止日期'] = dateDF1[['截止日期', '考核期结束日']].min(axis=1)
        # dateDF1['产品类型'] = '专户'
        dateDF1['考核期开始日'] = [x.strftime('%Y-%m-%d') for x in dateDF1['考核期开始日']]
        dateDF1['截止日期'] = [x.strftime('%Y-%m-%d') for x in dateDF1['截止日期']]
        dateDF1 = dateDF1.drop(columns=['考核期结束日'])
        prods1 = dateDF1['产品名称'].drop_duplicates().tolist()

        # 处理公募区间
        dateDF2 = self.holdings[['C_FULLNAME']].drop_duplicates().rename(columns={'C_FULLNAME': '产品名称'})
        dateDF2['startDate'] = baseDate.strftime('%Y-%m-%d')[:4] + '-01-01'     # 从年初开始
        dateDF2['endDate'] = baseDate.strftime('%Y-%m-%d')
        prods2 = dateDF2['产品名称'].drop_duplicates().tolist()

        data1 = AssetAllocation_all(self.db_risk, prods1, dateDF1, '3')                # 3为专户，1为公募
        data2 = AssetAllocation_all(self.db_risk, prods2, dateDF2, '1')
        data = data1.append(data2, sort=False)
        self._attri_data = data.copy()

        cols = ['R_FUND', 'ATTRI_STOCK', 'ATTRI_BOND', 'ATTRI_CONVERTBOND', 'ATTRI_ABS', 'ATTRI_FUNDINVEST', 'ATTRI_DERIVATIVE', 'ATTRI_CASH', 'ATTRI_REPOBUY', 'ATTRI_OTHER']
        res = data.groupby(['C_FULLNAME'])[cols].apply(AttributionCumulation).reset_index()
        res['D_DATE'] = baseDate

        name_map = self.val[['C_FULLNAME', 'C_FUNDNAME']].drop_duplicates()
        res = pd.merge(res, name_map, on='C_FULLNAME', how='left')

        return res

    def getBondType(self, cols_name):
        cols = ['D_DATE', 'C_FUNDNAME', 'CD', '产业债', '利率债', '城投债', '金融债']
        bond_holdings = self.bond_holdings[~self.bond_holdings['WINDL1TYPE'].isin(['资产支持证券', '可转债', '可交换债'])].copy()

        if bond_holdings.empty:
            return pd.DataFrame(columns=cols)

        bond_type = self.getDistribution(cols_name=cols_name, data_given=bond_holdings)
        return bond_type.reindex(columns=cols)

    def getStockType(self, cols_name):
        cols = ['D_DATE', 'C_FUNDNAME', '中盘股价值', '中盘股成长', '创业板', '大盘股价值', '大盘股成长', '小盘股价值',
                '小盘股成长', '新股', '科创板']
        if self.stock_holdings.empty:
            return pd.DataFrame(columns=cols)

        self.stock_holdings = self._dealCapType()
        self.stock_holdings = self._dealValueGrowth()
        self.stock_holdings[cols_name] = self.stock_holdings['CapType'] + self.stock_holdings['VGType'].fillna('')
        stock_type = self.getDistribution(cols_name=cols_name, col_values='F_ASSET', data_given=self.stock_holdings)

        return stock_type.reindex(columns=cols)

    def CalcAllIndices(self):
        data_type = self.getFundType()
        data_asset = self.getBasicData()
        data_asset['现金'] = data_asset[['Deposit', '买入返售']].sum(axis=1)

        bond_asset = self.getBondType(cols_name='城投or产业')  # 剔除ABS、可转债
        stock_type = self.getStockType(cols_name='股票风格')

        funds = self.val[['C_FUNDNAME']].drop_duplicates()
        res = pd.merge(funds, data_type, on=['C_FUNDNAME'], how='left')
        res = pd.merge(res, data_asset, on=['C_FUNDNAME'], how='left')
        res = pd.merge(res, bond_asset, on=['D_DATE', 'C_FUNDNAME'], how='left')
        res = pd.merge(res, stock_type, on=['D_DATE', 'C_FUNDNAME'], how='left')

        cols = ['CD', '产业债', '利率债', '城投债', '金融债']   # 需要还原成占总资产比的列
        for col in cols:
            res[col] = res[col] * res['债券']
        res = res.sort_values(by=['基金类型', '一级分类', '二级分类', '三级分类', '管理类型'])

        return res