'''
@Description: 
@Author: Wangp
@Date: 2020-05-20 13:56:56
LastEditTime: 2020-08-25 14:08:20
LastEditors: Wangp
'''
import os
import numpy as np
import pandas as pd
from WindPy import w
w.start()

from .settings import config
from .db import OracleDB, column, sqls_config
from .db.db_utils import JYDB_Query
from .utils.log_utils import logger
from .utils.catch_error_msg import show_catch_message


def check_if_tradeday(t):
    '''检查给定日期是否是交易日'''
    if type(t) != str:
        t = t.strftime('%Y-%m-%d')
    latest_tradeday = retrieve_n_tradeday(t, 0)
    if t != latest_tradeday:
        logger.info('%s非交易日，无需运行程序。'%t)
        exit(0)
    else:
        return latest_tradeday

def retrieve_n_tradeday(t, n):
    '''
    取给定日期过去第n个交易日日期
    :param t: string/datetime/timestamp, 需查询的给定日期
    :param n: int, 日期偏移量, 仅支持向历史偏移
    :return: string, 过去第n个交易日日期
    '''
    if type(t) != str:
        t = t.strftime('%Y-%m-%d')
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    q = sqls_config['past_tradeday']['Sql']%t
    tradeday = db_risk.read_sql(q).sort_values(by=['c_date'])
    return tradeday.iloc[(-1)*(n+1)][0]


class Liquidity_Equity(JYDB_Query):
    def __init__(self, baseDate, part=0.05):
        '''
        流动性模型(权益)，取公开市场的日均成交量\n
        :param baseDate: string, 计算基期
        :param part: float, 市场参与度
        '''
        JYDB_Query.__init__(self)
        self.part = part
        self.baseDate = baseDate
        if type(baseDate) != str:
            self.baseDate = baseDate.strftime('%Y-%m-%d')

    def format_codes(self, codes):
        self.codes = [codes] if isinstance(codes, str) else codes
        codes_adj = [('0' + i) if 'HK' in i else i for i in self.codes]  # jy数据库的HK代码前多个0
        self.codes_num = [i[:i.find(".")] if '.' in i else i for i in codes_adj]
        self.codes_dict = dict(zip(self.codes_num, self.codes))

    def calc_liquidity(self, codes, delta=20):
        '''
        给定股票的成交量基础数据\n
        :param codes: string or list, 代码清单
        :param delta: int, 计算日均成交量时采用的天数
        :return: DataFrame
        '''
        if len(codes) == 0:
            return pd.DataFrame(columns=['code', 'liq_amt'])

        self.format_codes(codes)
        bg_date = retrieve_n_tradeday(self.baseDate, delta-1)

        q_vol = sqls_config['liq_amount_stock']['Sql']
        avg_vol = self.db.read_sql(q_vol.format(t0=bg_date, t1=self.baseDate))

        avg_vol = avg_vol[avg_vol['secucode'].isin(self.codes_num)].copy()
        avg_vol['code'] = avg_vol['secucode'].map(self.codes_dict)
        avg_vol['liq_amt'] = avg_vol['avg_vol'] * self.part

        return avg_vol.reindex(columns=['code', 'liq_amt'])


class Liquidity_Convertible(JYDB_Query):
    def __init__(self, baseDate, part=0.15):
        '''
        流动性模型(可转债),
        :param baseDate: string, 计算基期
        :param part: float, 市场参与度
        '''
        JYDB_Query.__init__(self)
        self.part = part
        self.baseDate = baseDate
        if type(baseDate) != str:
            self.baseDate = baseDate.strftime('%Y-%m-%d')

    def format_codes(self, codes):
        self.codes = [codes] if isinstance(codes, str) else codes
        self.codes_num = [i[:i.find(".")] if '.' in i else i for i in self.codes]
        self.codes_dict = dict(zip(self.codes_num, self.codes))

    def calc_liquidity(self, codes, delta=20):
        '''
        给定转债的成交量基础数据（张数）
        :param codes: string or list, 代码清单
        :param delta: int, 计算日均成交量时采用的天数
        :return: DataFrame
        '''
        if len(codes) == 0:
            return pd.DataFrame(columns=['code', 'liq_amt'])

        self.format_codes(codes)
        bg_date = retrieve_n_tradeday(self.baseDate, delta-1)

        avg_vol = self.sec_query('liq_amt_cbond', self.codes_num, bg_date, self.baseDate)
        avg_vol['code'] = avg_vol['SECUCODE'].map(self.codes_dict)
        avg_vol['liq_amt'] = avg_vol['AVG_VOL'] * self.part

        return avg_vol.reindex(columns=['code', 'liq_amt'])


class Liquidity_Repurchse:
    def __init__(self, baseDate):
        '''
        流动性模型(回购)\n
        :param baseDate: string, 计算基期
        '''
        self.baseDate = baseDate
        if type(baseDate) != str:
            self.baseDate = baseDate.strftime('%Y-%m-%d')

        self.data_path = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\Collaterals\\' + self.baseDate.replace('-', '') + '\\'

    def __dealRepoFile(self):
        # TODO: 检查是否还在用公盘的回购数据
        if not os.path.exists(self.data_path + '\\综合信息查询_交易所回购.xls'):
            return pd.DataFrame(columns=['回购日期', '基金名称', '回购代码', '委托方向', '数量', '净资金额', '返回金额', '平均利率(%)','法定购回日期', '实际购回日期', '回购天数', '计算方向', 'C_FUNDNAME'])
        
        repo_ex = pd.read_excel(self.data_path + '\\综合信息查询_交易所回购.xls',skipfooter=1)
        repo_ib = pd.read_excel(self.data_path + '\\综合信息查询_银行间回购.xls',skipfooter=1)

        cols = ['回购日期', '基金名称', '回购代码', '委托方向', '数量', '净资金额', '返回金额', '平均利率(%)', '法定购回日期', '实际购回日期']
        repo_all = repo_ex[cols].append(repo_ib[cols], sort=False)
        repo_all['回购日期'] = pd.to_datetime(repo_all['回购日期'])
        repo_all['法定购回日期'] = pd.to_datetime(repo_all['法定购回日期'])
        repo_all['实际购回日期'] = pd.to_datetime(repo_all['实际购回日期'])
        repo_all = repo_all[repo_all['回购日期'] <= pd.to_datetime(self.baseDate)].copy()

        repo_all['回购天数'] = [x.days for x in repo_all['法定购回日期'] - repo_all['回购日期']]
        repo_all['计算方向'] = repo_all['委托方向'].map({'融券回购': 0, '融资回购': -1})
        repo_all['ttm'] = [x.days for x in repo_all['法定购回日期'] - pd.to_datetime(self.baseDate)]
        repo_all['D_DATE'] = pd.to_datetime(self.baseDate)

        fundNames = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl')[['O32产品名称', '估值系统简称']]
        fundNames.columns = ['基金名称', 'C_FUNDNAME']

        repo_all = pd.merge(repo_all, fundNames, on='基金名称', how='left')
        
        self.repo_all = repo_all.copy()
    
    def calc_liquidity(self, n=1):
        res = self.repo_all[self.repo_all['ttm'] <= n].groupby(['C_FUNDNAME', '基金名称']).apply(lambda x: (x['数量']*x['计算方向']*100).sum()).rename('Repo_%d_balance'%n).reset_index()

        return res


class Liquidity_CD:
    def __init__(self, baseDate, part=0.15):
        '''
        流动性模型(CD)\n
        :param baseDate: string, 计算基期
        :param part: float, 市场参与度
        '''
        self.part = part
        self.baseDate = baseDate
        if type(baseDate) != str:
            self.baseDate = baseDate.strftime('%Y-%m-%d')
        self.file_name = 'coef_cd'

        self.coef_path = os.path.join(config['shared_drive_data']['liquidity_coef']['path'], 'CD')
        self._connectingRiskDB()

    def _connectingRiskDB(self):
        # logger.info('开始读取数据库：RISK')
        self.db_risk = OracleDB(config['data_base']['QUANT']['url'])

    def __findLatestCoefs(self):
        '''查找最近一个交易日的流动性因子系数'''
        date_list = os.listdir(self.coef_path)
        date_list = [x for x in date_list if '.' not in x]
        delta_list = np.array([(pd.to_datetime(x) - pd.to_datetime(self.baseDate)).days for x in date_list])
        t = np.array(date_list)[delta_list<=0][-1]

        self.t_path = os.path.join(self.coef_path, t)

    def getIssuerInfo(self, codes):
        '''获取CD的发行人信息'''
        wind_temp = w.wss(codes, "issuer_banktype, comp_name")
        data_wind = pd.DataFrame(wind_temp.Data, columns=wind_temp.Codes, index=wind_temp.Fields).T
        data_wind = data_wind.reset_index().rename(columns={'index': 'code', 'COMP_NAME': '发行人名称', 'ISSUER_BANKTYPE': '银行类型'})

        return data_wind

    def loadCoeffients(self, file_name='coef_cd'):
        self.__findLatestCoefs()
        self.coef_ir = pd.read_excel(os.path.join(self.t_path, '{file_name}.xlsx'.format(file_name=self.file_name))
                                     , engine='openpyxl')

    def calc_issuer_size(self):
        q = sqls_config['liq_issuer_size']['Sql'].format(base_date=self.baseDate, type_column='bond_type', type_i='同业存单')
        self.issuer_size = self.db_risk.read_sql(q)

    def dataFormatting(self, data):
        '''清洗CD的持仓，包括剩余期限、发行主体等信息'''
        self.calc_issuer_size()

        data['剩余期限'] = pd.cut(data['PTMYEAR'].fillna(0), bins=[-np.inf, 0.33, 0.6, np.inf], labels=['3M', '6M', '9M'])
        data['剩余期限'] = data[ '剩余期限'].fillna('3M')

        codes = data['code'].unique().tolist()
        data_wind = self.getIssuerInfo(codes)
        data = pd.merge(data, data_wind, on=['code', '发行人名称'], how='left')
        data = pd.merge(data, self.issuer_size.rename(columns={'issuer': '发行人名称'}), on=['发行人名称'], how='left')

        return data

    def calc_liquidity(self, data):
        '''按照CD发行主体的评级、剩余期限和发行银行类型，计算CD的流动性'''
        if data.shape[0] == 0:
            return pd.DataFrame(columns=['code', 'liq_amt'])

        self.loadCoeffients()
        data = self.dataFormatting(data).rename(columns={'剩余期限': 'ptm_type',
                                                         '银行类型': 'bond_type'})
        data_m = pd.merge(data, self.coef_ir.drop(columns=['residual_size']), on=['mir_cnbd', 'ptm_type', 'bond_type'], how='left')
        data_m['liq_amt'] = data_m['residual_size'] * data_m['turnover_daily'] * self.part * 1e6    # 还原为张，part为参与度

        res = data_m[['code', 'liq_amt', 'residual_size', 'turnover_daily']].drop_duplicates()
        
        return res

    def calc_highLiq_bond(self, data):
        '''高流动性债券：隐含评级在AA+以上'''
        if data.empty:
            return []

        sec_list = data.loc[data['mir_cnbd'].isin(['AAA+', 'AAA', 'AAA-', 'AA+']), 'code'].unique().tolist()
        return sec_list


class Liquidity_IRBond:
    def __init__(self, baseDate, part=0.05):
        '''
        流动性模型(利率债)\n
        :param baseDate: string, 计算基期
        :param part: float, 市场参与度
        '''
        self.part = part
        self.baseDate = baseDate
        if type(baseDate) != str:
            self.baseDate = baseDate.strftime('%Y-%m-%d')
        self.file_name = 'coef_ir'

        self.coef_path = os.path.join(config['shared_drive_data']['liquidity_coef']['path'], 'IRBond')
        self._connectingRiskDB()

    def _connectingRiskDB(self):
        # logger.info('开始读取数据库：RISK')
        self.db_risk = OracleDB(config['data_base']['QUANT']['url'])

    def __findLatestCoefs(self):
        '''查找最近一个交易日的流动性因子系数'''
        date_list = os.listdir(self.coef_path)
        date_list = [x for x in date_list if '.' not in x]
        delta_list = np.array([(pd.to_datetime(x) - pd.to_datetime(self.baseDate)).days for x in date_list])
        t = np.array(date_list)[delta_list<=0][-1]

        self.t_path = os.path.join(self.coef_path, t)

    def loadCoeffients(self, file_name='coef_ir'):
        self.__findLatestCoefs()
        self.coef_ir = pd.read_excel(os.path.join(self.t_path, '{file_name}.xlsx'.format(file_name=self.file_name))
                                   , engine='openpyxl')

    def loadActivityBond(self):
        '''读取活跃券信息'''
        self.data_act = pd.read_excel(os.path.join(self.t_path, '活跃券划分.xlsx'), engine='openpyxl')
        self.data_act = self.data_act.loc[self.data_act['activity'].isin(['活跃券', '次活跃券']), ['code', 'sec_name', 'issuer', 'liquidity', 'ptmyear', 'activity']].copy()

    def DefineActivity(self, data):
        '''识别持仓券的活跃度等级'''
        self.loadActivityBond()
        
        if '活跃度' in data.columns:            
            return data
        else:
            data1 = pd.merge(data, self.data_act.drop(columns=['sec_name', 'issuer', 'ptmyear']), on='code', how='left')
            data1['活跃度'] = data1['activity'].fillna('不活跃券')
    
            return data1.drop(columns=['activity'])

    def calc_bond_size(self):
        '''取债券余额数据'''
        q = sqls_config['liq_ir_bond_size']['Sql'].format(base_date=self.baseDate, type_column='bond_type_l1', type_i='利率债')
        self.bond_size = self.db_risk.read_sql(q)

    def dataFormatting(self, data):
        self.calc_bond_size()
        data1 = self.DefineActivity(data)

        dict_ir_issuer = {'中华人民共和国财政部': '国债', '中国进出口银行': '进出', '中国农业发展银行': '农发', '国家开发银行': '国开'}
        data1['品种'] = data1['发行人名称'].map(dict_ir_issuer)
        data1['剩余期限'] = pd.cut(data1['PTMYEAR'].fillna(0), bins=[-np.inf, 2, 4, 6, 8, 10, 30, 50], labels=['1年', '3年', '5年', '7年', '10年', '30年', '50年'])
        data1 = pd.merge(data1, self.bond_size.rename(columns={'sec_code': 'code'}), on='code', how='left')

        return data1

    def calc_liquidity(self, data):
        '''按照利率债发行主体、剩余期限和活跃度等级，计算利率债的流动性'''
        if data.shape[0] == 0:
            return pd.DataFrame(columns=['code', 'liq_amt'])
        self.loadCoeffients()
        data = self.dataFormatting(data).rename(columns={'品种': 'bond_type',
                                                         '剩余期限': 'ptm_type',
                                                         '活跃度': 'activity'})
        data_m = pd.merge(data, self.coef_ir.drop(columns=['residual_size']), on=['bond_type', 'ptm_type', 'activity'], how='left')
        data_m['liq_amt'] = data_m['residual_size'] * data_m['turnover_daily'] * self.part * 1e6    # 还原为张，part为参与度

        res = data_m[['code', 'liq_amt', 'residual_size', 'turnover_daily']].drop_duplicates()

        return res
        
    def calc_highLiq_bond(self, data):
        '''高流动性债券：利率债均为高流动性'''
        if data.empty:
            return []
        return data['code'].unique().tolist()


# TODO: 考虑是否把dataformatting放入对应资产中，而非Asset大类里
class Liquidity_CreditBond:
    def __init__(self, baseDate, part=0.3):
        '''
        流动性模型(信用债)\n
        :param baseDate: string, 计算基期
        :param part: float, 市场参与度
        '''
        self.part = part
        self.baseDate = baseDate
        self.file_name = 'coef_cr'
        if type(baseDate) != str:
            self.baseDate = baseDate.strftime('%Y-%m-%d')

        self.coef_list = ['评级', '剩余期限', '行业类别', '企业属性', '含权条款', '发行方式']
        self.coef_num = [x + '系数' for x in self.coef_list]

        self.coef_path = os.path.join(config['shared_drive_data']['liquidity_coef']['path'], 'CreditBond')
        self._connectingRiskDB()

    def _connectingRiskDB(self):
        # logger.info('开始读取数据库：RISK')
        self.db_risk = OracleDB(config['data_base']['QUANT']['url'])

    def __findLatestCoefs(self):
        '''查找最近一个交易日的流动性因子系数'''
        date_list = os.listdir(self.coef_path)
        date_list = [x for x in date_list if ('.' not in x) and ('Coef' not in x)]
        delta_list = np.array([(pd.to_datetime(x) - pd.to_datetime(self.baseDate)).days for x in date_list])
        t = np.array(date_list)[delta_list<=0][-1]

        self.t_path = os.path.join(self.coef_path, t)

    def __dealNaValues(self, data):
        '''处理缺失值'''
        data['mir_cnbd'] = data['mir_cnbd'].fillna('AA-')
        data['剩余期限'] = data['剩余期限'].fillna('1年(1.08)以下')
        data['行业类别'] = data['行业类别'].fillna('产业')
        data['企业属性'] = data['企业属性'].replace('其他', '民企').fillna('民企')
        data['发行方式'] = data['发行方式'].fillna('私募')
        data['含权条款'] = data['含权条款'].fillna('永续')
        data['偿还顺序'] = data['偿还顺序'].fillna('次级')

        return data

    def loadCoefficients(self, coef_list=[], file_name='coef_cr'):
        self.__findLatestCoefs()

        if len(coef_list) == 0:
            coef_list = self.coef_list
        coef_excel = pd.read_excel(os.path.join(self.t_path, '{file_name}.xlsx'.format(file_name=self.file_name))
                                   , sheet_name=None, engine='openpyxl')

        dict_coef = {}
        for coef_name in coef_list:
            dict_coef[coef_name] = coef_excel.get(coef_name)
        self.dict_coef = dict_coef
        self.coef_rating = self.dict_coef['评级']
        self.coef_ptm = self.dict_coef['剩余期限']
        self.coef_opt = self.dict_coef['含权条款']
        self.coef_issue = self.dict_coef['发行方式']

        if '行业类别' in coef_list:
            self.coef_ind = self.dict_coef['行业类别']
        if '偿还顺序' in coef_list:
            self.coef_pay = self.dict_coef['偿还顺序']
        if '企业属性' in coef_list:
            self.coef_soe = self.dict_coef['企业属性']
        if '债券分类' in coef_list:
            self.coef_type = self.dict_coef['债券分类']

        # print('Coefficients Loaded.')

    def calc_bond_size(self):
        '''取债券余额数据'''
        q = sqls_config['liq_bond_size']['Sql'].format(base_date=self.baseDate)
        self.bond_size = self.db_risk.read_sql(q)

    def calc_liquidity(self, data):
        '''计算信用债流动性：根据隐含评级、剩余期限、行业类别、偿还顺序、企业属性、含权条款和发行方式打折'''
        if data.shape[0] == 0:
            return pd.DataFrame(columns=['code', 'liq_amt'])

        self.loadCoefficients()
        data = self.__dealNaValues(data)
        self.calc_bond_size()
        data = pd.merge(data, self.bond_size.rename(columns={'sec_code': 'code'}), on='code', how='left').rename(
            columns={'mir_cnbd': '评级'})

        if '交易场所' in self.coef_rating.columns:
            data_m = pd.merge(data, self.coef_rating, on=['评级', '交易场所'], how='left')
        else:
            data_m = pd.merge(data, self.coef_rating, on=['评级'], how='left')

        data_m = pd.merge(data_m, self.coef_ptm, on=['剩余期限'], how='left')
        data_m = pd.merge(data_m, self.coef_opt, on=['含权条款'], how='left')
        data_m = pd.merge(data_m, self.coef_issue, on=['发行方式'], how='left')

        if '行业类别' in self.coef_list:
            data_m = pd.merge(data_m, self.coef_ind, on=['行业类别'], how='left')
        if '偿还顺序' in self.coef_list:
            data_m = pd.merge(data_m, self.coef_pay, on=['偿还顺序'], how='left')
        if '企业属性' in self.coef_list:
            data_m = pd.merge(data_m, self.coef_soe, on=['企业属性'], how='left')
        if '债券分类' in self.coef_list:
            data_m = pd.merge(data_m, self.coef_type, on=['债券分类'], how='left')

        data_m['liq_amt'] = data_m['residual_size'] * data_m[self.coef_num].product(axis=1, skipna=False) * 1e6 * self.part    # 还原为张，part为参与度

        res = data_m[['code', 'liq_amt']+self.coef_num].drop_duplicates()

        return res

    def _defineHighLiq(self):
        '''取全市场信用债前8%日均成交量阈值（单位：张）'''
        if 'dict_coef' not in dir(self):
            self.loadCoefficients()

        self.data_c = pd.read_excel(os.path.join(self.t_path, 'data_cr.xlsx'), engine='openpyxl')
        data = self.__dealNaValues(self.data_c)
        data_m = pd.merge(data, self.coef_rating.rename(columns={'评级': 'mir_cnbd'}), on=['mir_cnbd', '交易场所'], how='left')
        data_m = pd.merge(data_m.drop(columns=['剩余期限系数']), self.coef_ptm, on=['剩余期限'], how='left')
        data_m = pd.merge(data_m.drop(columns=['行业类别系数']), self.coef_ind, on=['行业类别'], how='left')
        # data_m = pd.merge(data_m.drop(columns=['偿还顺序系数']), self.coef_pay, on=['偿还顺序'], how='left')
        data_m = pd.merge(data_m.drop(columns=['企业属性系数']), self.coef_soe, on=['企业属性'], how='left')
        data_m = pd.merge(data_m.drop(columns=['含权条款系数']), self.coef_opt, on=['含权条款'], how='left')
        data_m = pd.merge(data_m.drop(columns=['发行方式系数']), self.coef_issue, on=['发行方式'], how='left')

        data_m['liq_amt'] = data_m['residual_size'] * data_m[self.coef_num].product(axis=1, skipna=False) * 1e6

        threshold = np.percentile(data_m['liq_amt'].dropna(), 92, interpolation='higher')  # 取前8%成交量阈值

        return threshold

    def calc_highLiq_bond(self, data):
        '''高流动性债券：经信用债流动性因子计算的市场成交量前8%'''
        if data.empty:
            return []
        threshold = self._defineHighLiq()
        print('%s -- Threshold(张): %.2f'%(self.baseDate, threshold))
        print('%s --Threshold(Discounted)(张):  %.2f'%(self.baseDate, threshold * self.part))
        df = pd.DataFrame([threshold], columns=['Volume_Threshold(8%)(张)'])
        df.to_excel(os.path.join(self.t_path, 'Volume_Threshold.xlsx'), index=False)

        if 'liq_amt' not in data.columns:
            data = self.calc_liquidity(data)
        sec_list = data.loc[data['liq_amt'] >= threshold * self.part, 'code'].unique().tolist()

        return sec_list


class Liquidity_SubordinateBond(Liquidity_CreditBond):
    def __init__(self, baseDate, part=0.15):
        Liquidity_CreditBond.__init__(self, baseDate, part)
        self.coef_list = ['剩余期限', '含权条款', '评级', '债券分类', '发行方式']
        self.coef_num = [x + '系数' for x in self.coef_list]
        self.file_name = 'coef_cr_sub'


class Liquidity_ABS(Liquidity_CreditBond):
    def __init__(self, baseDate, part=0.3):
        '''流动性模型(ABS), 沿用信用债框架\n'''
        Liquidity_CreditBond.__init__(self, baseDate, part)


class Liquidity_collaterals:
    def __init__(self, baseDate):
        '''
        流动性模型(质押券)\n
        :param baseDate: string, 计算基期
        '''
        self.baseDate = baseDate
        if type(baseDate) != str:
            self.baseDate = baseDate.strftime('%Y-%m-%d')

        self.file_path = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\Collaterals\%s\\'%(self.baseDate.replace('-', ''))
        self.coding_table = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl')

        self._connectingQuantDB()
        self._fundNameMapping()


    def _fundNameMapping(self):
        '''产品全称与估值表简称对应表'''
        q = sqls_config['portfolio_type']['Sql']
        self.ptf_info = self.db_risk.read_sql(q)
        self.fundname_to_code = self.ptf_info.set_index('c_fundname')['c_fundcode'].to_dict()

    def _connectingQuantDB(self):
        # logger.info('开始读取数据库：Quant')
        self.db_risk = OracleDB(config['data_base']['QUANT']['url'])

    def check_stock_holding(self):
        '''检查数据库中股票持仓表是否已更新'''
        q = sqls_config['stock_holding_check']['Sql']%self.baseDate
        check_data = self.db_risk.read_sql(q)
        if check_data.shape[0] > 0:
            logger.info('股票持仓数据已插入.')
        else:
            logger.error('股票持仓表dpe_sdp_portfolio无数据')
            title = 'Error'
            message = '股票持仓表dpe_sdp_portfolio无数据，是否继续运行程序？'
            show_catch_message(title, message)

    def deal_restricted_stock(self):
        '''获取流动受限股票清单'''
        self.check_stock_holding()
        q = sqls_config['dpe_restricted_stock']['Sql']%self.baseDate
        restr_stock = self.db_risk.read_sql(q)
        return restr_stock

    def deal_sec_code(self, x):
        mkt_dict = {'银行间': '.IB', '上交所A': '.SH', '深交所A': '.SZ'}
        x['code_suffix'] = x['交易市场'].map(mkt_dict)
        x['证券代码'] = x[['证券代码', 'code_suffix']].sum(axis=1)
        x = x.drop(columns=['code_suffix'])
        return x

    def check_repo_file(self, file_path):
        '''
        检查文件是否存在\n
        :param file_path: string, 文件路径
        :return: None, 若文件不存在则弹窗提醒
        '''
        if not os.path.exists(file_path):
            title = 'Error'
            message = '数据缺失：{}，是否继续运行程序？'.format(file_path)
            show_catch_message(title, message)

    def loadRepoFile(self):
        '''读取质押券相关基础数据'''
        repo_ib_path = self.file_path+'银行间回购质押券.xls'
        self.check_repo_file(repo_ib_path)
        self.repo_ib = pd.read_excel(repo_ib_path, skipfooter=1, dtype={'证券代码': str}).rename(columns={'回购日期': '质押日期', '购回日期': '质押结束日期'})
        self.repo_ib = self.repo_ib.drop_duplicates()
        self.repo_ib['证券代码'] = self.repo_ib['证券代码'].apply(lambda x: x+'.IB')

        repo_exchg_path = self.file_path+'综合信息查询_质押券.xls'
        self.check_repo_file(repo_exchg_path)
        self.repo_exchg = self.deal_sec_code(pd.read_excel(repo_exchg_path, skipfooter=1, dtype={'证券代码': str}))

        repo_exchg_s_path = self.file_path+'综合信息查询_标准券.xls'
        self.check_repo_file(repo_exchg_s_path)
        self.repo_exchg_s = pd.read_excel(repo_exchg_s_path, skipfooter=1, dtype={'证券代码': str})
        # self.restr_stock = pd.read_excel(self.file_path+'限售股名单.xlsx', engine='openpyxl')
        self.restr_stock = self.deal_restricted_stock()

    def calc_collateral(self):
        '''转换各资产的质押情况和受限情况，涵盖股票和债券等资产'''
        self.loadRepoFile()
        coding_table = self.coding_table[['O32产品名称', '估值系统简称']].rename(columns={'O32产品名称': '基金名称', '估值系统简称': 'C_FUNDNAME'})

        # 银行间质押券
        repo_ib = self.repo_ib[self.repo_ib['委托方向'] == '融资回购'].rename(columns={'质押数量': '受限数量'})
        repo_ib['质押日期'] = pd.to_datetime(repo_ib['质押日期'])
        repo_ib['质押结束日期'] = pd.to_datetime(repo_ib['质押结束日期'])
        repo_ib = repo_ib[repo_ib['质押结束日期'] > pd.to_datetime(self.baseDate)].copy()
        repo_ib = repo_ib.groupby(['基金名称', '证券代码', '证券名称'])['受限数量'].sum().reset_index()
        
        # 交易所质押券
        repo_exchg_1 = self.repo_exchg_s.groupby(['基金名称'])['融资数量'].sum().rename('已融资数量').reset_index()
        repo_exchg_2 = self.repo_exchg.groupby(['基金名称'])['已转标准券数量'].sum().rename('总标准券').reset_index()
        repo_exchg_ratio = pd.merge(repo_exchg_1, repo_exchg_2, on=['基金名称'], how='right')
        repo_exchg_ratio['质押比例'] = repo_exchg_ratio['已融资数量'] / repo_exchg_ratio['总标准券']

        repo_exchg = self.repo_exchg.copy()
        repo_exchg = pd.merge(repo_exchg, repo_exchg_ratio[['基金名称', '质押比例']], on=['基金名称'], how='left')
        repo_exchg['受限数量'] = repo_exchg['已质押数量'] * repo_exchg['质押比例'].fillna(0)

        # 限售股名单
        restr_stock = self.restr_stock.rename(columns={'stock_code': '证券代码', 'fund_name': 'C_FUNDNAME'})
        restr_stock['受限数量'] = restr_stock['amount'].copy()
        restr_stock = pd.merge(restr_stock, coding_table, on=['C_FUNDNAME'], how='left')

        cols_nd = ['基金名称', '证券代码', '受限数量']
        restr_secs = repo_ib[cols_nd].append(repo_exchg[cols_nd])
        restr_secs = pd.merge(restr_secs, coding_table, on=['基金名称'], how='left')
        restr_secs = restr_secs.append(restr_stock[['基金名称', 'C_FUNDNAME', '证券代码', '受限数量']], sort=False)
        restr_secs = restr_secs.groupby(['基金名称', 'C_FUNDNAME', '证券代码']).sum().reset_index()  # 合并同一组合同一只券的质押数量
        restr_secs['PORTFOLIO_CODE'] = restr_secs['C_FUNDNAME'].map(self.fundname_to_code)
        return restr_secs


class Liquidity_Asset:
    def __init__(self, t, holdings):
        '''
        流动性模型(总), 汇总各资产的流动性模型\n
        :param t: string, 计算基期
        :param holdings: DataFrame, 持仓证券表
        '''
        self.holdings = holdings.drop_duplicates().rename(columns={'RATE_LATESTMIR_CNBD': 'mir_cnbd'})
        self.baseDate = t

        self._equity = Liquidity_Equity(self.baseDate)
        self._convertible = Liquidity_Convertible(self.baseDate)
        self._repo = Liquidity_Repurchse(self.baseDate)
        self._irBond = Liquidity_IRBond(self.baseDate)
        self._cd = Liquidity_CD(self.baseDate)
        self._creditBond = Liquidity_CreditBond(self.baseDate)
        self._subordinate_bond = Liquidity_SubordinateBond(self.baseDate)
        self._abs = Liquidity_ABS(self.baseDate)
        self._collateral = Liquidity_collaterals(self.baseDate)
    
    def _defineLeverageHaircut(self):
        creditBond = {'AAA+':0.95,
                    'AAA':0.95,
                    'AAA-':0.9,
                    'AA+':0.85,
                    'AA':0.8,
                    'AA(2)':0.7,
                    'AA-':0.5,
                    'NR':0.95}
        convertBond = {'NR': 0}   # 可转债无隐含评级
        all_asset = {'CreditBond': creditBond, 'ConvertBond': convertBond}
        self._haircut = pd.DataFrame.from_dict(all_asset, orient='columns').stack().reset_index()
        self._haircut.columns = ['impRating', 'AssetType', 'Haircut']

    def dataFormatting(self):
        '''将各资产的各属性划分为流动性模型的标准分类'''
        data1_final = self.holdings.copy()

        dict_mkt = {'SH': '交易所', 'SZ': '交易所', 'IB': '银行间', 'HK': '交易所', 'BJ': '交易所'}
        data1_final['发行方式'] = data1_final['ISSUE_ISSUEMETHOD'].fillna('私募')
        data1_final['交易场所'] = [dict_mkt[x.split('.')[1]] for x in data1_final['code']]
        data1_final['mir_cnbd'] = data1_final['mir_cnbd'].replace('NR', np.nan)
        data1_final.loc[data1_final['WINDL2TYPE'].isin(['地方政府债', '政府支持机构债']), 'mir_cnbd'] = 'AAA+'
        data1_final['剩余期限'] = pd.cut(data1_final['PTMYEAR'], bins=[-np.inf, 1.08, 2.5, 4, 6, np.inf], labels=['1年(1.08)以下', '1-2.5年' ,'2.5-4年', '4-6年', '6年以上'])
        data1_final['企业属性'] = data1_final['NATURE1'].map({'地方国有企业': '国企', '中央国有企业': '央企', '国有企业': '国企', '民营企业': '民企', '集体企业': '民企', '公众企业': '上市企业', 
                                                            '中外合资企业': '外资', '外商独资企业': '外资', '外资企业': '外资'})
        data1_final['偿还顺序'] = data1_final['是否次级债'].map({'是': '次级', '否': '正常'})
        data1_final['行业类别'] = data1_final['MUNICIPALBOND'].map({'是': '城投', '否': '产业'})
        data1_final.loc[data1_final['WINDL1TYPE'] == '金融债', '行业类别'] = '金融'
        data1_final['含权条款'] = ['永续' if '延期' in x else '正常' for x in data1_final['CLAUSE'].fillna('*')]
        data1_final['债券分类'] = np.where(data1_final['WINDL2TYPE'].isin(['商业银行次级债券', '商业银行债']), '商业银行',
                                     np.where(data1_final['WINDL2TYPE'].isin(['保险公司债', '证券公司债', '证券公司短期融资券', '其他金融机构债']),
                                         '其他金融机构', '信用'))

        return data1_final

    def calc_liquidity(self):
        '''计算各资产的流动性情况'''
        data = self.dataFormatting()

        data_credit = data[(data['L_STOCKTYPE'] == '2') & ~data['WINDL2TYPE'].isin(
            ['同业存单', '政策银行债', '国债', '银保监会主管ABS', '证监会主管ABS', '交易商协会ABN', '可转债', '可交换债', '可分离转债存债'])
                           & data['偿还顺序'].isin(['正常'])].copy()
        data_sub = data[data['偿还顺序'].isin(['次级'])].copy()
        data_convbt = data[data['WINDL1TYPE'].isin(['可转债', '可交换债', '可分离转债存债'])].copy()
        data_ir = data[data['WINDL2TYPE'].isin(['政策银行债', '国债'])].copy()
        data_abs = data[data['WINDL1TYPE'].isin(['资产支持证券'])].copy()
        data_cd = data[data['WINDL1TYPE'].isin(['同业存单'])].copy()
        data_equity = data[data['L_STOCKTYPE'] == '1'].copy()

        self.res_credit = self._creditBond.calc_liquidity(data_credit)
        self.res_sub = self._subordinate_bond.calc_liquidity(data_sub)
        self.res_ir = self._irBond.calc_liquidity(data_ir)
        self.res_abs = self._abs.calc_liquidity(data_abs)
        self.res_cd = self._cd.calc_liquidity(data_cd)

        codes_equity = data_equity['code'].unique().tolist()
        self.res_equity = self._equity.calc_liquidity(codes_equity)
        codes_convbt = data_convbt['code'].unique().tolist()
        self.res_convbt = self._convertible.calc_liquidity(codes_convbt)

        # 合并各资产流动性结果，存入self.liq_asset
        liq_asset = pd.DataFrame()
        asset_list = [self.res_credit, self.res_sub, self.res_ir, self.res_cd, self.res_abs, self.res_equity, self.res_convbt]
        for asset in asset_list:
            liq_asset = pd.concat([liq_asset, asset[['code', 'liq_amt']]])

        # 如单券存在多组合持仓，则按持仓组合数量对测算的单日变现量进行分摊【针对**所有证券**】
        hld_port_cnt = data.groupby(['code'])['c_fundname'.upper()].nunique().rename('hld_port_cnt').reset_index()
        liq_asset = pd.merge(liq_asset, hld_port_cnt, on='code', how='left')
        liq_asset['liq_amt_orig'] = liq_asset['liq_amt'].copy()
        liq_asset['liq_amt'] = (liq_asset['liq_amt_orig'] / liq_asset['hld_port_cnt']).fillna(0)
        if not liq_asset.empty:
        # ABS和信用债、及利率不活跃券：鉴于交易不活跃，因此不按持仓组合数量分摊
            code_exclude = self.res_credit['code'].unique().tolist() + self.res_abs['code'].unique().tolist() + \
                           data_ir.loc[~data_ir['code'].isin(self._irBond.data_act['code']), 'code'].unique().tolist()
            liq_asset['liq_amt'] = np.where(liq_asset['code'].isin(code_exclude), liq_asset['liq_amt_orig'], liq_asset['liq_amt'])
        self.liq_asset = liq_asset.drop_duplicates().reindex(columns=['code', 'liq_amt'])

        logger.info('-- %s 各资产的流动性情况已计算完毕' % self.baseDate)
        return self.liq_asset

    def calc_collateral(self, ptf_codes=None):
        '''转换各资产的质押情况和受限情况，涵盖股票和债券等资产'''
        if 'restr_asset' not in dir(self):
            restr_asset = self._collateral.calc_collateral()
            if ptf_codes is not None:
                restr_asset = restr_asset[restr_asset['PORTFOLIO_CODE'].isin(ptf_codes)]

            self.restr_asset = restr_asset.drop(columns=['PORTFOLIO_CODE'])
            logger.info('-- %s 各资产的质押情况和受限情况已计算完毕' % self.baseDate)

        return self.restr_asset

    # def calc_liquidity_1d(self):
    #     '''计算各资产1日可变现情况'''
    #     if 'liq_asset' not in dir(self):
    #         self.calc_liquidity()
    #     if 'restr_asset' not in dir(self):
    #         self.restr_asset = self._collateral.calc_collateral()
    #
    #     data_m = pd.merge(self.holdings, self.liq_asset, on=['code'], how='left')
    #     data_m = pd.merge(data_m, self.restr_asset, left_on=['C_FUNDNAME', 'code'], right_on=['C_FUNDNAME', '证券代码'], how='left')
    #
    #     def _simple_stat(x):
    #         sec_amt = x['F_MOUNT'].sum()
    #         liq_amt = x['liq_amt'].min()
    #         amt_restr = x['受限数量'].min()
    #         return pd.DataFrame([sec_amt, liq_amt, amt_restr], index=['total_amt', 'liq_amt', 'restr_amt']).T
    #
    #     data_gp = data_m.groupby(['C_FUNDNAME', 'D_DATE', 'code']).apply(_simple_stat).reset_index()
    #     data_gp['liq_amt1'] = data_gp['liq_amt'] * 1
    #     data_gp['unrestr_amt'] = data_gp['total_amt'] - data_gp['restr_amt']
    #     # TODO: 未考虑解禁日，可能涉及新股限售解禁和大宗交易的限售解禁等情形
    #     data_gp['1日可变现_张'] = data_gp[['unrestr_amt', 'liq_amt1']].min(axis=1)
    #
    #     self.liq_1 = data_gp[['C_FUNDNAME', 'D_DATE', 'code', 'liq_amt1', '1日可变现_张']].drop_duplicates()
    #
    #     return self.liq_1
    #
    # def calc_liquidity_5d(self):
    #     '''计算各资产5日可变现情况'''
    #     if 'liq_asset' not in dir(self):
    #         self.calc_liquidity()
    #     if 'restr_asset' not in dir(self):
    #         self.restr_asset = self._collateral.calc_collateral()
    #
    #     data_m = pd.merge(self.holdings, self.liq_asset, on=['code'], how='left')
    #     data_m = pd.merge(data_m, self.restr_asset, left_on=['C_FUNDNAME', 'code'], right_on=['C_FUNDNAME', '证券代码'], how='left')
    #
    #     def _simple_stat(x):
    #         sec_amt = x['F_MOUNT'].sum()
    #         liq_amt = x['liq_amt'].min()
    #         amt_restr = x['受限数量'].min()
    #         return pd.DataFrame([sec_amt, liq_amt, amt_restr], index=['total_amt', 'liq_amt', 'restr_amt']).T
    #
    #     data_gp = data_m.groupby(['C_FUNDNAME', 'D_DATE', 'code']).apply(_simple_stat).reset_index()
    #     data_gp['liq_amt5'] = data_gp['liq_amt'] * 5
    #     data_gp['unrestr_amt'] = data_gp['total_amt'] - data_gp['restr_amt']
    #     # TODO: 未考虑解禁日，可能涉及新股限售解禁和大宗交易的限售解禁等情形
    #     data_gp['1日可变现_张'] = data_gp[['unrestr_amt', 'liq_amt5']].min(axis=1)
    #
    #     self.liq_5 = data_m[['C_FUNDNAME', 'D_DATE', 'code', 'liq_amt5', '5日可变现_张']].drop_duplicates()
    #
    #     return self.liq_5

    def calc_liquidity_Tday(self, t=1):
        '''计算各资产T日可变现情况'''
        if 'liq_asset' not in dir(self):
            self.calc_liquidity()
        if 'restr_asset' not in dir(self):
            self._collateral.calc_collateral()

        data_m = pd.merge(self.holdings, self.liq_asset, on=['code'], how='left')
        data_m = pd.merge(data_m, self.restr_asset, left_on=['C_FUNDNAME', 'code'], right_on=['C_FUNDNAME', '证券代码'], how='left')

        def _simple_stat(x):
            sec_amt = x['F_MOUNT'].sum()
            liq_amt = x['liq_amt'].min()
            amt_restr = x['受限数量'].fillna(0).min()
            return pd.DataFrame([sec_amt, liq_amt, amt_restr], index=['total_amt', 'liq_amt', 'restr_amt']).T

        index = ['C_FUNDNAME', 'D_DATE', 'code']
        data_gp = data_m.groupby(index).apply(_simple_stat).reset_index()
        if len(data_gp) == 1:
            for i in index:
                data_gp[i] = data_m[i].iloc[0]
        else:
            data_gp = data_gp.reset_index()
        data_gp['liq_amt%d'%t] = data_gp['liq_amt'] * t
        data_gp['unrestr_amt'] = data_gp['total_amt'] - data_gp['restr_amt']
        # TODO: 未考虑解禁日，可能涉及新股限售解禁和大宗交易的限售解禁等情形
        data_gp['%d日可变现_张'%t] = data_gp[['unrestr_amt', 'liq_amt%d'%t]].min(axis=1)

        self.liq_t = data_gp[['C_FUNDNAME', 'D_DATE', 'code', 'liq_amt%d'%t, '%d日可变现_张'%t]].drop_duplicates()
        
        return self.liq_t

    def calc_leverage_space(self):
        '''计算剩余可用杠杆空间，扣除现有已质押券'''
        if '受限数量' in self.holdings.columns:
            self.holdings = self.holdings.drop(columns=['受限数量'])

        self._defineLeverageHaircut()
        if 'restr_asset' not in dir(self):
            self._collateral.calc_collateral()

        data_bond = self.holdings[self.holdings['L_STOCKTYPE'] == '2'].copy()
        data_bond['AssetType'] = data_bond['WINDL1TYPE'].map({'可转债': 'ConvertBond', '可交换债': 'ConvertBond', '可分离转债存债': 'ConvertBond'}).fillna('CreditBond')
        data_bond['impRating'] = data_bond['mir_cnbd'].replace('评级_利率债', 'NR').replace('无评级', 'NR')

        data_bond = pd.merge(data_bond, self._haircut, on=['AssetType', 'impRating'], how='left')
        data_bond = pd.merge(data_bond, self.restr_asset, left_on=['C_FUNDNAME', 'D_DATE', 'code'], right_on=['C_FUNDNAME', 'D_DATE', '证券代码'], how='left')
        data_bond['受限数量'] = data_bond['受限数量'].fillna(0)
        data_bond['Haircut'] = data_bond['Haircut'].fillna(0)   # 评级过低则视为0
        data_bond['可加杠杆_张'] = (data_bond['F_MOUNT'] - data_bond['受限数量']) * data_bond['Haircut']

        return data_bond[['C_FUNDNAME', 'D_DATE', 'code', '受限数量', '可加杠杆_张']].drop_duplicates()

    def calcHighLiqBond(self):
        '''计算高流动性资产清单'''
        data = self.dataFormatting()

        data_credit = data[(data['L_STOCKTYPE'] == '2') & ~data['WINDL2TYPE'].isin(
            ['同业存单', '政策银行债', '国债', '银保监会主管ABS', '证监会主管ABS', '交易商协会ABN', '可转债', '可交换债', '可分离转债存债'])].copy()
        data_ir = data[data['WINDL2TYPE'].isin(['政策银行债', '国债'])].copy()
        data_abs = data[data['WINDL1TYPE'].isin(['资产支持证券'])].copy()
        data_cd = data[data['WINDL1TYPE'].isin(['同业存单'])].copy()

        self.sec_credit = self._creditBond.calc_highLiq_bond(data_credit)
        self.sec_ir = self._irBond.calc_highLiq_bond(data_ir)
        self.sec_abs = self._abs.calc_highLiq_bond(data_abs)
        self.sec_cd = self._cd.calc_highLiq_bond(data_cd)
        
        res_list = []
        sec_list = [self.sec_credit, self.sec_ir, self.sec_abs, self.sec_cd]
        for sec in sec_list:
            res_list += sec
        
        return res_list

#if __name__ == '__main__':
#    data = pd.read_excel(r'E:\RiskQuant\LiquidityRisk\BackTesting\颐利.xlsx', dtype={'L_STOCKTYPE': str}).drop_duplicates()
#    t = '2020-06-05'
#    
#    liq = Liquidity_Asset(t, data)
#    liq.calc_liquidity()
#    liq_1 = liq.calc_liquidity_1d()
#    liq_5 = liq.calc_liquidity_5d()
#    lev_spc = liq.calc_leverage_space()
#    
#    data = pd.merge(data, liq_1, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')
#    data = pd.merge(data, liq_5, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')
#    data['1日可变现'] = data['1日可变现_张'] * data['F_PRICE'].astype(float)
#    data['5日可变现'] = data['5日可变现_张'] * data['F_PRICE'].astype(float)
#    grouped = data.groupby(['C_FUNDNAME', 'D_DATE'])
#    res_1 = grouped['1日可变现'].sum().reset_index()
#    res_5 = grouped['5日可变现'].sum().reset_index()
#    
#    data = pd.merge(data, lev_spc, on=['C_FUNDNAME', 'D_DATE', 'code'], how='left')
#    data['可加杠杆_市值'] = data['可加杠杆_张'] * data['F_PRICE'].astype(float)
#    res = data.groupby(['C_FUNDNAME', 'D_DATE'])['可加杠杆_市值'].sum().reset_index()
