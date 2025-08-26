'''
Description: 风险预警，市场风险、信用风险、流动性风险等
Author: Wangp
Date: 2021-06-22 13:31:01
LastEditTime: 2021-06-22 15:58:47
LastEditors: Wangp
'''
import os
import datetime
import numpy as np
import pandas as pd
from WindPy import w

from .settings import config
from .db import OracleDB, column, sqls_config
from .utils_ri.RiskIndicators import RiskIndicators
from . import demo_code


def dealCode(data):
    data0 = data.copy()
    data0['mkt'] = data0['secumarket'].map({89: '.IB', 83: '.SH', 90: '.SZ'})
    data0['code'] = data0['secucode'] + data0['mkt']

    return data0.drop(columns=['secucode', 'secumarket', 'mkt'])

class PortfolioWarning(RiskIndicators):
    def __init__(self, t, save_path, ptf_codes=None):
        '''
        初始化
        :param t: string format, yyyy-mm-dd, 计算基期
        :param save_path: string format, DPE相关文件的存储路径
        '''
        self.basedate = t
        self.save_path = save_path
        self._format_ptf_codes(ptf_codes)
    
        self._loadFile()
        self._loadHoldings(self.save_path)
        self._loadTypeFile()

    def _loadTypeFile(self):
        '''
        加载各类码表，包括CodingTable, 产品管理类型, 存量专户产品业绩比较基准, 回撤阈值
        :return:无
        '''
        self._fundName = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl').rename(columns={'产品名称': 'C_FULLNAME', '估值系统简称': 'C_FUNDNAME', 'O32产品名称': 'C_FUNDNAME_o32'})
        self._fundType = pd.read_excel(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\PortfolioType\产品管理类型.xlsx', engine='openpyxl').rename(columns={'基金简称': 'C_FUNDNAME_o32', '基金全称': 'C_FULLNAME'})
        self._bchPeriod = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\存量专户产品业绩比较基准.xlsx', engine='openpyxl')
        self._dd_map = pd.read_excel(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\PortfolioType\回撤阈值.xlsx', engine='openpyxl').rename(columns={'基金名称': 'C_FUNDNAME', '回撤阈值': 'dd_threshold'})

    def connect_jydb(self):
        self.db_jy = OracleDB(config['data_base']['JY']['url'])

    def getSoldOutSecs(self):
        '''
        从数据库读取当日处置意见为择机卖出的所有债券
        :return: 无
        '''
        sql_sold = sqls_config['sold_out_secs']['Sql']    
        self.soldOutsecs = self.db_risk.read_sql(sql_sold)
    
    def getLatestPar(self, code_list):
        '''
        从wind API读取债券最新面值
        :param code_list: list,债券代码列表
        :return: DataFrame, 债券最新面值
        '''
        w.start()
        wind_temp = w.wss(code_list, "latestpar","tradeDate=%s"%self.basedate)
        data_wind1 = pd.DataFrame(wind_temp.Data, columns=wind_temp.Codes, index=wind_temp.Fields).T
        data_wind1 = data_wind1.reset_index().rename(columns={'index': 'code'})
        return data_wind1

    def get_bond_ptm_option(self, code_list):
        '''
        从聚源获取债券的行权剩余期限
        :param code_list: list，债券代码表
        :return:
        '''
        self.connect_jydb()
        sec_data = pd.DataFrame()
        loop_num = int(np.ceil(len(code_list)/1000))
        for i in range(loop_num):
            secs_temp = code_list[i * 1000: min((i + 1) * 1000, len(code_list))]
            secs_temp_str = ','.join('\'' + str(x.split('.')[0]) + '\'' for x in secs_temp)
            q = sqls_config['bond_ptm_option']['Sql']% (secs_temp_str, self.basedate)
            sec_data_temp = self.db_jy.read_sql(q)
            sec_data = pd.concat([sec_data, sec_data_temp])

        sec_data = dealCode(sec_data)
        sec_data['trueremainmaturity'] = sec_data['trueremainmaturity'].astype(float)
        sec_data = sec_data[sec_data['code'].isin(code_list)].copy()

        return sec_data

    def drawdown_warning(self):
        '''
        执行回撤预警
        :return: DataFrame, 触警的所有组合相关信息
        '''
        dd_df = self.db_risk.read_sql(sqls_config["rc_alert_dd"]["sql"].format(t=self.basedate))
        dd_df_sma = self.db_risk.read_sql(sqls_config["rc_alert_dd_sma"]["sql"].format(t=self.basedate))
        dd_df = pd.concat([dd_df, dd_df_sma])
        self.dd_warning = dd_df.copy()
        self.insert2db_single("rc_alert_dd", dd_df, t=self.basedate)

    def credit_warning(self):
        '''
        执行信用风险预警，依据信评给的“择机卖出"意见
        :return: DataFrame, 触警的所有组合相关信息
        '''
        self.getSoldOutSecs()
        data_soldout = self.soldOutsecs.reindex(columns=['债项代码', '债项名称', '公司名称', '处置建议', '评级级别', '主体评级', '评级日期', '评级状态'])
        data_soldout.columns = ['code', 'sec_name', 'issuer_name', 'disposal', 'innerRating_bond', 'innerRating_issuer', 'rating_date', 'rating_status']
        data_wind = self.getLatestPar(data_soldout['code'].unique().tolist())
        data_type = self._fundType.reindex(columns=['C_FULLNAME', '产品类型', '管理类型', '一级分类', '基金经理'])
        data_type.columns = ['C_FULLNAME', 'FUNDTYPE', 'TYPE', 'TYPE_L1', 'MANAGER']
        data = self.bond_holdings.reindex(columns=['C_FULLNAME', 'C_FUNDNAME', 'D_DATE', 'code', 'C_SUBNAME_BSH', 'F_MOUNT', 'F_ASSETRATIO', 'RATE_LATESTMIR_CNBD', 'YIELD_CNBD', 'PTMYEAR'])

        # todo: 债券持仓增加t+n产品

        res = pd.merge(data_soldout, data, on=['code'], how='left')
        res = pd.merge(data_type, res, on='C_FULLNAME', how='right')
        res = pd.merge(res, data_wind, on='code', how='left')
        # TODO: 债券的剩余期限改为行权
        code_list = res['code'].unique().tolist()
        if len(code_list) == 0:
            self.cr_warning = pd.DataFrame()
            return
        bond_ptm = self.get_bond_ptm_option(code_list)
        res = pd.merge(res, bond_ptm[['code', 'trueremainmaturity']], on=['code'], how='left')
        res = res.dropna(subset=['C_FULLNAME'])
        res['asset_par'] = (res['F_MOUNT'] * res['LATESTPAR']).astype(int)
        res.loc[res['trueremainmaturity'].notna(), 'PTMYEAR'] = res.loc[res['trueremainmaturity'].notna(), 'trueremainmaturity']
        res = res.drop(columns=['trueremainmaturity'])

        self.cr_warning = res.copy()
        self.cr_warning.loc[self.cr_warning['TYPE'] == '一类', 'TYPE_L1'] = '多策略户'

        self.insert2db_single("RC_ALERT_CR", self.cr_warning, t=self.basedate)

    def liquidity_warning(self):
        '''
        执行流动性预警。针对一个月内临近到期的专户是否持仓低流动性债券
        :return: DataFrame, 触警的所有组合相关信息
        '''
        # todo: 债券持仓增加t+n产品
        sql_liq = sqls_config['liq_warning']['Sql']%self.basedate
        liq_warning = self.db_risk.read_sql(sql_liq)
        liq_warning['d_date'] = pd.to_datetime(liq_warning['d_date'])
        liq_warning['maturity'] = liq_warning['maturity'].astype(int)
        liq_warning['lowliq_ratio'] = liq_warning['lowliq_ratio'] / 100
        liq_warning.columns = [x.upper() if x.islower() else x for x in liq_warning.columns]
        self.liq_warning = liq_warning.copy()

        self.insert2db_single("RC_ALERT_LR", self.liq_warning, t=self.basedate)

    def calc_all_warnings(self):
        '''
        整合所有预警函数
        :return: 无
        '''
        self.drawdown_warning()
        self.credit_warning()
        self.liquidity_warning()

    def insert2db(self):
        '''
        将所有预警相关底稿存入数据库
        :return: 无
        '''
        tables = ['RC_ALERT_DD', 'RC_ALERT_CR', 'RC_ALERT_LR']
        res_list = [self.dd_warning, self.cr_warning, self.liq_warning]
        for table, data in zip(tables, res_list):
            self.insert2db_single(table, data, t=self.basedate)

    def saveAll(self, save_path):
        '''
        将所有预警底稿存入Excel
        :param save_path: string format, 预警结果文件的存储路径
        :return:
        '''
        writer = pd.ExcelWriter(os.path.join(save_path, '%s_RiskAlert.xlsx'%self.basedate.replace('-', '')))
        self.dd_warning.to_excel(writer, sheet_name='RC_ALERT_DD', index=False)
        self.cr_warning.to_excel(writer, sheet_name='RC_ALERT_CR', index=False)
        self.liq_warning.to_excel(writer, sheet_name='RC_ALERT_LR', index=False)
        writer.save()