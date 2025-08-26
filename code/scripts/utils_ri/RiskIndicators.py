'''
Description: 
Author: Wangp
Date: 2020-03-10 13:36:59
LastEditTime: 2021-06-17 16:48:11
LastEditors: Wangp
'''

import os
import time
import warnings
import datetime
import numpy as np
import pandas as pd
from WindPy import w
import cx_Oracle as cx
from sqlalchemy import exc
from tkinter import messagebox
w.start()

from scripts.settings import config
from scripts.utils.catch_error_msg import show_catch_message
from scripts.utils.log_utils import logger
from scripts.db import OracleDB, column, sqls_config, and_, column
from scripts.db.db_utils import JYDB_Query
from scripts.db.util import DoesNotExist
from scripts.settings import config, DIR_OF_MAIN_PROG

import sys
sys.path.append(config['pkgs_custom']['path_1'])
sys.path.append(config['pkgs_custom']['path_2'])

class RiskIndicators:
    def __init__(self, t, ptf_codes=None):
        self.basedate = t
        self._format_ptf_codes(ptf_codes)

        self.holdings = pd.DataFrame()
        self.bond_holdings = pd.DataFrame()
        self._InterestBond = ['政策银行债', '国债', '央行票据', '地方政府债', '政府支持机构债']
        self._CBond = ['可转债', '可交换债']
        self._SectorMap = pd.DataFrame([np.arange(1, 8), ['主板', '中小板', '三板', '其他', '大宗交易系统', '创业板', '科创板']], index=['LISTEDSECTOR', '上市板块']).T
        self.path_masterData = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\MasterData\\'
        self.path_repo = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\Collaterals\\'

        self._connectingRiskDB()
        self.check_if_tradeday()
        self._fundNameMapping()
        self._loadFile()
        self.check_nav_data()
        self._loadSpreadAD()
        self._loadIndustryID()
        self._loadMasterData()
        self._formatCurveMap()
        self._loadTableColumns()

    def _format_ptf_codes(self, ptf_codes):
        if ptf_codes is not None and isinstance(ptf_codes, str):
            self.ptf_codes = [ptf_codes]
        self.ptf_codes = ptf_codes

    def _connectingRiskDB(self):
        # logger.info('开始读取数据库：RISK')
        self.db_risk = OracleDB(config['data_base']['QUANT']['url'])

    def get_latest_tradeday(self, t):
        '''取最近的一个交易日日期'''
        q = sqls_config['check_latest_day']['sql'] % t
        if 'db_risk' not in dir(self):
            self._connectingRiskDB()
        latest_tradeday = self.db_risk.read_sql(q)['c_date'].iloc[0]
        return latest_tradeday

    def get_offset_tradeday(self, t, n):
        '''
        取给定日期过去或未来的第n个交易日
        :param t: string/datetime/timestamp, 需查询的给定日期
        :param n: int, 日期偏移量, 可正可负
        :return: string, 过去或未来的第n个交易日日期
        '''
        if type(t) != str:
            t = t.strftime('%Y-%m-%d')
        sql_name = 'offset_history_date' if n <= 0 else 'offset_future_date'
        sql_raw = sqls_config[sql_name]['Sql']
        if 'db_risk' not in dir(self):
            self._connectingRiskDB()
        tds = self.db_risk.read_sql(sql_raw.format(t=t, n=abs(n)))
        return tds['c_date'].iloc[-1]

    def get_period_tradedays(self, t0: str, t1: str, df_or_list='list'):
        '''获取时间区间内的交易日期，列名为:c_date'''
        sql_raw = sqls_config['period_trade_days']['Sql']
        if 'db_risk' not in dir(self):
            self._connectingRiskDB()
        td_df = self.db_risk.read_sql(sql_raw.format(t0=t0, t1=t1))
        return td_df if df_or_list == 'df' else td_df['c_date'].to_list()

    def check_if_tradeday(self):
        '''检查t日是否为交易日，非交易日则不运行模型'''
        latest_tradeday = self.get_latest_tradeday(self.basedate)
        trade_status = latest_tradeday == self.basedate
        if not trade_status:
            logger.info('%s非交易日，无需运行模型。' % self.basedate)
            exit(0)

    def _loadFile(self, t=''):
        '''
        取估值表数据\n
        :param t: string, yyyy-mm-dd, 默认取基期
        :return: DataFrame, 原始估值表
        '''
        if t == '':
            givenDate = self.basedate
        else:
            givenDate = t

        if 'db_risk' not in dir(self):
            self._connectingRiskDB()
        if 'name_dict' not in dir(self):
            self._fundNameMapping()

        drop_cols = ['insert_time', 'srcsys', 'src_insert_time', 'src_update_time']
        sql_val = sqls_config['dws_ptfpos_detail']['Sql']%givenDate
        data = self.db_risk.read_sql(sql_val).drop(columns=drop_cols)

        # 取非t日产品的预估数据
        if self.ptf_codes is None:
            sql_pre = sqls_config['dws_ptfpos_detail_pre']['Sql'].format(t=givenDate)
            data_pre = self.db_risk.read_sql(sql_pre).drop(columns=['natural_date'] + drop_cols)
            data = pd.concat([data, data_pre], ignore_index=True)

        data.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                        else x.upper() for x in data.columns]
        # logger.info('估值表读取成功')

        data['D_DATE'] = pd.to_datetime(data['C_DATE'])  # 不规定format会出现乱码，如2019-7-12变为2019-12-7
        data['L_FUNDTYPE'] = data['PORTFOLIO_CODE'].str.contains('SMA').map({True: '3', False: '1'})  # 公募or专户
        data['L_FUNDKIND2'] = data['C_FULLNAME'].str.contains('货币').map({True: '5'})  # 识别货币基金
        data['C_STOPINFO'] = data['C_STOPINFO'].replace(np.nan, '')
        data['C_SUBCODE'] = data['C_SUBCODE'].replace(np.nan, '')
        data['C_FUNDNAME'] = data['PORTFOLIO_CODE'].map(self.code_to_fundname)
        data['C_SUBNAME'] = data['SEC_ABBR'].copy()   # 后续会用到
        data['C_SUBNAME_BSH'] = data['SEC_ABBR'].copy()  # 后续都需要用到证券简称字段
        data['L_STOCKTYPE'] = data['SEC_TYPE_L1'].map(config['sec_type_map'])  # 证券类型映射
        data = data.dropna(subset=['C_FUNDNAME'])

        if self.ptf_codes is not None:
            data = data[data['PORTFOLIO_CODE'].isin(self.ptf_codes)].copy()
        if data.empty:
            logger.warning('估值表中无下述产品相关数据：%s' % self.ptf_codes)
            exit(1)

        if t == '':
            self.val = data.copy()
            self.fund_list = self.val['C_FUNDNAME'].unique().tolist()
            self.ptf_list = self.val['PORTFOLIO_CODE'].unique().tolist()
            self.date_list = self.val['D_DATE'].drop_duplicates().to_list()
        else:
            return data

    def check_nav_data(self):
        '''检查基期的单位净值数据是否有误，弹窗提醒'''
        q_val = sqls_config['nav_check']['Sql']%self.basedate
        nav_check = self.db_risk.read_sql(q_val)
        if nav_check['total_num'].iloc[0] > 0:
            logger.info('单位净值数据已插入')

            q_ret = sqls_config['return_check']['Sql']%self.basedate
            ret_check = self.db_risk.read_sql(q_ret)
            if ret_check.shape[0] == 0:
                logger.info('单位净值及收益率数据无误')
            else:
                logger.error('当日收益率数据有误，详见dpe_nav_check表')
                print(ret_check)
                title = 'Error'
                message = '当日收益率数据有误，请检查。是否继续运行程序？'
                # show_catch_message(title, message)
        else:
            logger.error('单位净值表无数据')
            title = 'Error'
            message = '单位净值表无数据，是否继续运行程序？'
            show_catch_message(title, message)

    def _loadNavData(self):
        '''取单位净值数据（成立以来）'''
        # prods = self.val.loc[self.val['C_MANAGER'] == '1', 'C_FUNDNAME'].drop_duplicates().tolist()
        prods = self.val.loc[~self.val['C_FUNDNAME'].isna(), 'C_FUNDNAME'].unique().tolist()
        prods_str = ','.join('\'' + x + '\'' for x in prods)

        sql_nav = sqls_config['nav_data']['Sql']%prods_str
        res = self.db_risk.read_sql(sql_nav)
        res.columns = [x.upper() for x in res.columns]
        res = res.rename(columns={'NAV':'NAV_orig', 'NAV_ADJ': 'NAV', 'RETURN':'ret'})
        self.nav_f = res[res['D_DATE'] <= pd.to_datetime(self.basedate)].dropna(subset=['NAV']).sort_values(by=['C_FUNDNAME', 'D_DATE']).drop_duplicates()

    def _loadTableColumns(self):
        '''基础数表的列名汇总'''
        col_path = os.path.join(DIR_OF_MAIN_PROG, 'data', '常量表', 'table_columns.xlsx')
        col_df = pd.read_excel(col_path, sheet_name='table_columns', header=0, engine='openpyxl')
        col_dict = col_df.set_index('TABLE_NAME')['TABLE_COLUMNS'].to_dict()

        self.column_dict = {}
        for key, value in col_dict.items():
            self.column_dict[key] = value.split(',')

    def _loadSpreadAD(self):
        '''取信用利差调整的基础数据'''
        sql_spread_ad = sqls_config['dpe_spread_ad']['Sql']
        self._spreadAD = self.db_risk.read_sql(sql_spread_ad)
        self._spreadAD.columns = ['隐含评级', '类别', 'spread_bp', '是否城投']
    
    def _loadIndustryID(self):
        '''取行业利差与其ID的匹配表'''
        sql_industry_id = sqls_config['dpe_industry_id']['Sql']
        self._IndustryID = self.db_risk.read_sql(sql_industry_id)
        self._IndustryID.columns = ['行业分类', '行业利差ID', '利差分类', '城投or产业', '行业分类2']

    def _loadMasterData(self):
        '''取债券的masterdata'''
        q = sqls_config['masterdata']['Sql']
        self._masterData = self.db_risk.read_sql(q)

    def _loadHoldings(self, save_path):
        '''加载当日持仓明细数据(中间数据)及回购明细数据'''
        file_db = 'data_db.xlsx' if self.ptf_codes is None else 'data_db_t.xlsx'
        data_db = pd.read_excel(save_path + file_db, sheet_name=None, engine='openpyxl')
        self.data_jy = data_db.get('data_jy')
        self.data_wind = data_db.get('data_wind')

        data_benchmark = pd.read_excel(save_path + 'data_benchmark.xlsx', sheet_name=None, engine='openpyxl')
        self._benchmark = data_benchmark.get('benchmark')
        self._benchmark_gk = data_benchmark.get('benchmark_gk')
        self._benchmark_ind = data_benchmark.get('benchmark_ind')
        self._benchmark_rating = data_benchmark.get('benchmark_rating')

        file_holdings = 'Holdings.xlsx' if self.ptf_codes is None else 'Holdings_t.xlsx'
        holdings = pd.read_excel(save_path + file_holdings, sheet_name=None, engine='openpyxl')
        self.stock_holdings = holdings.get('stock_holdings')
        self.bond_holdings = holdings.get('bond_holdings')
        self.holdings = holdings.get('holdings')

        file_fund = 'data_fund.xlsx' if self.ptf_codes is None else 'data_fund_t.xlsx'
        self.data_fund = pd.read_excel(save_path + file_fund, sheet_name='Sheet1', engine='openpyxl')

        file_repo = '_repo_all.xlsx' if self.ptf_codes is None else '_repo_all_t.xlsx'
        self._repo_all = pd.read_excel(save_path + file_repo, sheet_name='Sheet1', engine='openpyxl')
        if 'd_date' in self._repo_all.columns:
            self._repo_all = self._repo_all.rename(columns={'d_date': 'D_DATE', 'c_fundname': 'C_FUNDNAME'})
            self._repo_all['D_DATE'] = pd.to_datetime(self._repo_all['D_DATE'])

        file_lev = '_lev_all.xlsx' if self.ptf_codes is None else '_lev_all_t.xlsx'
        self._lev_all = pd.read_excel(save_path + file_lev, sheet_name='Sheet1', engine='openpyxl')
        if 'd_date' in self._lev_all.columns:
            self._lev_all = self._lev_all.rename(columns={'d_date': 'D_DATE', 'c_fundname': 'C_FUNDNAME'})
            self._lev_all['D_DATE'] = pd.to_datetime(self._lev_all['D_DATE'])

        print('Holdings Cleaning Done.')

    def __dealRepoFile(self):
        '''【已弃用】取回购明细表，数据源为公盘27.RiskQuant/Data/Collateral'''
        warnings.warn('此方法在数据源切换为数据库idc_repo表后已弃用，不推荐使用', DeprecationWarning)

        baseDay = self.val['D_DATE'].iloc[0].strftime('%Y%m%d')
        path_day = self.path_repo + baseDay
        if not os.path.exists(path_day + '\\综合信息查询_交易所回购.xls'):
            return pd.DataFrame(columns=['回购日期', '基金名称', '回购代码', '委托方向', '数量', '净资金额', '返回金额', '平均利率(%)','法定购回日期', '实际购回日期', '回购天数', '计算方向', 'C_FUNDNAME'])
        
        repo_ex = pd.read_excel(path_day + '\\综合信息查询_交易所回购.xls', skipfooter=1)
        repo_ib = pd.read_excel(path_day + '\\综合信息查询_银行间回购.xls', skipfooter=1)

        cols = ['回购日期', '基金名称', '回购代码', '委托方向', '数量', '净资金额', '返回金额', '平均利率(%)', '法定购回日期', '实际购回日期']
        repo_all = repo_ex[cols].append(repo_ib[cols], sort=False)
        repo_all['回购日期'] = pd.to_datetime(repo_all['回购日期'])
        repo_all['法定购回日期'] = pd.to_datetime(repo_all['法定购回日期'])
        repo_all['实际购回日期'] = pd.to_datetime(repo_all['实际购回日期'])
        repo_all['回购天数'] = [x.days for x in repo_all['法定购回日期'] - repo_all['回购日期']]
        repo_all['计算方向'] = repo_all['委托方向'].map({'融券回购': 0, '融资回购': -1, '正回购': -1})
        repo_all = repo_all[repo_all['回购日期'] <= pd.to_datetime(baseDay)].copy()
        repo_all['D_DATE'] = pd.to_datetime(baseDay)

        fundNames = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl')[['O32产品名称', '估值系统简称']]
        fundNames.columns = ['基金名称', 'C_FUNDNAME']

        repo_all = pd.merge(repo_all, fundNames, on='基金名称', how='left')
        repo_all = repo_all.reset_index().rename(columns={'index': 'c_key'})
        repo_all = repo_all.reindex(columns=repo_all.columns[1:].tolist() + ['c_key'])
        
        return repo_all

    def _deal_repo(self):
        '''清洗回购明细，取自数据库idc_repo表，并将清洗后的回购明细插入数据库dpe_repoall'''
        q = sqls_config['idc_repo_t']['Sql'] % self.basedate
        repo_all = self.db_risk.read_sql(q).rename(columns={'c_date': 'd_date'})
        repo_all = repo_all[(repo_all['actu_matu_date'] > self.basedate) & (repo_all['repo_date'] <= self.basedate)].copy()
        repo_all['repo_date'] = pd.to_datetime(repo_all['repo_date'])
        repo_all['buyback_date_legal'] = pd.to_datetime(repo_all['legal_matu_date'])
        repo_all['buyback_date_real'] = pd.to_datetime(repo_all['actu_matu_date'])
        repo_all['direction_num'] = repo_all['repo_dir'].map({'融券回购': 0, '融资回购': -1, '正回购': -1})
        repo_all['balance_all'] = repo_all[['repo_nprc', 'repo_intr']].sum(axis=1)

        repo_all['amount'] = repo_all['repo_vol'] * ((-1) * (repo_all['repo_dir'].isin(['融资回购', '正回购'])) + 1 * (repo_all['repo_dir'].isin(['融券回购'])))
        repo_all['balance_net'] = repo_all['repo_nprc'] * (1 * (repo_all['repo_dir'].isin(['融资回购', '正回购'])) + (-1) * (repo_all['repo_dir'].isin(['融券回购'])))
        repo_all['balance_all'] = repo_all['balance_all'] * ((-1) * (repo_all['repo_dir'].isin(['融资回购', '正回购'])) + 1 * (repo_all['repo_dir'].isin(['融券回购'])))
        repo_all['repo_code'] = [x[:6] if x[:2] != 'HG' else x.replace('HG', '').replace('YH', '') for x in repo_all['repo_code']]
        repo_all = repo_all.reset_index().rename(columns={'index': 'c_key'})

        col_map = {'repo_intr_rate': 'interest_rate', 'repo_dir': 'direction'}
        cols_new = ['repo_date', 'c_fundname_o32', 'repo_code', 'direction', 'amount', 'balance_net', 'balance_all',
                    'interest_rate', 'buyback_date_legal', 'buyback_date_real', 'repo_days', 'direction_num',
                    'd_date', 'c_fundname', 'c_key']
        repo_all = repo_all.rename(columns=col_map).reindex(columns=cols_new + ['portfolio_code'])
        repo_all['c_fundname_o32'] = repo_all['c_fundname']

        if self.ptf_codes is not None:
            repo_all = repo_all[repo_all['portfolio_code'].isin(self.ptf_codes)].copy()

        if not repo_all.empty:
            self.insert2db_single('dpe_repoall', repo_all, 'quant', ptf_code=self.ptf_codes, code_colname='portfolio_code')

        self._repo_all = repo_all.reindex(columns=cols_new).rename(columns={'d_date': 'D_DATE', 'c_fundname': 'C_FUNDNAME'})
        self._repo_all['D_DATE'] = pd.to_datetime(self._repo_all['D_DATE'])
        return repo_all

    # def __loadLeverageCost(self):
    #     '''【已弃用】清洗各组合杠杆成本的数据，与__dealRepoFile一同弃用'''
    #     warnings.warn('此方法在数据源切换为数据库idc_repo表后已弃用，不推荐使用', DeprecationWarning)
    #
    #     self._repo_all = self.__dealRepoFile()
    #     if self._repo_all.shape[0] == 0:
    #         self._lev_all = pd.DataFrame(columns=['基金名称', 'Lev_cost', 'Lev_amt', 'C_FUNDNAME', 'D_DATE'])
    #     else:
    #         lev_cost = self._repo_all.groupby(['基金名称']).apply(lambda x: (x['数量']*x['计算方向']*x['平均利率(%)']/100).sum()/(x['数量']*x['计算方向']).sum()).rename('Lev_cost').reset_index()
    #         repo_amt = self._repo_all.groupby(['基金名称']).apply(lambda x: (x['数量']*x['计算方向']*100).sum()).rename('Lev_amt').reset_index()
    #         lev_all = pd.merge(lev_cost, repo_amt, on=['基金名称'], how='left')
    #
    #         fundNames = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl')[['O32产品名称', '估值系统简称']]
    #         fundNames.columns = ['基金名称', 'C_FUNDNAME']
    #
    #         lev_all = pd.merge(lev_all, fundNames, on='基金名称', how='left')
    #         baseDay = self.val['D_DATE'].iloc[0]
    #         lev_all['D_DATE'] = baseDay
    #
    #         self._lev_all = lev_all.copy()

    def _fundNameMapping(self):
        '''产品全称与估值表简称对应表'''
        q = sqls_config['portfolio_type']['Sql']
        self.ptf_info = self.db_risk.read_sql(q)
        self.name_dict = self.ptf_info.set_index('c_fullname')['c_fundname'].to_dict()
        self.code_to_fundname = self.ptf_info.set_index('c_fundcode')['c_fundname'].to_dict()
        self.fundname_to_code = self.ptf_info.set_index('c_fundname')['c_fundcode'].to_dict()
        self.fullname_to_code = self.ptf_info.set_index('c_fullname')['c_fundcode'].to_dict()

    def _formatCurveMap(self):
        '''信用评级曲线代码匹配表'''
        sql_curve_map = sqls_config['dpe_curve_map']['Sql']
        self._yield_map = self.db_risk.read_sql(sql_curve_map)
        self._yield_map.columns = ['曲线名称', '曲线代码', 'WINDL2TYPE_curve', 'MUNICIPALBOND', 'RATE_LATESTMIR_CNBD']

    def _formatHolding(self, data=''):
        '''
        从估值表清洗证券持仓信息，默认清洗基期的持仓\n
        :param data: DataFrame, 原始估值表
        :return: 证券持仓基础信息表
        '''
        if type(data) == str:
            data = self.val.copy()
        cols = ['D_DATE', 'C_FULLNAME', 'C_FUNDNAME', 'code', 'C_SUBNAME_BSH', 'C_STOPINFO', 'F_MOUNT', 'F_PRICE',
                'F_ASSET', 'F_ASSETRATIO', 'F_NETCOST', 'F_COST', 'F_COSTRATIO', 'L_STOCKTYPE', 'L_FUNDKIND',
                'L_FUNDKIND2', 'L_FUNDTYPE']
        # 取出有交易字段的持仓数据
        data_filter = data.loc[data['C_STOPINFO'] != '', :].reindex(columns=cols).drop_duplicates(subset=cols[:-1])

        # 剔除无code的条目
        holdings = data_filter[~data_filter['code'].isnull()].copy()
        cols_num = ['F_MOUNT', 'F_ASSET', 'F_ASSETRATIO', 'F_COST', 'F_COSTRATIO']
        for col in cols_num:
            holdings[col] = holdings[col].astype(float)

        # 合并续发券和老券的持仓
        bond_hld = holdings[holdings['L_STOCKTYPE'] == '2'].copy()
        basic_cols = ['D_DATE', 'C_FULLNAME', 'C_FUNDNAME', 'code', 'C_SUBNAME_BSH', 'C_STOPINFO',
                      'L_STOCKTYPE', 'L_FUNDTYPE']
        df_duplicates = bond_hld[bond_hld.duplicated(subset=basic_cols, keep=False)].copy()
        if df_duplicates.empty:
            logger.info('无重复持仓')
            return holdings
        keep_df = bond_hld[~bond_hld.duplicated(subset=basic_cols, keep=False)].copy()
        sum_col = ['F_MOUNT', 'F_ASSET', 'F_ASSETRATIO', 'F_COST', 'F_COSTRATIO']
        avg_col = ['F_PRICE', 'F_NETCOST']
        agg_func = dict(zip(sum_col + avg_col, ['sum'] * 5 + ['mean'] * 2))
        df_adj = df_duplicates.groupby(basic_cols).agg(agg_func).reset_index()
        holdings = pd.concat([holdings[holdings['L_STOCKTYPE'] != '2'], keep_df, df_adj], ignore_index=True)

        return holdings
    
    def _formatBondType(self, data=''):
        '''
        规范债券分类，利率债VS信用债、产业债VS城投债、高中低评级等分类。默认清洗基期债券持仓\n
        :param data: DataFrame, 债券持仓基础信息表，非估值表or全量证券持仓表
        :return: DataFrame, 含债券分类的债券持仓数据
        '''
        # 须有WINDL1TYPE、WINDL2TYPE、MUNICIPALBOND、RATE_LATESTMIR_CNBD、INDUSTRY_SW列
        if type(data) == str:
            bond_holding = self.bond_holdings.copy()
        else:
            bond_holding = data.copy()

        # todo: 如果出现ABS的wind分类缺失的情况
        # abs_type = self.db_risk.read_sql(sqls_config['idc_position_abs']['sql'].format(t=self.basedate))
        # bond_holding = bond_holding.merge(abs_type, how='left', left_on='code', right_on='abs_code')
        # bond_holding['WINDL1TYPE'] = bond_holding['WINDL1TYPE'].fillna(bond_holding['abs_type1'])
        # bond_holding['WINDL2TYPE'] = bond_holding['WINDL2TYPE'].fillna(bond_holding['abs_type2'])
        # bond_holding.drop(list(abs_type.columns), axis=1, inplace=True)

        # 补充WINDL1TYPE,WINDL2TYPE的类型 - 20230920
        miss_type = pd.read_excel(r"\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\AssetType\wind一二级分类缺失数据补充.xlsx",
                                  sheet_name='Sheet1', engine='openpyxl')

        bond_holding = bond_holding.merge(miss_type, how='left', left_on='code', right_on='code_missing')
        bond_holding['WINDL1TYPE'] = bond_holding['WINDL1TYPE'].fillna(bond_holding['windl1type_missing'])
        bond_holding['WINDL2TYPE'] = bond_holding['WINDL2TYPE'].fillna(bond_holding['windl2type_missing'])
        bond_holding.drop(list(miss_type.columns), axis=1, inplace=True)

        # 区分利率债\信用债
        bond_holding['利率or信用'] = np.where(bond_holding['WINDL2TYPE'].isin(self._InterestBond), '利率债',
                                          np.where(bond_holding['WINDL2TYPE'].isin(self._CBond), '可转债',
                                          np.where(~bond_holding['WINDL2TYPE'].isna(), '信用债', '无分类')))

        # 区分城投、产业、金融债，计入“城投or产业”字段(内评的城投口径)
        # todo: ABS是否单独区分
        municpal_df = self.db_risk.read_sql(sqls_config['inner_municipal']['Sql'].format(t=self.basedate))
        municipal_map = municpal_df.set_index('code')['municipalbond'].to_dict()
        bond_holding['MUNICIPALBOND'] = bond_holding['code'].map(municipal_map).fillna('否')
        bond_holding['城投or产业'] = np.where(bond_holding['WINDL2TYPE'].isin(self._InterestBond), '利率债',
                                    np.where(bond_holding['WINDL2TYPE'].isin(self._CBond), '可转债',
                                    np.where(bond_holding['WINDL1TYPE'] == '同业存单', 'CD',
                                    np.where(bond_holding['MUNICIPALBOND'] == '是', '城投债',
                                    np.where(bond_holding['WINDL1TYPE'] == '金融债', '金融债',
                                    np.where(~bond_holding['WINDL2TYPE'].isna(), '产业债', '无分类'))))))

        # 处理利率债无评级
        bond_holding.loc[bond_holding['WINDL2TYPE'].isin(self._InterestBond), 'RATE_LATESTMIR_CNBD'] = '评级_利率债'
        bond_holding['RATE_LATESTMIR_CNBD'] = bond_holding['RATE_LATESTMIR_CNBD'].fillna('无评级')

        # 处理利率债无行业分类
        bond_holding.loc[bond_holding['WINDL2TYPE'].isin(self._InterestBond), 'INDUSTRY_SW'] = '行业_利率债'
        bond_holding.loc[bond_holding['城投or产业'] == '城投债', 'INDUSTRY_SW'] = '城投债'

        # 处理高中低评级分类
        bond_holding['评级分类'] = bond_holding['RATE_LATESTMIR_CNBD'].map({'AAA+': '高评级', 'AAA': '高评级', 'AAA-': '高评级', 'AA+': '高评级', 'AA': '中评级', 'AA(2)': '中评级', 'AA-': '低评级', '评级_利率债': '高评级_利率债', '无评级': '无评级'}).fillna('低评级')
        bond_holding.loc[bond_holding['RATE_LATESTMIR_CNBD'].isna(), '评级分类'] = '无评级'
        
        return bond_holding

    def _dealSpread(self, data=''):
        '''
        依据债券持仓清洗债券ytm及对应不同基准的不同利差spread数据(含特殊品种债券的spread调整)。默认清洗基期债券持仓\n
        :param data: DataFrame, 债券持仓基础信息表
        :return:
            (1) bond_holdings2: DataFrame, 债券持仓明细
            (2) benchmark: DataFrame, 中短票AAA+曲线各期限收益率数据
            (3) benchmark_gk: DataFrame, 国开各期限收益率数据
            (4) benchmark_rating: DataFrame, 中短票各评级各期限收益率数据
            (5) benchmark_ind: DataFrame, 行业利差数据(兴业研究，更新时间不确定，且部分行业现已停更)
        '''
        # 须有WINDL1TYPE、WINDL2TYPE、MUNICIPALBOND、RATE_LATESTMIR_CNBD、RATE_RATEBOND1、INDUSTRY_SW、CLAUSE列
        if type(data) == str:
            bond_holdings = self.bond_holdings.copy()
        else:
            bond_holdings = data.copy()
        
        # 获取基础Benchmark
        bond_holdings, benchmark = self.retrieveBenchmark(bond_holdings)       # 中短票AAA+曲线
        bond_holdings, benchmark_gk = self.retrieveBenchmark_gk(bond_holdings)    # 国开债曲线

        # 匹配对应评级的曲线收益率，数据来源：JY
        bond_holdings['WINDL2TYPE_curve'] = bond_holdings['WINDL2TYPE'].copy()
        bond_holdings.loc[bond_holdings['MUNICIPALBOND'] == '是', 'WINDL2TYPE_curve'] = '城投债'        # 城投债优先
        bond_holdings = pd.merge(bond_holdings, self._yield_map, on=['WINDL2TYPE_curve', 'MUNICIPALBOND', 'RATE_LATESTMIR_CNBD'], how='left')
        bond_holdings = bond_holdings.drop(columns=['WINDL2TYPE_curve'])

        if '信用债' not in bond_holdings['利率or信用'].unique().tolist():  # 持仓中无信用券
            add_cols = ['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY', 'benchmark_rating', 'spread_rating',
                        '行业分类2', '行业利差ID', '利差分类', '行业利差']
            bond_holdings2 = bond_holdings.drop(['ENDDATE', 'YEARSTOMATURITY'], axis=1)
            bond_holdings2 = bond_holdings2.reindex(columns=list(bond_holdings2.columns) + add_cols)
            benchmark_rating = pd.DataFrame(columns=self.column_dict['_benchmark_rating'])
            benchmark_ind = pd.DataFrame(columns=self.column_dict['_benchmark_ind'])
        else:
            bond_holdings1, benchmark_rating = self.retrieveBenchmark_Rating(bond_holdings)

            # 匹配兴业的行业利差数据，数据来源：wind
            bond_holdings1['行业分类2'] = bond_holdings1['INDUSTRY_SW'] + bond_holdings1['RATE_RATEBOND1']
            bond_holdings1 = pd.merge(bond_holdings1, self._IndustryID.drop(columns='行业分类'), on=['城投or产业', '行业分类2'], how='left')
            bond_holdings2, benchmark_ind = self.retrieveBenchmark_ind(bond_holdings1)

        # 处理私募、永续、提前偿还溢差
        bond_holdings2['永续'] = ['延期' in x for x in bond_holdings2['CLAUSE'].fillna('*')]
        bond_holdings2['永续'] = bond_holdings2['永续'].map({True: '永续'})
        bond_holdings2['提前偿还'] = ['提前偿还' in x for x in bond_holdings2['CLAUSE'].fillna('*')]
        bond_holdings2['提前偿还'] = bond_holdings2['提前偿还'].map({True: '提前偿还'})

        bond_holdings2 = pd.merge(bond_holdings2, self._spreadAD.rename(columns={'类别': '发行方式','spread_bp': '私募bp'}), left_on=['RATE_LATESTMIR_CNBD', 'ISSUE_ISSUEMETHOD', 'MUNICIPALBOND'], right_on=['隐含评级', '发行方式', '是否城投'], how='left').drop(columns=['隐含评级', '是否城投'])
        bond_holdings2 = pd.merge(bond_holdings2, self._spreadAD.rename(columns={'类别': '永续','spread_bp': '永续bp'}), left_on=['RATE_LATESTMIR_CNBD', '永续', 'MUNICIPALBOND'], right_on=['隐含评级', '永续', '是否城投'], how='left').drop(columns=['隐含评级', '是否城投'])
        bond_holdings2 = pd.merge(bond_holdings2, self._spreadAD.rename(columns={'类别': '提前偿还','spread_bp': '提前偿还bp'}), left_on=['RATE_LATESTMIR_CNBD', '提前偿还', 'MUNICIPALBOND'], right_on=['隐含评级', '提前偿还', '是否城投'], how='left').drop(columns=['隐含评级', '是否城投'])

        bond_holdings2['bp_ad'] = bond_holdings2[['私募bp', '永续bp', '提前偿还bp']].sum(axis=1)
        bond_holdings2['spread_ad'] = bond_holdings2['spread'] - bond_holdings2['bp_ad']
        bond_holdings2['spread_ad_zd'] = bond_holdings2['spread_ad'].copy()
        bond_holdings2['spread_ad'] = bond_holdings2['spread_gk'] - bond_holdings2['bp_ad']   # 将信用利差改为对标国开债的信用利差，原有对标AAA+中短票的利差记作spread_ad_zd
        bond_holdings2['spread_rating_ad'] = bond_holdings2['spread_rating'] - bond_holdings2['bp_ad']
        bond_holdings2['spread_ind_ad'] = bond_holdings2['spread_rating'] - bond_holdings2['行业利差']

        return bond_holdings2, benchmark, benchmark_gk, benchmark_rating, benchmark_ind

    def __formatStockHoldings(self, data_holdings='', data_indice=''):
        '''
        清洗股票持仓明细，默认清洗基期，结果录入self。 \n
        :param data_holdings: DataFrame, 全量证券持仓表
        :param data_indice: DataFrame, 聚源的股票mv(市值)\pe\pb等数据
        :return: 清洗后的股票持仓
        '''
        status = type(data_holdings) == str   # 若从data_val开始均未给参数，则视为默认初始化清洗；若给定某个val，则该函数有返回值
        if status:
            holdings = self.holdings.copy()
            data_indice = self.data_jy_equity.copy()
        else:
            holdings = data_holdings.copy()
            
        stock_holdings = holdings[holdings['L_STOCKTYPE'] == '1'].copy()
        if stock_holdings.empty:
            if status:
                self.stock_holdings = pd.DataFrame(columns=self.column_dict['stock_holdings'])
            return self.stock_holdings

        stock_holdings = pd.merge(stock_holdings, data_indice, on=['code', 'D_DATE'], how='left')
        if status:
            self.stock_holdings = stock_holdings.copy()
        else:
            return stock_holdings

    def __formatBondHoldings(self, data_holdings='', data_wind='', data_jy=''):
        '''
        清洗债券持仓数据，包含债券的基础信息、估值收益率、分类等各项信息\n
        :param data_holdings: DataFrame, 全量证券持仓表
        :param data_wind: DataFrame, wind的债券基础信息表
        :param data_jy: DataFrame, 聚源的债券基础信息表
        :return: 清洗后的债券持仓表及相关数据表
        '''
        status = type(data_holdings) == str   # 若从data_val开始均未给参数，则视为默认初始化清洗；若给定某个val，则该函数有返回值
        if status:
            holdings = self.holdings.copy()
            data_wind = self.data_wind.copy()
            data_jy = self.data_jy.copy()
        else:
            holdings = data_holdings.copy()

        bond_holdings = holdings[holdings['L_STOCKTYPE'] == '2'].copy()
        if bond_holdings.empty:
            if status:
                self.bond_holdings = pd.DataFrame(columns=self.column_dict['bond_holdings'])
                self._benchmark = pd.DataFrame(columns=self.column_dict['_benchmark'])
                self._benchmark_gk = pd.DataFrame(columns=self.column_dict['_benchmark_gk'])
                self._benchmark_rating = pd.DataFrame(columns=self.column_dict['_benchmark_rating'])
                self._benchmark_ind = pd.DataFrame(columns=self.column_dict['_benchmark_ind'])
            return self.bond_holdings, self._benchmark, self._benchmark_gk, self._benchmark_rating, self._benchmark_ind

        bond_holdings = pd.merge(bond_holdings, data_wind, on=['code', 'D_DATE'], how='left')
        bond_holdings = pd.merge(bond_holdings, data_jy, on=['code', 'D_DATE'], how='left')
        bond_holdings = self._formatBondType(bond_holdings)
        bond_holdings2, benchmark, benchmark_gk, benchmark_rating, benchmark_ind = self._dealSpread(bond_holdings)
        bond_holdings2['MODIDURA_CNBD'] = bond_holdings2['MODIDURA_CNBD'].fillna(0)
        bond_holdings2 = bond_holdings2.drop_duplicates()             # 地方政府债在jy里会取出两个估值信息

        if status:
            self.bond_holdings = bond_holdings2.copy()
            self._benchmark = benchmark.copy()
            self._benchmark_gk = benchmark_gk.copy()
            self._benchmark_rating = benchmark_rating.copy()
            self._benchmark_ind = benchmark_ind.copy()
        else:
            return bond_holdings2, benchmark, benchmark_gk, benchmark_rating, benchmark_ind

    def __retrieveWindData(self, data=''):
        '''
        从wind取债券的相关信息，如wind债券分类、外部评级、中债隐含评级等\n
        :param data: DataFrame, 全量证券持仓表, 用于获取债券代码清单
        :return: DataFrame, wind债券基础信息
        '''
        if type(data) == str:
            holdings = self.holdings.copy()
        else:
            holdings = data.copy()

        # # 回溯历史数据：直接从data中获取历史已经取好的wind数据
        # t = holdings['D_DATE'].iloc[0].strftime('%Y-%m-%d')
        # save_path = os.path.join(DIR_OF_MAIN_PROG, 'data') + '\\%s\\' % t.replace('-', '')
        # file_db = 'data_db.xlsx' if self.ptf_codes is None else 'data_db_t.xlsx'
        # data_db = pd.read_excel(save_path + file_db, sheet_name=None, engine='openpyxl')
        # data_wind = data_db.get('data_wind')

        start = time.time()
        bond_list = holdings.loc[holdings['L_STOCKTYPE'] == '2', 'code'].unique().tolist()
        if len(bond_list) == 0:
            return pd.DataFrame(columns=self.column_dict['data_wind'])

        date_list = holdings['D_DATE'].unique()

        if '_masterData' in dir(self):
            data_temp = self._masterData.loc[self._masterData['万德代码'].isin(bond_list), ['万德代码', 'Wind债券一级分类', 'Wind债券二级分类', '发行人名称', '担保人名称', '发行方式', '起息日期', '到期日期', '申万一级', '是否次级债', '是否混合资本债', '特殊条款(万德)', '回售日期']]
            data_wind1 = data_temp.rename(columns={'万德代码': 'code', 'Wind债券一级分类': 'WINDL1TYPE', 'Wind债券二级分类': 'WINDL2TYPE', '发行方式': 'ISSUE_ISSUEMETHOD', '申万一级': 'INDUSTRY_SW', '特殊条款(万德)': 'CLAUSE', '回售日期': 'REPURCHASEDATE'})
        else:
            wind_temp1 = w.wss(bond_list, "windl2type,windl1type,industry_sw,clause,issue_issuemethod,repurchasedate","industryType=1")
            data_wind1 = pd.DataFrame(wind_temp1.Data, columns=wind_temp1.Codes, index=wind_temp1.Fields).T
            data_wind1 = data_wind1.reset_index().rename(columns={'index': 'code'})

        # masterdata中缺少的债券类别信息，用WIND api进行补充
        miss_bonds = [i for i in bond_list if i not in data_wind1['code'].to_list()]  # master data中缺失的债券
        miss_bonds += data_wind1.loc[data_wind1['WINDL1TYPE'].isna(), 'code'].to_list()  # master data中缺失的wind分类
        if len(miss_bonds) > 0:
            add_temp = w.wss(miss_bonds, "windl1type,windl2type,issuerupdated,agency_guarantor,issue_issuemethod,carrydate,maturitydate,subordinateornot,mixcapital,clause,repurchasedate, industry_sw","industryType=1")

            cols = {'index': 'code', 'ISSUERUPDATED': '发行人名称', 'AGENCY_GUARANTOR': '担保人名称', 'CARRYDATE': '起息日期',
                    'MATURITYDATE': '到期日期', 'SUBORDINATEORNOT': '是否次级债', 'MIXCAPITAL': '是否混合资本债'}
            add_df = pd.DataFrame(add_temp.Data, columns=add_temp.Codes, index=add_temp.Fields).T.reset_index()
            add_df = add_df[~add_df['WINDL1TYPE'].isnull()].copy().rename(columns=cols)
            data_wind1 = pd.concat([data_wind1[~data_wind1['WINDL1TYPE'].isna()], add_df], ignore_index=True)

        data_wind2 = pd.DataFrame(columns=['Date', 'NATURE1', 'RATE_LATESTMIR_CNBD', 'COUPONRATE3', 'YIELD_CNBD', 'PTMYEAR', 'MODIDURA_CNBD', 'LATESTISSURERCREDITRATING2', 'RATE_RATEBOND', 'RATE_RATEGUARANTOR'])
        for temp_date in date_list:
            wind_temp = w.wss(bond_list, "nature1,rate_latestMIR_cnbd,couponrate3,yield_cnbd,matu_cnbd,modidura_cnbd,latestissurercreditrating2,rate_ratebond,rate_rateguarantor","tradeDate=%s;ratingAgency=101;type=1;credibility=1"%pd.to_datetime(temp_date).strftime('%Y%m%d'))
            wind_df = pd.DataFrame(wind_temp.Data, columns=wind_temp.Codes, index=wind_temp.Fields).T.rename(columns={'MATU_CNBD': 'PTMYEAR'})
            wind_df['Date'] = temp_date
            data_wind2 = data_wind2.append(wind_df, sort=False)
        data_wind2 = data_wind2.reset_index().rename(columns={'index': 'code', 'Date': 'D_DATE'})

        data_wind = pd.merge(data_wind1, data_wind2, on=['code'], how='outer')
        data_wind['PTMYEAR'] = data_wind['PTMYEAR'].astype(float)
        data_wind['RATE_RATEBOND1'] = data_wind['RATE_RATEBOND'].fillna(value=data_wind['LATESTISSURERCREDITRATING2'])
        data_wind.loc[data_wind['WINDL2TYPE'].isin(['证券公司短期融资券', '一般短期融资券', '超短期融资债券']), 'RATE_RATEBOND1'] = data_wind.loc[data_wind['WINDL2TYPE'].isin(['证券公司短期融资券', '一般短期融资券', '超短期融资债券']), 'LATESTISSURERCREDITRATING2']
        data_wind['RATE_RATEBOND1'] = data_wind['RATE_RATEBOND1'].fillna('无评级')

        end = time.time()
        print('Total Time Cost:', end - start)

        return data_wind
    
    def __retrieveWindData_equity(self, data=''):
        '''
        【已弃用】wind的股票申万一级行业分类数据\n
        :param data: DataFrame, 全量证券持仓表, 用于获取股票代码清单
        :return: DataFrame, wind股票申万一级行业分类信息
        '''
        if type(data) == str:
            holdings = self.holdings.copy()
        else:
            holdings = data.copy()
        
        equity_list = holdings.loc[holdings['L_STOCKTYPE'] == '1', 'code'].unique().tolist()
        
        wind_temp = w.wss(equity_list, "industry_sw","industryType=1")
        data_wind = pd.DataFrame(wind_temp.Data, columns=wind_temp.Codes, index=wind_temp.Fields).T
        data_wind = data_wind.reset_index().rename(columns={'index': 'code'})

        return data_wind

    def __retrieveJYData(self, data=''):
        '''
        聚源的债券ytm和久期数据\n
        :param data: DataFrame, 全量证券持仓表, 用于获取债券代码清单
        :return: DataFrame,聚源的债券基础数据
        '''
        if type(data) == str:
            holdings = self.holdings.copy()
        else:
            holdings = data.copy()

        dq = JYDB_Query()
        bond_holding = holdings.loc[~holdings['code'].isna() & (holdings['L_STOCKTYPE'] == '2'), :].copy()

        if bond_holding.empty:
            return pd.DataFrame(columns=self.column_dict['data_jy'])

        date_list = bond_holding['D_DATE'].unique()
        
        data_jy = pd.DataFrame(columns=['ENDDATE', 'SECUCODE', 'SECUMARKET', 'VPYIELD', 'VPADURATION', 'VPCONVEXITY', 'CREDIBILITYCODE', 'CREDIBILITYDESC', 'YIELDCODE'])
        for temp_date in date_list:
            temp_date = pd.to_datetime(temp_date).strftime('%Y-%m-%d')
            temp_sec = bond_holding.loc[bond_holding['D_DATE'] == temp_date, 'code'].unique().tolist()
            jy_temp = dq.sec_query('bond_yield', temp_sec, temp_date)
            # jy_temp = dq.bond_yield(temp_sec, temp_date)
            data_jy = data_jy.append(jy_temp, sort=False)
            
        data_jy = self._dealCode(data_jy)
        data_jy = data_jy.rename(columns={'ENDDATE': 'D_DATE', 'VPADURATION': 'Duration', 'VPCONVEXITY': 'Convexity'}).drop_duplicates()  # 地方政府债在jy里会取出两个估值信息

        return data_jy

    def __retrieveJYData_equity(self, data=''):
        '''
        聚源的股票基础数据，如上市板别、总市值、pe、pb等\n
        :param data: DataFrame, 全量证券持仓表, 用于获取股票代码清单
        :return: DataFrame,聚源的股票基础数据
        '''
        if type(data) == str:
            holdings = self.holdings.copy()
        else:
            holdings = data.copy()

        dq = JYDB_Query()
        stock_holding = holdings.loc[~holdings['code'].isna() & (holdings['L_STOCKTYPE'] == '1'), :].copy()
        if stock_holding.empty:
            cols = ['D_DATE', 'LISTEDSECTOR', 'PE_TTM', 'PB', 'code', '上市板块']
            return pd.DataFrame(columns=cols)

        date_list = stock_holding['D_DATE'].unique()
        data_jy = pd.DataFrame(columns=['TRADINGDAY', 'SECUCODE', 'SECUMARKET', 'LISTEDSECTOR', 'TOTALMV', 'PE', 'PB'])
        for temp_date in date_list:
            temp_date = pd.to_datetime(temp_date).strftime('%Y-%m-%d')
            temp_sec = stock_holding.loc[stock_holding['D_DATE'] == temp_date, 'code'].unique().tolist()
            jy_temp = dq.sec_query('stock_eval_indice', temp_sec, temp_date)
            data_jy = data_jy.append(jy_temp, sort=False)
            
        data_jy = self._dealCode(data_jy)
        data_jy = pd.merge(data_jy, self._SectorMap, on=['LISTEDSECTOR'], how='left')
        data_jy = data_jy.rename(columns={'TRADINGDAY': 'D_DATE', 'PE': 'PE_TTM'})

        return data_jy

    def _dealCode(self, data):
        '''依据证券的上市市场处理证券代码后缀问题'''
        data0 = data.copy()
        data0['mkt'] = data0['SECUMARKET'].map({89: '.IB', 83: '.SH', 90: '.SZ', 18: '.BJ'})
        data0['code'] = data0['SECUCODE'] + data0['mkt']

        return data0.drop(columns=['SECUCODE', 'SECUMARKET', 'mkt'])

    def _dealRepurchaseDate(self):
        '''
        处理债券回售及赎回日期等信息\n
        :return: None, 结果记入self.bond_holdings变量
        '''
        data = self.bond_holdings.copy()
        if data.empty:
            return None

        bond_list = data['code'].unique().tolist()

        dq = JYDB_Query()
        data_rep = dq.sec_query('bond_repurchase', bond_list, 201)
        data_rep = self._dealCode(data_rep).drop(columns=['OPTYPE']).rename(columns={'EXPECTEDEXERCISEDATE': 'REPURCHASEDATE'}).dropna(subset=['REPURCHASEDATE'])
        data_rep = data_rep.groupby('code')['REPURCHASEDATE'].apply(lambda x: ','.join(i.strftime('%Y-%m-%d') for i in x)).reset_index()    # 将JY所有回售日期合并为一条

        data_call = dq.sec_query('bond_repurchase', bond_list, 101)
        data_call = self._dealCode(data_call).drop(columns=['OPTYPE']).rename(columns={'EXPECTEDEXERCISEDATE': 'CALLDATE'}).dropna(subset=['CALLDATE'])
        data_call = data_call.groupby('code')['CALLDATE'].apply(lambda x: ','.join(i.strftime('%Y-%m-%d') for i in x)).reset_index()

        data_option = pd.merge(data_rep, data_call, on='code', how='outer')

        data1 = pd.merge(data.rename(columns={'REPURCHASEDATE': 'REPURCHASEDATE_wind'}), data_option, on='code', how='left')
        self.bond_holdings = data1.copy()

    def getBasicData(self):
        '''获取产品组合的基础信息，如净值、总资产、份额、各大类资产持仓量等'''
        data = self.val.copy()

        # 基金类型
        data['基金类型'] = data['L_FUNDTYPE'].map({'1': '公募', '3': '专户'})
        data_kind = data[['PORTFOLIO_CODE', 'C_FULLNAME', 'C_FUNDNAME', '基金类型']].drop_duplicates()

        # 基金净值：单位净值、累计净值（组合层净值）
        q_nav = sqls_config['portfolio_nav']['Sql'].format(t=self.basedate)
        data_nav = self.db_risk.read_sql(q_nav).rename(columns={'nav_cum': 'NAV_累计'})
        data_nav['d_date'] = pd.to_datetime(data_nav['d_date'])
        data_nav.columns = [x.upper() for x in data_nav.columns]

        # 各大类资产规模
        cols_map = {'net_asset': 'NetAsset', 'total_asset': 'TotalAsset', 'total_shares': '基金份额',
                    'deposit': 'Deposit', 'stock': '股票', 'bond': '债券', 'convertible': '可转债', 'abs': 'ABS',
                    'fund': '基金投资', 'repo_reverse': '买入返售', 'repo': '卖出回购', 'derivatives': '衍生品',
                    'portfolio_code': 'PORTFOLIO_CODE'}
        q_asset = sqls_config['dws_main_asset']['Sql'].format(t=self.basedate)
        data_asset = self.db_risk.read_sql(q_asset).rename(columns=cols_map).drop('c_date', axis=1)

        data_fund = pd.merge(data_kind, data_nav, on=['PORTFOLIO_CODE'], how='left')
        data_fund = pd.merge(data_fund, data_asset, on=['PORTFOLIO_CODE'], how='left')

        cols = ['Deposit', '股票', '债券', '可转债', 'ABS', '基金投资', '买入返售', '卖出回购', '衍生品']
        for col in cols:
            if col in data_fund.columns:
                data_fund[col] = data_fund[col] / data_fund['TotalAsset']

        return data_fund

    def retrieveBenchmark(self, data=''):
        '''
        获取聚源中短票AAA+收益率曲线数据\n
        :param data: DataFrame, 全量债券持仓表, 用于获取债券代码清单
        :return:
            (1) bond_holdings1, DataFrame, 含债券ytm及对应收益率曲线的债券持仓表;
            (2) benchamrk: DataFrame, 聚源中票AAA+的收益率曲线数据
        '''
        if type(data) == str:
            bond_holdings = self.bond_holdings.copy()
        else:
            bond_holdings = data.copy()

        dq = JYDB_Query()

        date_list = bond_holdings['D_DATE'].unique()
        benchmark = pd.DataFrame(columns=['ENDDATE', 'YEARSTOMATURITY', 'YIELD'])
        for temp_date in date_list:
            temp_date = pd.to_datetime(temp_date).strftime('%Y-%m-%d')
            temp_b = dq.sec_query('curve_yield_all', '87', temp_date)    # 步长只有0.1, 87为中短票AAA+
            benchmark = benchmark.append(temp_b, sort=False)
        benchmark['YIELD'] = benchmark['YIELD']*100

        bond_holdings['剩余期限'] = [round(x, 1) for x in bond_holdings['PTMYEAR'].fillna(0)]    # 将取不出剩余期限的PTM取为0
        bond_holdings1 = pd.merge(bond_holdings, benchmark, left_on=['D_DATE', '剩余期限'], right_on=['ENDDATE', 'YEARSTOMATURITY'], how='left')
        bond_holdings1 = bond_holdings1.rename(columns={'YIELD': 'benchmark'})
        bond_holdings1['spread'] = bond_holdings1['YIELD_CNBD'] - bond_holdings1['benchmark']

        return bond_holdings1, benchmark

    def retrieveBenchmark_gk(self, data=''):
        '''
        获取聚源国开收益率曲线数据\n
        :param data: DataFrame, 全量债券持仓表, 用于获取债券代码清单
        :return:
            (1) bond_holdings1, DataFrame, 含债券ytm及对应国开收益率曲线的债券持仓表;
            (2) benchamrk: DataFrame, 聚源国开的收益率曲线数据
        '''
        if type(data) == str:
            bond_holdings = self.bond_holdings.copy()
        else:
            bond_holdings = data.copy()

        dq = JYDB_Query()

        date_list = bond_holdings['D_DATE'].unique()
        benchmark = pd.DataFrame(columns=['ENDDATE', 'YEARSTOMATURITY', 'YIELD'])
        for temp_date in date_list:
            temp_date = pd.to_datetime(temp_date).strftime('%Y-%m-%d')
            temp_b = dq.sec_query('curve_yield_all', '195', temp_date)    # 步长只有0.1, 195为国开
            benchmark = benchmark.append(temp_b, sort=False)
        benchmark['YIELD'] = benchmark['YIELD']*100
        benchmark.columns = ['ENDDATE_gk', 'YEARSTOMATURITY_gk', 'benchmark_gk']

        bond_holdings['剩余期限'] = [round(x, 1) for x in bond_holdings['PTMYEAR'].fillna(0)]    # 将取不出剩余期限的PTM取为0
        bond_holdings1 = pd.merge(bond_holdings, benchmark, left_on=['D_DATE', '剩余期限'], right_on=['ENDDATE_gk', 'YEARSTOMATURITY_gk'], how='left')
        bond_holdings1['spread_gk'] = bond_holdings1['YIELD_CNBD'] - bond_holdings1['benchmark_gk']

        return bond_holdings1, benchmark

    # 获取聚源对应隐含评级、对应剩余期限的收益率
    def retrieveBenchmark_Rating(self, data=''):
        '''
        获取聚源不同评级对应的收益率曲线数据\n
        :param data: DataFrame, 全量债券持仓表, 用于获取债券代码清单
        :return:
            (1) bond_holdings1, DataFrame, 含债券ytm及对应评级收益率曲线的债券持仓表;
            (2) benchamrk: DataFrame, 聚源国开不同评级的收益率曲线数据
        '''
        if type(data) == str:
            bond_holdings = self.bond_holdings.copy()
        else:
            bond_holdings = data.copy()
        
        dq = JYDB_Query()

        startDate = bond_holdings['D_DATE'].min().strftime('%Y-%m-%d')
        endDate = bond_holdings['D_DATE'].max().strftime('%Y-%m-%d')
        curve_list = bond_holdings['曲线代码'].dropna().drop_duplicates().astype(str).tolist()
        benchmark = dq.sec_query('curve_yield_interval', curve_list, startDate, endDate)
        benchmark['YIELD'] = benchmark['YIELD']*100

        bond_holdings['剩余期限'] = [round(x, 1) for x in bond_holdings['PTMYEAR'].fillna(0)]
        bond_holdings1 = pd.merge(bond_holdings.drop(columns=['ENDDATE', 'YEARSTOMATURITY']), benchmark, left_on=['D_DATE', '曲线代码', '剩余期限'], right_on=['ENDDATE', 'CURVECODE', 'YEARSTOMATURITY'], how='left')
        bond_holdings1 = bond_holdings1.rename(columns={'YIELD': 'benchmark_rating'})
        bond_holdings1['spread_rating'] = bond_holdings1['YIELD_CNBD'] - bond_holdings1['benchmark_rating']

        return bond_holdings1, benchmark   

    def retrieveBenchmark_ind(self, data=''):
        '''
        获取wind的行业利差数据(来源：兴业研究)\n
        :param data: DataFrame, 全量债券持仓表, 用于获取债券代码清单
        :return:
            (1) bond_holdings1, DataFrame, 含债券ytm及对应利差的债券持仓表;
            (2) benchamrk: DataFrame, wind行业利差数据
        '''
        if type(data) == str:
            bond_holdings = self.bond_holdings.copy()
        else:
            bond_holdings = data.copy()
            
        startDate = w.tdaysoffset(-1, bond_holdings['D_DATE'].min(), "").Data[0][0].strftime('%Y-%m-%d')
        endDate = w.tdaysoffset(-1, bond_holdings['D_DATE'].max(), "").Data[0][0].strftime('%Y-%m-%d')
        curve_list = bond_holdings['行业利差ID'].dropna().drop_duplicates().tolist()

        wind_temp = w.wsd(curve_list, "close", startDate, endDate, "")
        data_wind = pd.DataFrame(wind_temp.Data, columns=wind_temp.Times, index=wind_temp.Codes).T
        data_wind = data_wind.stack().reset_index()
        data_wind.columns = ['D_DATE', '行业利差ID', '行业利差']
        data_wind['D_DATE'] = bond_holdings['D_DATE'].min()
        data_wind['行业利差'] = data_wind['行业利差'] / 100

        bond_holdings1 = pd.merge(bond_holdings, data_wind, on=['D_DATE', '行业利差ID'], how='left')

        return bond_holdings1, data_wind

    def HoldingsCleaning(self, data_val='', ):
        '''
        整合后的全量证券持仓清洗\n
        :param data_val: DataFrame, 原始估值表, 默认为基期
        :return: 默认无返回值, 结果直接计入self.bond_holdings & self.stock_holdings。
                若给定某个估值表，则返回两个DataFrame, 分别为清洗后的债券持仓bond_holdings和股票持仓stock_holdings。
        '''
        status = type(data_val) == str   # 若从data_val开始均未给参数，则视为默认初始化清洗；若给定某个val，则该函数有返回值
        if status:
            self.holdings = self._formatHolding()   # 持仓明细(基于self.val，含筛选项）
            self.data_fund = self.getBasicData()    # 大类持仓(基于self.val，含筛选项）
            self.data_wind = self.__retrieveWindData()
            self.data_jy = self.__retrieveJYData()
            self.data_jy_equity = self.__retrieveJYData_equity()
            
            # 清洗债券及股票持仓数据
            self.__formatBondHoldings()
            self.__formatStockHoldings()
            self._dealRepurchaseDate()                                 # 用JY的回售日字段替换wind的回售日，因为JY可能更齐全
            logger.info('%s - Holdings Clean Done.' % self.basedate)

        else:
            holdings = self._formatHolding(data_val)
            data_wind = self.__retrieveWindData(holdings)
            data_jy = self.__retrieveJYData(holdings)
            data_indice = self.__retrieveJYData_equity(holdings)

            bond_holdings2, benchmark, benchmark_gk, benchmark_rating, benchmark_ind = self.__formatBondHoldings(holdings, data_wind, data_jy)
            stock_holdings = self.__formatStockHoldings(holdings, data_indice)

            return bond_holdings2, stock_holdings

    def RepoCleaning(self):
        '''清洗各组合杠杆成本数据，并落入数据库dpe_repoall、dpe_levcost表'''
        repo_all = self._deal_repo()

        lev_cols = ['c_fundname_o32', 'Lev_cost', 'Lev_amt', 'c_fundname', 'd_date']
        col_dict = {'d_date': 'D_DATE', 'c_fundname': 'C_FUNDNAME'}
        if repo_all.empty:
            self._lev_all = pd.DataFrame(columns=lev_cols).rename(columns=col_dict)
            return None

        lev_cost = repo_all.groupby(['c_fundname', 'd_date', 'portfolio_code']).apply(
            lambda x: (x['amount'] * x['direction_num'] * x['interest_rate'] / 100).sum() / (x['amount'] * x['direction_num']).sum()).rename('Lev_cost').reset_index()
        repo_amt = repo_all.groupby(['c_fundname', 'd_date', 'portfolio_code']).apply(
            lambda x: (x['amount'] * x['direction_num'] * 100).sum()).rename('Lev_amt').reset_index()
        lev_all = pd.merge(lev_cost, repo_amt, on=['c_fundname', 'd_date', 'portfolio_code'], how='left')

        lev_all = lev_all.reindex(columns=lev_cols + ['portfolio_code'])
        self.insert2db_single('dpe_levcost', lev_all, 'quant', ptf_code=self.ptf_codes, code_colname='portfolio_code')

        self._lev_all = lev_all.reindex(columns=lev_cols).rename(columns=col_dict)
        self._lev_all['D_DATE'] = pd.to_datetime(self._lev_all['D_DATE'])

    def save_excel(self, folder_path, file_name, data_dict, key_col='C_FUNDNAME', update=True, cover=False):
        '''
        将中间数据存储至excel
        :param folder_path: 文件夹路径
        :param file_name: 文件名称
        :param data_dict: 待存储数据
        :param key_col: 更新文件时，用于识别待更新的数据
        :param update: 若文件已存在，是否对其进行更新
        :param cover: 若文件已存在，是否对其进行覆盖
        :return:
        '''
        file_path = os.path.join(folder_path, file_name)

        res_dict = {}
        # 若文件已存在，且不对其进行覆盖
        if os.path.exists(file_path) and not cover:
            # 如果选择不更新原文件，则直接结束
            if not update:
                return None

            # 如果
            exist_file = pd.read_excel(file_path, sheet_name=None, header=0, engine='openpyxl')
            for key, exist_df in exist_file.items():
                if data_dict[key].empty:  # 如果新数据未空，则全部保留原数据
                    res_dict[key] = exist_df.copy()
                else:                    # 如果新数据不为空，则保留原数据中与新数据不重复的部分，并与新数据合并
                    new_info = data_dict[key][key_col].unique().tolist()
                    keep_df = exist_df[~exist_df[key_col].isin(new_info)].copy()
                    res_dict[key] = pd.concat([keep_df, data_dict[key]], ignore_index=True)

        writer = pd.ExcelWriter(file_path)
        res_dict = data_dict if len(res_dict) == 0 else res_dict
        for key, value in res_dict.items():
            value.to_excel(writer, sheet_name=key, index=False)
        writer.save()

    def SaveCleaningData(self, folder_path):
        '''保存清洗好的数据'''
        self.check_path(folder_path)

        holding_dict = {'bond_holdings': self.bond_holdings, 'stock_holdings': self.stock_holdings, 'holdings': self.holdings}
        self.save_excel(folder_path, 'Holdings.xlsx', holding_dict, key_col='C_FUNDNAME', update=True)

        db_dict = {'data_jy': self.data_jy, 'data_wind': self.data_wind}
        self.save_excel(folder_path, 'data_db.xlsx', db_dict, key_col='code', update=True)

        self.save_excel(folder_path, 'data_fund.xlsx', {'Sheet1': self.data_fund}, key_col='C_FUNDNAME', update=True)
        self.save_excel(folder_path, '_lev_all.xlsx', {'Sheet1': self._lev_all}, key_col='C_FUNDNAME', update=True)
        self.save_excel(folder_path, '_repo_all.xlsx', {'Sheet1': self._repo_all}, key_col='C_FUNDNAME', update=True)

        bm_dict = {'benchmark': self._benchmark, 'benchmark_gk': self._benchmark_gk,
                   'benchmark_ind': self._benchmark_ind, 'benchmark_rating': self._benchmark_rating}
        self.save_excel(folder_path, 'data_benchmark.xlsx', bm_dict, update=False)
        self.save_excel(folder_path, 'yield_map.xlsx', {'Sheet1': self._yield_map}, update=False)

        if self.ptf_codes is not None:
            self.save_excel(folder_path, 'Holdings_t.xlsx', holding_dict, cover=True)
            self.save_excel(folder_path, 'data_db_t.xlsx', db_dict, cover=True)
            self.save_excel(folder_path, 'data_fund_t.xlsx', {'Sheet1': self.data_fund}, cover=True)
            self.save_excel(folder_path, '_lev_all_t.xlsx', {'Sheet1': self._lev_all}, cover=True)
            self.save_excel(folder_path, '_repo_all_t.xlsx', {'Sheet1': self._repo_all}, cover=True)

        logger.info('%s - Holding data Saved.' % self.basedate)

    def calcCoV(self, x):
        '''计算给定序列的变异系数CoV(abs(std/mean))'''
        return abs(x.std(ddof=0) / x.mean())

    def calcMAE_basic(self, x):
        '''计算标准化后的MAE(中位数绝对偏差/中位数)'''
        return np.median(np.abs(x - x.median())) / abs(x.median())

    def check_path(self, data_path):
        '''检查给定路径是否存在，若不存在则创建该路径'''
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            logger.info('输入路径不存在，已创建该路径。')
    
    def saveFile(self, save_path, file_name, data):
        self.check_path(save_path)
        writer = pd.ExcelWriter(os.path.join(save_path,file_name))
        data.to_excel(writer, sheet_name=file_name)
        writer.save()
    
    def IntegrateAll(self):
        basic_list = ['_masterData', 'holdings', 'bond_holdings', 'bond_holdings2', 'stock_holdings', 'val', '_spreadAD', '_yield_map', '_IndustryID', '_SectorMap', 'data_wind', 'data_jy', 'data_jy_equity', '_benchmark', '_benchmark_gk', '_benchmark_rating', '_benchmark_ind', 
        'data_fund', 'bond_asset', 'ind_temp', 'dist_ind', 'dist_type1', 'dist_type2', 'dist_mir', 'ind_cov_total', 'ind_mae_total', 'spread_total', 'spread_ind_total',
        'spread_diff_1', 'spread_diff_5', 'spread_diff_20', 'spread_diff_60', 'spread_diff_120', 'spread_ind_diff_1', 'spread_ind_diff_5', 'spread_ind_diff_20', 'spread_ind_diff_60', 'spread_ind_diff_120', 
        'port_creditDown_t20', 'port_creditDown_t60', 'port_creditDown_t120', 'port_creditDown_ind_t20', 'port_creditDown_ind_t60', 'port_creditDown_ind_t120', 
        'asset_creditDown_t120', 'asset_creditDown_t20', 'asset_creditDown_t252', 'asset_creditDown_t60', 'asset_creditDown', 'asset_creditDown_holdings_ex',
        'data_prod', 'data_type', 'redeem_map', '_fundName', '_lev_all', '_repo_all', 'ptf_info', 'liq_holdings', 'ptf_codes']      # 最底层的数据不用保存出来
        res = pd.DataFrame(self.fund_list, columns=['C_FUNDNAME'])
        for (key, value) in vars(self).items():
            if type(value) != OracleDB:
                # print(key)
                if (key in basic_list) or (value is None):
                    continue
                if (type(value) == pd.core.frame.DataFrame) & (len(value) > 0):
                    # print(key, value.shape)

                    period_status = ('D_DATE' in value.columns) and ('D_DATE' in res.columns)    # 判断是否为多期数据，是则按日期和基金名称同时连接，否则按基金名称连接
                    if period_status:
                        res = pd.merge(res, value, on=['D_DATE', 'C_FUNDNAME'], how='outer')
                    else:
                        res = pd.merge(res, value, on='C_FUNDNAME', how='outer')

    #                print('Before:', key, value.shape, '\t', 'After:', res.shape)
        
        if 'C_FULLNAME' in res.columns:
            res = res.drop(columns='C_FULLNAME')

        if 'PORTFOLIO_CODE' not in res.columns:
            res['PORTFOLIO_CODE'] = res['C_FUNDNAME'].map(self.fundname_to_code)

        return res
    
    def IntegrateHoldings(self):
        res = pd.DataFrame(self.fund_list, columns=['C_FUNDNAME'])
        res = pd.merge(res, self.bond_holdings, on=['C_FUNDNAME'], how='left')

        holdings_list = [self.stock_holdings]                                            # 不同类型的资产采用直接向下填充
        for x in holdings_list:
            res = res.append(x, sort=False)

        for (key, value) in vars(self).items():            
            if ('_holdings_ex' in key) and (type(value) == pd.core.frame.DataFrame) and (len(value) > 0):     # 用变量名_holdings_ex来识别新增的资产层属性数据
                period_status = ('D_DATE' in value.columns) and ('D_DATE' in res.columns)    # 判断是否为多期数据，是则按日期和基金名称同时连接，否则按基金名称连接
                fund_status = 'C_FUNDNAME' in value.columns
                if period_status and fund_status:
                    res = pd.merge(res, value, on=['D_DATE', 'C_FUNDNAME', 'code'], how='left')
                elif period_status and not fund_status:
                    res = pd.merge(res, value, on=['D_DATE', 'code'], how='left')
                elif not period_status and fund_status:
                    res = pd.merge(res, value, on=['C_FUNDNAME', 'code'], how='left')
                else:
                    res = pd.merge(res, value, on=['code'], how='left')

                # print('Before:', key, value.shape, '\t', 'After:', res.shape)
        
        res = res.sort_values(by=['C_FUNDNAME', 'D_DATE'])
        
        return res

    def credit_rating_map(self):
        '''根据DPE_CREDIT_RATING_MAP表，得到债券评级的等级映射'''
        df = self.db_risk.read_sql(sqls_config['credit_rating_map']['Sql'])
        cr_map = dict()
        cr_map['rating'] = df.set_index('rating_bond')['rating_class'].to_dict()
        cr_map['mir_rating'] = df.set_index('rating_bond')['mir_rating_class'].to_dict()
        cr_map['inner_rating'] = df.set_index('rating_bond')['inner_rating_class'].to_dict()
        return cr_map

    def classify_credit_bond_info(self, t=""):
        t = self.basedate if t == "" else t
        q = sqls_config['dpe_portfoliobond']['Sql'] % t
        check_res = self.db_risk.read_sql(q)
        if check_res.empty:
            logger.error('dpe_portfoliobond表无当日数据，请检查。')
            return None

        q = sqls_config['classify_creditbond_info']['Sql']
        data = self.db_risk.read_sql(q.format(t=t))  # 次级债成为一级分类

        # 补充当天卖出的债券
        t0 = self.get_offset_tradeday(t, -1)
        data_t0 = self.db_risk.read_sql(q.format(t=t0))
        sell_bonds = data_t0[~data_t0['bond_code'].isin(data['bond_code'].to_list())].copy()
        data = pd.concat([data, sell_bonds], ignore_index=True)

        # 评级等级映射
        cr_map = self.credit_rating_map()
        data['rating_cls'] = data['rating_bond'].fillna('无评级').map(cr_map['rating']).fillna('低等级')
        data['mir_rating_cls'] = data['mir_rating_b'].fillna('无评级').map(cr_map['mir_rating']).fillna('低等级')
        data['inner_rating_cls'] = data['inner_rating_b'].fillna(value=data['inner_rating_i']).fillna('无评级').map(cr_map['inner_rating']).fillna('低等级')

        data['inner_industry'] = data[['industry_level2', 'city_level']].replace('城投', '城投-').replace('国资经营', '国资经营-').fillna('').sum(axis=1)
        data.loc[data['bond_style'] == 'CD', 'inner_industry'] = 'CD'
        data['ind_level2'] = data['industry_level2'].copy()
        data.loc[data['industry_level2'] == '城投', 'ind_level2'] = data.loc[data['industry_level2'] == '城投', 'level1'].copy()
        data.loc[data['bond_style'].isin(['CD', '金融债']), 'ind_level2'] = data.loc[data['bond_style'].isin(['CD', '金融债']), 'issuer_type'].copy()
        data['ind_level2'] = data['ind_level2'].fillna('-')
        data['bond_style'] = data['bond_style'].str.replace('CD', '同业存单')
        data['c_date'] = t

        if self.ptf_codes is not None:
            check_res['portfolio_code'] = check_res['c_fundname'].map(self.fundname_to_code)
            codes = check_res[check_res['portfolio_code'].isin(self.ptf_codes)]['code'].to_list()
            data = data[data['bond_code'].isin(codes)].copy()
            if data.empty:
                logger.info('-- dpe_creditbond_info: 无信用债持仓，无需修改数表。')
                return None
            self.insert2db_single('dpe_creditbond_info', data, t=t, t_colname='c_date', ptf_code=codes, code_colname='bond_code')

        self.insert2db_single('dpe_creditbond_info', data, 'quant', t=t, t_colname='c_date')

    # 获取该对象下的所有函数，所有变量可以通过遍历vars(myClass)得到
    def getMethods(self):
        '''获取该对象下的所有函数名称清单'''
        return list(filter(lambda x: not x.startswith('_') and callable(getattr(self, x)), dir(self)))
    
    # 时间对齐，根据D_DATE字段，以传入的x为基准
    def timeAlign(self, x, y):
        '''
        两组序列的时间对齐\n
        :param x: series, 若无时间标签d_date则报错
        :param y: series, 若无时间标签d_date则报错
        :return: (series, series), 时间对齐后的series x & series y
        '''
        if 'D_DATE' not in x.reset_index().columns or 'D_DATE' not in y.reset_index().columns:
            print( 'Error: No Time label in given variables!')
        
        new_x = x.reset_index().dropna(how='any')
        new_y = y.reset_index().dropna(how='any')
        x_date = new_x['D_DATE'].drop_duplicates().tolist()
        y_date = new_y['D_DATE'].drop_duplicates().tolist()
        idx_list = set(x_date) & set(y_date)
        
        new_x = new_x.loc[new_x['D_DATE'].isin(idx_list), :].set_index('D_DATE').sort_index()
        new_y = new_y.loc[new_y['D_DATE'].isin(idx_list), :].set_index('D_DATE').sort_index()
        
        if 'index' in new_x.columns:
            new_x = new_x.drop(columns=['index'])
        if 'index' in new_y.columns:
            new_y = new_y.drop(columns=['index'])

        return new_x, new_y

    def insert2db_single(self, table, data, schema='quant', t='', ptf_code=None,
                         t_colname='D_DATE', code_colname='PORTFOLIO_CODE'):
        self.delete_table(table, schema=schema, t=t, ptf_codes=ptf_code, column_name=t_colname, code_colname=code_colname)
        self.insert_table(table, data,  schema=schema, t=t)

    def insert_table(self, table, data, schema='quant', t='', if_exists='append'):
        t = self.basedate if t == '' else t
        if data.shape[0] == 0:
            return
        if 'db_risk' not in dir(self):
            self._connectingRiskDB()
        if 'D_DATE' in data.columns and type(data['D_DATE'].iloc[0]) == pd.Timestamp:
            data['D_DATE'] = [x.strftime('%Y-%m-%d') for x in data['D_DATE']]
        data['insert_time'] = datetime.datetime.now()
        self.db_risk.insert_dataframe(table=table.lower(), data=data, schema=schema, if_exists=if_exists)
        logger.info('%s数据插入成功，table: %s inserted to database.' % (t, table))

    def delete_table(self, table, schema='quant', t='', column_name='D_DATE', ptf_codes=None, code_colname='PORTFOLIO_CODE'):
        t = self.basedate if t == '' else t
        ptf_codes = self.ptf_codes if ptf_codes is None else ptf_codes

        if 'db_risk' not in dir(self):
            self._connectingRiskDB()
        if ptf_codes is not None:
            self._delete_table_in(table, schema, t=t, ptf_codes=ptf_codes, t_colname=column_name, code_colname=code_colname)
            return None

        condition = column(column_name) == t
        try:
            self.db_risk.delete(table.lower(), condition, schema)
            logger.info('%s删除成功，table: %s deleted from database.' % (t, table))
        except (DoesNotExist, exc.NoSuchTableError):
            logger.warning('%s删除失败，table: %s data not found.' % (t, table))
            pass

    def _delete_table_in(self, table, schema, t, ptf_codes, t_colname='D_DATE', code_colname='PORTFOLIO_CODE'):
        condition = and_(column(t_colname) == t, column(code_colname).in_(ptf_codes))
        cond_str = '%s = %s, %s in (%s)' % (t_colname, t, code_colname, ', '.join(ptf_codes))

        try:
            self.db_risk.delete(table.lower(), condition, schema)
            logger.info('删除成功，table: %s - %s deleted from database.' % (table, cond_str))
        except (DoesNotExist, exc.NoSuchTableError):
            logger.warning('删除失败，table: %s - %s not found.' % (table, cond_str))

    def load_sheet_columns(self, file_name, sheet_name=None):
        '''
        加载管理层风险指标的数据字典表，以方便插入数据库\n
        :return:
        '''
        dict_cols = pd.read_excel(os.path.join(DIR_OF_MAIN_PROG, 'data', 'SheetColumns_%s.xlsx'%file_name),
                                  sheet_name=sheet_name, engine='openpyxl')
        return dict_cols

    def insert2db(self, rc_type, tablename, sheetname, data0, ptf_codes=None):
        '''
        插入数据库，进行统一的列名转换\n
        :param rc_type: string, 指标类型，用于对应不同的SheetColumns，如RiskIndicators、RiskIndicators_mg
        :param tablename: string, 插入数据库的表名，如rc_credit
        :param sheetname: string, 列名map文件(SheetColumns.xlsx)中对应的sheet页名，如Concentration
        :param data0: DataFrame, 需要插入数据库的结果文件
        :return: None
        '''
        data = data0.copy()
        if data.empty:
            logger.info('%s -- %s 无新增数据' % (self.basedate, tablename))
            return None
        dict_sheet = self.load_sheet_columns(rc_type, sheetname)
        data.columns = dict_sheet['Sheet_columns']
        data = data.dropna(subset=['D_DATE'], how='any')
        data = data.replace(np.inf, np.nan).replace((-1) * np.inf, np.nan)

        ptf_codes = self.ptf_codes if ptf_codes is None else ptf_codes
        self.insert2db_single(tablename, data, ptf_code=ptf_codes, code_colname='PORTFOLIO_CODE')

    def general_delete_in(self, table, schema='quant', **kwargs):
        condition = and_(and_(column(key) == kwargs[key] if type(kwargs[key]) != list
                              else column(key).in_(kwargs[key]) for key in kwargs))
        operator_dict = {key: " in " if type(kwargs[key]) == list else " = " for key in kwargs}

        try:
            self.db_risk.delete(table.lower(), condition, schema)
            logger.info('删除成功，table: %s - %s deleted from database.' % (
                table, (", ".join([key + operator_dict[key] + str(kwargs[key]) for key in kwargs]))))
        except (DoesNotExist, exc.NoSuchTableError):
            logger.warning('删除失败，table: %s - %s not found.' % (
                table, (", ".join([key + operator_dict[key] + str(kwargs[key]) for key in kwargs]))))

    def add_portfolio_code(self, table, t=''):
        t = self.basedate if t == '' else t
        q = "select * from {table} where d_date = '{t}'"
        data = self.db_risk.read_sql(q.format(table=table, t=t))
        data['portfolio_code'] = data['c_fundname'].map(self.fundname_to_code)
        self.insert2db_single(table=table, data=data, t=t)

    # 已停用函数
    # def getDerivatives(self):
    #     '''获取衍生品全量持仓信息, 依据估值表的l_stocktype为6来识别'''
    #     data = self.val[(self.val['C_STOPINFO'] != '') & (self.val['L_STOCKTYPE'] == '6')].copy()
    #     data['F_ASSET'] = data['F_ASSET'].astype(float)
    #     res = data.groupby(['C_FUNDNAME', 'D_DATE'])['F_ASSET'].sum().rename('衍生品')
    #     return res

    # def getBondAsset(self):
    #     '''获取债券资产持仓总市值'''
    #     data = self.val.copy()
    #     bond_asset = data.loc[data['C_SUBCODE'] == '其中债券投资:', ['D_DATE', 'C_FUNDNAME', 'C_SUBCODE', 'F_ASSET']].rename(columns={'F_ASSET': 'TotalBond'}).drop(columns='C_SUBCODE')   # 债券投资总资产
    #
    #     return bond_asset