# -*- coding: utf-8 -*-
'''
# @Desc    : 个性化代码模块
# @Author  : zhouyy
# @Time    : 2021/5/10
'''
import datetime
import pandas as pd
import numpy as np
import os
from .utils.log_utils import logger
from .settings import config, DIR_OF_MAIN_PROG
from scripts.db import OracleDB, column, sqls_config, and_, column
from .db import MySQLDB
from .db.util import DoesNotExist
from sqlalchemy import exc
from scripts.db.db_utils import db_quant, convert_columns

def test_read_data():
    """
    从data文件夹取数
    :return:
    """
    print("*"*30, '测试data读取', "*"*30)
    file_name = 'CreditTransitionMatrix.xlsx'
    df = pd.read_excel(os.path.join(DIR_OF_MAIN_PROG, 'data', file_name), engine='openpyxl')
    print(df.head)


def test_database():
    """
    测试数据库
    :return:
    """
    # 测试读取(JY)
    logger.info('测试读取数据库：')
    db_jy = OracleDB(config['data_base']['JY']['url'])
    ssql_calendar = sqls_config['ashare_calendar']['Sql']
    ashare_calendar = db_jy.read_sql(sql=ssql_calendar)
    print(ashare_calendar.head(5))

    # 测试读取(RISK)
    logger.info('测试读取数据库：')
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    ssql =  sqls_config['demo_set_sql']['Sql']
    data = db_risk.read_sql(ssql)
    data.columns = list(map(lambda x: x.upper(), data.columns))
    print(data.head(5))

    # 测试插入
    table = "ef_base_tx_peer"
    logger.info(f"测试插入数据库, 表名：{table}")
    db_risk.insert_dataframe(table=table, data=data, schema='quant')

    # 测试删除
    logger.info(f"测试删除数据库， 表名：{table}, 删除表中AVGPRICE < 100 的数据...")
    condition = column("AVGPRICE") < 100
    db_risk.delete(table="ef_base_tx_peer", condition=condition, schema="risk")

    # 更多数据库功能请见 /code/scripts/db/README.md ...
    pass


def test_mysql():
    logger.info('测试读取mysql数据库：')
    db_mysql = MySQLDB(config['data_base']['Mysql_test']['url'])
    ssql =  sqls_config['demo_testmysql']['Sql']
    df = db_mysql.read_sql(ssql)
    print(df.head(5))


def test_log_utils():
    """
    测试日志工具
    :return:
    """
    print("*" * 30, '测试日志模块', "*" * 30)
    logger.debug("---测试开始----")
    logger.info("操作步骤")
    logger.warning("----测试结束----")
    logger.error("----测试错误----")


def run_my_function():
    test_read_data()
    test_database()
    test_log_utils()


def insert_table(table, data, schema, if_exists='append'):
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    db_risk.insert_dataframe(table=table.lower(), data=data, schema=schema, if_exists=if_exists)


def delete_table(table, t, schema):
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    condition = column('D_DATE') == t
    try:
        db_risk.delete(table.lower(), condition, schema)
        print('%s deleted.' % table)
    except (DoesNotExist, exc.NoSuchTableError):
        # print('%s does not exist.'%t)
        pass


def insert_by_sheet(data, sheet_pairs, schema, t=''):
    for sheet_name, table_name in sheet_pairs:
        if 'C_FUNDNAME' in data.get(sheet_name).columns:
            data_temp = data.get(sheet_name).dropna(subset=['C_FUNDNAME', 'D_DATE'], how='any')
        else:
            data_temp = data.get(sheet_name).dropna(subset=['D_DATE'], how='any')

        data_temp['D_DATE'] = [x.strftime('%Y-%m-%d') for x in data_temp['D_DATE']]
        data_temp = data_temp.replace(np.inf, np.nan).replace((-1)*np.inf, np.nan)
        if t != '':
            delete_table(table_name, t, schema)
        insert_table(table_name, data_temp, schema)
        print('%s inserted.'%sheet_name)


def insert_t_indicator(data_path, file_name, t, sheet_pairs, schema):
    '''将t日的风险指标文件写入数据库'''
    dict_sheet = load_sheet_columns(file_name)
    ri_t = pd.read_excel(data_path + '%s_%s.xlsx' % (t.replace('-', ''), file_name), sheet_name=None, engine='openpyxl')
    for sheet in dict_sheet.keys():
        if sheet == 'Market_return_bch':
            continue
        if 'D_DATE' not in ri_t[sheet].columns:
            ri_t[sheet]['D_DATE'] = pd.to_datetime(t)
        if sheet == '变现个券明细':
            ri_t[sheet] = ri_t[sheet].reindex(columns=['C_FUNDNAME', 'D_DATE', 'code', 'C_SUBNAME_BSH', '变现天数'])
        ri_t[sheet].columns = dict_sheet[sheet]['Sheet_columns']
    print('=='*3, t, '=='*3)
    insert_by_sheet(ri_t, sheet_pairs, schema, t)


def get_sheet_columns(data, writer):
    '''保存风险指标列的英文映射关系'''
    dict_cols = {}
    for sheet in data.keys():
        col = data[sheet].columns
        dict_cols[sheet] = col
        pd.DataFrame(col, columns=['Sheet_columns']).to_excel(writer, sheet_name=sheet, index=False)
    writer.save()
    print('Columns saved.')


def load_sheet_columns(file_name):
    '''加载风险指标列的英文映射关系'''
    dict_cols = pd.read_excel(os.path.join(DIR_OF_MAIN_PROG, 'data', 'SheetColumns_%s.xlsx'%file_name), sheet_name=None, engine='openpyxl')
    return dict_cols


def delete_t_sheet(sheets, t, schema):
    '''删除t日的表格们'''
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    condition = column('D_DATE') == t
    for table in sheets:
        db_risk.delete(table.lower(), condition, schema)
        print('%s deleted.'%table)


def insert_coef(folder_path, coef_type, t=''):
    '''更新流动性模型系数，若t不给定日期，则默认全部重新插入'''
    coef_dict = {'CD': ['RATE_LATESTMIR_CNBD', 'ptmyear_type', 'bank_type', 'coef', 'D_DATE'],
                        'IRBond': ['issuer_type', 'ptmyear_type', 'activity', 'coef', 'D_DATE'],
                        'ActivityBond': ['row_num', 'code', 'sec_name', 'volume', 'turnover', 'volume_days', 'liquidity', 'ytm', 'net_price', 'dirty_price', 'duration', 'duration_modi', 'convexity', 'ptmyear', 'issuer', 'issue_repeat', 'ptmyear_type', 'issue_addition', 'activity_rank', 'activity_cut', 'avtivity', 'D_DATE']}
    tablename_dict = {'CD': 'RC_LR_FACTOR_CD', 'IRBond': 'RC_LR_FACTOR_IR', 'ActivityBond': 'RC_LR_FACTOR_IR_DATA'}

    file_type = 'coefs' if coef_type in ['CD', 'IRBond'] else '活跃券划分'
    folder_path = folder_path if coef_type in ['CD', 'IRBond'] else folder_path.replace(coef_type, 'IRBond')

    if t == '':
        folders = os.listdir(folder_path)
    else:
        folders = [t.replace('-', '')]

    print('===='*3, coef_type, '===='*3)
    for folder in folders:
        if '.' not in folder:
            coef = pd.read_excel(folder_path + '\\%s\\%s.xlsx'%(folder, file_type), engine='openpyxl')
            if 'D_DATE' not in coef.columns:
                coef['D_DATE'] = pd.to_datetime(folder).strftime('%Y-%m-%d')
            if 'Unnamed: 0' in coef.columns:
                coef = coef.drop(columns='Unnamed: 0')
            if '债券余额（亿）' in coef.columns:
                coef = coef.drop(columns='债券余额（亿）')
            coef.columns = coef_dict[coef_type]
            coef = _format_string(coef)
            tablename = tablename_dict[coef_type]
            coef['insert_time'] = datetime.datetime.now()
            delete_table(tablename, pd.to_datetime(folder).strftime('%Y-%m-%d'), 'quant')
            insert_table(tablename, coef, 'quant')
            print(folder, 'inserted.')


def insert_coef_cr(folder_path, t=''):
    dict_cols = pd.read_excel(os.path.join(DIR_OF_MAIN_PROG, 'data', 'SheetColumns_CR_Factor.xlsx'), sheet_name=None, engine='openpyxl')
    if t == '':
        folders = os.listdir(folder_path)
    else:
        folders = [t.replace('-', '')]

    sheet_names = ['评级系数', '剩余期限系数', '行业类别系数', '偿还顺序系数', '企业属性系数', '含权条款系数', '发行方式系数']
    table_names = ['RC_LR_FACTOR_RATING', 'RC_LR_FACTOR_PTM', 'RC_LR_FACTOR_IND', 'RC_LR_FACTOR_PAY', 'RC_LR_FACTOR_ISSUER', 'RC_LR_FACTOR_OPTION', 'RC_LR_FACTOR_METHOD']
    sheet_pairs = [(x, y) for x, y in zip(sheet_names, table_names)]

    for folder in folders:
        if '.' not in folder:
            coef = pd.read_excel(folder_path + '\\%s\\coefs.xlsx'%folder, sheet_name=None, engine='openpyxl')
            for sheet in dict_cols.keys():
                if 'D_DATE' not in coef[sheet].columns:
                    coef[sheet]['D_DATE'] = pd.to_datetime(folder)
                coef[sheet].columns = dict_cols[sheet]['Sheet_columns']
                coef[sheet]['insert_time'] = datetime.datetime.now()

            print('==' * 3, t, '==' * 3)
            insert_by_sheet(coef, sheet_pairs, 'quant', t)


def insert_DPE(folder_path, file_type, t='', sec_type='', ptf_codes=None):
    '''更新DPE，若t不给定日期，则默认全部重新插入'''
    dict_cols = pd.read_excel(os.path.join(DIR_OF_MAIN_PROG, 'data', 'SheetColumns_DPE.xlsx'), sheet_name=None, engine='openpyxl')
    tablename_dict = {'_lev_all': 'DPE_LEVCOST', '_repo_all': 'DPE_REPOALL', 'data_fund': 'DPE_VALASSET', 'bond_holdings': 'DPE_PORTFOLIOBOND', 'stock_holdings': 'DPE_PORTFOLIOSTOCK2', 'holdings':'DPE_PORTFOLIOSECS', 'liq_holdings': 'dpe_lr_holding'}

    if t == '':
        folders = os.listdir(folder_path)
    else:
        folders = [t.replace('-', '')]

    print('===='*3, file_type, '===='*3)
    for folder in folders:
        if '.' not in folder:
            file_path = folder_path + '\\%s\\%s.xlsx'%(folder, file_type)
            if os.path.exists(file_path):
                sht = 0 if sec_type == '' else sec_type.replace('all', '') + file_type.lower()
                coef = pd.read_excel(file_path, sheet_name=sht, engine='openpyxl')
                if 'D_DATE' not in coef.columns:
                    coef['D_DATE'] = pd.to_datetime(folder).strftime('%Y-%m-%d')
                else:
                    coef = coef.dropna(subset=['D_DATE'])
                    coef['D_DATE'] = [x.strftime('%Y-%m-%d') for x in coef['D_DATE']]
                if 'Unnamed: 0' in coef.columns:
                    coef = coef.drop(columns='Unnamed: 0')
                if file_type == 'data_fund':
                    coef = coef.reindex(columns=['C_FUNDNAME', '基金类型', 'D_DATE', 'C_FULLNAME', 'NAV', 'NAV_累计', 'NetAsset', 'TotalAsset', '基金份额', 'Deposit', '股票', '债券', 'ABS', '基金投资', '买入返售', '卖出回购', '衍生品', '可转债'])
                elif file_type == 'Holdings' and sec_type == 'stock_':
                    coef = coef.reindex(columns=['D_DATE', 'L_SETCODE', 'C_FULLNAME', 'C_FUNDNAME', 'C_SUBNAME_BSH', 'C_STOPINFO', 'F_MOUNT', 'F_PRICE', 'F_ASSET', 'F_ASSETRATIO', 'F_NETCOST', 'F_COST', 'F_COSTRATIO', 'L_STOCKTYPE', 'L_FUNDKIND', 'L_FUNDKIND2', 'L_FUNDTYPE', 'code', 'LISTEDSECTOR', 'TOTALMV', 'PE_TTM', 'PB', '上市板块'])
                elif file_type == 'Liq_holdings':
                    coef = coef.reindex(columns=['D_DATE', 'C_FULLNAME', 'C_FUNDNAME', 'C_SUBNAME_BSH', 'C_STOPINFO', 'code', 'F_MOUNT', 'F_ASSET', 'F_ASSETRATIO', '1日可变现_张', '5日可变现_张', '1日可变现', '5日可变现', '质押量_张', '质押市值', '受限数量', '变现天数', '是否高流动性债券'])
                if file_type == 'Holdings' and sec_type == 'all':
                    pass
                elif file_type == 'Holdings':
                    new_cols = dict_cols[sec_type.replace('all', '') + file_type.lower()].set_index('Sheet_columns_orig')['Sheet_columns'].to_dict()
                    coef.columns = [new_cols[x] for x in coef.columns]
                else:
                    coef.columns = dict_cols[sec_type.replace('all', '') + file_type.lower()]['Sheet_columns']
                coef = _format_string(coef)
                tablename = tablename_dict[sec_type.replace('all', '') + file_type.lower()]
                delete_table(tablename, pd.to_datetime(folder).strftime('%Y-%m-%d'), 'quant')
                insert_table(tablename, coef, 'quant')
                print(folder, 'inserted.')
            else:
                print('! %s no file %s.'%(folder, file_type))


def insert_tmpdata(file_path, file_name, schema):
    dict_cols = pd.read_excel(os.path.join(DIR_OF_MAIN_PROG, 'data', 'SheetColumns_%s.xlsx'%file_name), sheet_name=None, engine='openpyxl')
    data = pd.read_excel(file_path + '%s.xlsx'%file_name, sheet_name=None, engine='openpyxl')
    table_dict = {'benchmark':'DPE_YIELDCURVE_AAA', 'benchmark_gk': 'DPE_YIELDCURVE_CDB',
                  'benchmark_ind': 'DPE_YIELDSPREAD_IND', 'benchmark_rating': 'DPE_YIELDCURVE_RATING',
                  'data_jy': 'DPE_TMPDATA_JY', 'data_wind': 'DPE_TMPDATA_WIND'}

    for sheet in dict_cols.keys():
        data_temp = _format_string(data[sheet])
        data_temp.columns = dict_cols[sheet]['Sheet_columns']
        table_name = table_dict[sheet]
        insert_table(table_name, data_temp, schema, 'replace')
        print('%s replaced.'%table_name)


def _format_string(data):
    '''强制将object列的类型转换为string，防止往数据库写入时某些数据不规范导致报错'''
    for col in data.columns:
        if data[col].dtypes == object:
            data[col] = data[col].astype(str)

    return data


def _define_sheet_pairs(rc_type='RiskIndicators'):
    sheet_names = ['Basic', 'Credit', 'Liquidity', 'Liquidity_core']
    table_indicators = ['RC_STYLE', 'RC_CREDIT', 'RC_LIQUIDITY', 'RC_LR_CORE']
    pairs_indcators = [(x, y) for x, y in zip(sheet_names, table_indicators)]

    sheet_names = ['久期变化', '货基指标', '内部评级分布', '指数跟踪正偏离', '专户_错配', '专户_低流动性', '专户_收益率缺口']
    table_mg = ['RC_MGMT_DURACHG', 'RC_MGMT_MONETARYFUND', 'RC_MGMT_INNERRATING_DIST', 'RC_MGMT_INDEXFUND', 'RC_MGMT_OVERMTRT', 'RC_MGMT_LOWLIQSECS', 'RC_MGMT_RETURNGAP']
    pairs_mg = [(x, y) for x, y in zip(sheet_names, table_mg)]

    sheet_names = ['到期前变现', '变现个券明细', '非核心流动性1', '核心流动性_证券', '核心流动性_久期', '核心流动性_各级']
    table_core = ['RC_LR_CORE_MATURITY', 'RC_LR_CORE_OVERDUESECS', 'RC_LR_CORE_ASSET', 'RC_LR_CORE_SECS', 'RC_LR_CORE_DURACHG', 'RC_LR_CORE_ALL']
    pairs_core = [(x, y) for x, y in zip(sheet_names, table_core)]

    pairs_dict = {'RiskIndicators': pairs_indcators, 'RiskIndicators_mg': pairs_mg, 'LiquidityIndicators_core': pairs_core}
    table_dict = {'RiskIndicators': table_indicators, 'RiskIndicators_mg': table_mg, 'LiquidityIndicators_core': table_core}

    pairs_need = {rc_type: pairs_dict[rc_type]}
    table_need = {rc_type: table_dict[rc_type]}

    return pairs_need, table_need


def insert_indicators(data_path, t, rc_type, schema='quant'):
    '''批量插入当日指标，包括风险指标、管理层指标和流动性指标'''
    pairs_dict, table_dict = _define_sheet_pairs(rc_type)
    indicator_list = pairs_dict.keys()
    for indicator in indicator_list:
        insert_t_indicator(data_path, indicator, t, pairs_dict[indicator], schema)


def delete_indicators(t, schema='quant'):
    '''批量删除当日指标，包括风险指标、管理层指标和流动性指标'''
    table_dict, pairs_dict = _define_sheet_pairs()
    indicator_list = pairs_dict.keys()
    for indicator in indicator_list:
        delete_t_sheet(table_dict[indicator], t, schema)


def insert_liq_factor(t):
    '''批量更细流动性模型系数'''
    # CD、利率债、利率债活跃券
    coef_list = ['CD', 'IRBond', 'ActivityBond']
    for coef_type in coef_list:
        folder_path = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\LiquidityModel\%s\\'%coef_type
        insert_coef(folder_path, coef_type, t)

    # 信用债
    folder_cr = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\LiquidityModel\CreditBond\Coefficients\\'
    insert_coef_cr(folder_cr, t)


def insert_dpe_t(folder_path, t, ptf_codes=None):
    '''批量更新当日指标DPE的结果'''
    file_list = ['data_fund', 'Holdings', 'Holdings', 'Holdings']
    sec_type = ['', 'all', 'bond_', 'stock_']
    for file_type, sec_type in zip(file_list, sec_type):
        insert_DPE(folder_path, file_type, t, sec_type, ptf_codes)


def insert_temp_t(folder_path, t, file_list=None, schema='quant'):
    '''批量更新当日指标的temp data，数据库仅保留当日结果'''
    file_path = folder_path + t.replace('-', '') + '\\'
    file_list = ['data_benchmark', 'data_db'] if file_list is None else file_list
    for file_name in file_list:
        insert_tmpdata(file_path, file_name, schema)


def saveData(data_path, Indicators):
    '''
    保存文件\n
    :param data_path: string, 保存路径
    :param Indicators: class, python类，当日指标（如市场风险指标等）
    :return: None
    '''
    writer = pd.ExcelWriter(os.path.join(data_path, 'Holdings.xlsx'))
    Indicators.bond_holdings.to_excel(writer, sheet_name='bond_holdings', index=False)
    Indicators.stock_holdings.to_excel(writer, sheet_name='stock_holdings', index=False)
    Indicators.holdings.to_excel(writer, sheet_name='holdings', index=False)
    writer.save()

    writer = pd.ExcelWriter(os.path.join(data_path, 'data_db.xlsx'))
    Indicators.data_jy.to_excel(writer, sheet_name='data_jy', index=False)
    Indicators.data_wind.to_excel(writer, sheet_name='data_wind', index=False)
    writer.save()

    # Indicators.data_wind_equity.to_excel(os.path.join(data_path, 'data_wind_equity.xlsx'), index=False)

    writer = pd.ExcelWriter(os.path.join(data_path, 'data_benchmark.xlsx'))
    Indicators._benchmark.to_excel(writer, sheet_name='benchmark', index=False)
    Indicators._benchmark_gk.to_excel(writer, sheet_name='benchmark_gk', index=False)
    Indicators._benchmark_ind.to_excel(writer, sheet_name='benchmark_ind', index=False)
    Indicators._benchmark_rating.to_excel(writer, sheet_name='benchmark_rating', index=False)
    writer.save()

    Indicators._yield_map.to_excel(os.path.join(data_path, 'yield_map.xlsx'), index=False)
    Indicators.data_fund.to_excel(os.path.join(data_path, 'data_fund.xlsx'), index=False)

    Indicators._lev_all.to_excel(os.path.join(data_path, '_lev_all.xlsx'), index=False)
    Indicators._repo_all.to_excel(os.path.join(data_path ,'_repo_all.xlsx'), index=False)


def retrieve_n_tradeday(t, n):
    '''
    取给定日期过去第n个交易日日期
    :param t: string/datetime/timestamp, 需查询的给定日期
    :param n: int, 日期偏移量, 仅支持向历史偏移
    :return: string, 过去第n个交易日日期
    '''
    if type(t) != str:
        t = t.strftime('%Y-%m-%d')
    q = sqls_config['past_tradeday']['Sql']%t
    tradeday = db_quant.read_sql(q).sort_values(by=['c_date'])
    return tradeday.iloc[(-1)*(n+1)][0]


def check_path(folder_path):
    '''
    检查文件夹是否存在, 若不存在则创建该文件夹\n
    :param folder_path: string, 文件夹路径
    :return: None
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def tn_ptf_codes():
    tn_ptf = db_quant.read_sql(sqls_config['tn_portfolio']['Sql'])
    tn_ptf_dict = {}
    for i in range(tn_ptf['n_days'].min(), tn_ptf['n_days'].max() + 1):
        tn_ptf_dict[i] = tn_ptf[tn_ptf['n_days'] == i]['portfolio_code'].to_list()
    return tn_ptf_dict


def map_ptf_code_name(key='ptf_code'):
    '''产品全称与估值表简称对应表'''
    ptf_info = db_quant.read_sql(sqls_config['portfolio_type']['Sql'])
    code_to_name = ptf_info.set_index('c_fundcode')['c_fundname'].to_dict()
    name_to_code = ptf_info.set_index('c_fundname')['c_fundcode'].to_dict()
    return code_to_name if key == 'ptf_code' else name_to_code


def retrieve_time_series(save_path, startdate, enddate, file_name='RiskIndicators-v3.xlsx'):
    '''
    生成风险指标的时间序列数据，从数据库取数并保存至本地\n
    :param save_path: string, 保存的文件夹路径
    :param startdate: string, yyyy-mm-dd格式, 起始日期(含)
    :param enddate: string, yyyy-mm-dd格式, 结束日期(含)
    :param file_name: string, 保存的文件名及格式
    :return: None
    '''
    logger.info('%s -- 开始生成 %s' % (enddate, file_name))
    q = sqls_config['rc_series']['Sql']
    tn_ptf_dict = tn_ptf_codes()

    writer = pd.ExcelWriter(os.path.join(save_path, file_name), engine='openpyxl')
    sheets = ['Concentration', 'Basic', 'Market_return', 'Market_return_bch', 'Market_Holding', 'Credit', 'Liquidity',
              'Liquidity_core']
    tables = ['RC_CONCENTRATION', 'RC_STYLE', 'RC_MR_RETURN', 'RC_MR_RETURN_BCH', 'RC_MR_HOLDING', 'RC_CREDIT',
              'RC_LIQUIDITY', 'RC_LR_CORE']
    dict_cols = pd.read_excel(os.path.join(save_path, '20220805_RiskIndicators.xlsx'), sheet_name=None,
                              engine='openpyxl')
    for (sheet, table) in zip(sheets, tables):
        idx_temp = db_quant.read_sql(q % (table, startdate, enddate))

        # 非T日估值产品
        for n, ptf_codes in tn_ptf_dict.items():
            enddate_n = retrieve_n_tradeday(enddate, n)
            startdate_n = startdate if startdate < enddate_n else enddate_n
            idx_temp_n = db_quant.read_sql(q % (table, startdate_n, enddate_n))
            keep_df = idx_temp[~idx_temp['portfolio_code'].isin(ptf_codes)].copy()
            new_df = idx_temp_n[idx_temp_n['portfolio_code'].isin(ptf_codes)].copy()
            idx_temp = pd.concat([keep_df, new_df], ignore_index=True)

        idx_temp = convert_columns(idx_temp.rename(columns={'PAIN_INDEX': 'pain_index', 'HIT_RATE': 'hit_rate'}))
        idx_temp['D_DATE'] = pd.to_datetime(idx_temp['D_DATE'])
        idx_temp = idx_temp.sort_values(by=['D_DATE', 'C_FUNDNAME'])
        if 'insert_time'.upper() in idx_temp:
            idx_temp = idx_temp.drop(columns=['insert_time'.upper()])
        idx_temp.columns = dict_cols[sheet].columns
        idx_temp.to_excel(writer, sheet_name=sheet, index=False)
    writer.save()
    logger.info('%s -- 生成完毕' % enddate)


def retrieve_idc_repo(t):
    '''
    回购明细表(idc_repo)\n
    :param t: string, yyyy-mm-dd格式, 取数日期
    :return: None
    '''
    q = sqls_config['idc_repo_t']['Sql']
    data_repo = db_quant.read_sql(q % t)

    tn_ptf_dict = tn_ptf_codes()
    # 非T日估值产品
    for n, ptf_codes in tn_ptf_dict.items():
        t_n = retrieve_n_tradeday(t, n)
        data_n = db_quant.read_sql(q % t_n)
        keep_df = data_repo[~data_repo['portfolio_code'].isin(ptf_codes)].copy()
        new_df = data_n[data_n['portfolio_code'].isin(ptf_codes)].copy()
        data_repo = pd.concat([keep_df, new_df], ignore_index=True)

    data_repo = data_repo.loc[(data_repo['trade_spot'] == '银行间') & (data_repo['repo_dir'] == '融资回购') & (data_repo['actu_matu_date'] > t), :].copy()
    data_repo = data_repo.drop(columns=['src_insert_time'])
    path_collateral = config['shared_drive_data']['collateral_file']['path']
    data_repo.to_excel(os.path.join(path_collateral, t.replace('-', ''), 'idc_repo.xlsx'), index=False)


def retrieveBchIdxLastDay(t):
    '''
    比较基准的风险收益指标数据\n
    :param t: string, yyyy-mm-dd格式, 取数日期
    :return: DataFrame
    '''
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    q = sqls_config['rc_mr_return_bch_t']['Sql']
    t1 = retrieve_n_tradeday(t, 1)
    idx_t1 = convert_columns(db_risk.read_sql(q%t1)).rename(columns={'PAIN_INDEX': 'pain_index', 'HIT_RATE': 'hit_rate'}).drop(columns=['INSERT_TIME'])
    idx_t1['D_DATE'] = pd.to_datetime(t)
    return idx_t1


class InsertData():
    def __init__(self):
        self.db_quant = OracleDB(config['data_base']['QUANT']['url'])
        self._map_ptf_code_name()

        self.table_dict = {'data_fund': 'DPE_VALASSET', 'bond_holdings': 'DPE_PORTFOLIOBOND',
                           'stock_holdings': 'DPE_PORTFOLIOSTOCK2', 'holdings': 'DPE_PORTFOLIOSECS',
                           'benchmark': 'DPE_YIELDCURVE_AAA', 'benchmark_gk': 'DPE_YIELDCURVE_CDB',
                           'benchmark_ind': 'DPE_YIELDSPREAD_IND', 'benchmark_rating': 'DPE_YIELDCURVE_RATING',
                           'data_jy': 'DPE_TMPDATA_JY', 'data_wind': 'DPE_TMPDATA_WIND'}

    def _map_ptf_code_name(self):
        '''产品全称与估值表简称对应表'''
        ptf_info = self.db_quant.read_sql(sqls_config['portfolio_type']['Sql'])
        self.code_to_name = ptf_info.set_index('c_fundcode')['c_fundname'].to_dict()
        self.name_to_code = ptf_info.set_index('c_fundname')['c_fundcode'].to_dict()

    def insert_dpe_t(self, folder_path, t, ptf_codes=None):
        '''批量更新当日指标DPE的结果'''
        self.dict_dpe_cols = pd.read_excel(os.path.join(DIR_OF_MAIN_PROG, 'data', 'SheetColumns_DPE.xlsx'), sheet_name=None,
                                       engine='openpyxl')
        self.file_list = ['data_fund', 'Holdings', 'Holdings', 'Holdings']
        self.sec_type = ['', 'all', 'bond_', 'stock_']

        ptf_names = [self.code_to_name[i] for i in ptf_codes if i in self.code_to_name.keys()] if ptf_codes is not None else None

        for file_type, sec_type in zip(self.file_list, self.sec_type):
            add_code = True if file_type == 'Holdings' and sec_type == 'bond_' else False
            self.insert_DPE(folder_path, file_type, t, sec_type, ptf_names, add_code=add_code)

    def insert_DPE(self, folder_path, file_type, t='', sec_type='', ptf_names=None, add_code=False):
        '''更新DPE，若t不给定日期，则默认全部重新插入'''
        if t == '':
            folders = os.listdir(folder_path)
        else:
            folders = [t.replace('-', '')]

        logger.info('====' * 3 + file_type +  '====' * 3)
        for folder in folders:
            if '.' in folder:
                continue

            file_path = folder_path + '\\%s\\%s.xlsx' % (folder, file_type)
            if not os.path.exists(file_path):
                logger.warning('=== %s no file %s.' % (folder, file_type))
                continue

            sht = 0 if sec_type == '' else sec_type.replace('all', '') + file_type.lower()
            coef = pd.read_excel(file_path, sheet_name=sht, engine='openpyxl')
            if 'D_DATE' not in coef.columns:
                coef['D_DATE'] = pd.to_datetime(folder).strftime('%Y-%m-%d')
            else:
                coef = coef.dropna(subset=['D_DATE'])
                coef['D_DATE'] = [x.strftime('%Y-%m-%d') for x in coef['D_DATE']]
            if 'Unnamed: 0' in coef.columns:
                coef = coef.drop(columns='Unnamed: 0')
            if file_type == 'data_fund':
                coef = coef.reindex(
                    columns=['C_FUNDNAME', '基金类型', 'D_DATE', 'C_FULLNAME', 'NAV', 'NAV_累计', 'NetAsset',
                             'TotalAsset', '基金份额', 'Deposit', '股票', '债券', 'ABS', '基金投资', '买入返售', '卖出回购', '衍生品',
                             '可转债'])
            elif file_type == 'Holdings' and sec_type == 'stock_':
                coef = coef.reindex(
                    columns=['D_DATE', 'L_SETCODE', 'C_FULLNAME', 'C_FUNDNAME', 'C_SUBNAME_BSH', 'C_STOPINFO',
                             'F_MOUNT', 'F_PRICE', 'F_ASSET', 'F_ASSETRATIO', 'F_NETCOST', 'F_COST',
                             'F_COSTRATIO', 'L_STOCKTYPE', 'L_FUNDKIND', 'L_FUNDKIND2', 'L_FUNDTYPE', 'code',
                             'LISTEDSECTOR', 'TOTALMV', 'PE_TTM', 'PB', '上市板块'])
            elif file_type == 'Liq_holdings':
                coef = coef.reindex(
                    columns=['D_DATE', 'C_FULLNAME', 'C_FUNDNAME', 'C_SUBNAME_BSH', 'C_STOPINFO', 'code',
                             'F_MOUNT', 'F_ASSET', 'F_ASSETRATIO', '1日可变现_张', '5日可变现_张', '1日可变现', '5日可变现',
                             '质押量_张', '质押市值', '受限数量', '变现天数', '是否高流动性债券'])

            if file_type == 'Holdings' and sec_type == 'all':
                pass
            elif file_type == 'Holdings':
                new_cols = self.dict_dpe_cols[sec_type.replace('all', '') + file_type.lower()].set_index('Sheet_columns_orig')[
                    'Sheet_columns'].to_dict()
                coef.columns = [new_cols[x] for x in coef.columns]
            else:
                coef.columns = self.dict_dpe_cols[sec_type.replace('all', '') + file_type.lower()]['Sheet_columns']

            if ptf_names is not None:
                coef = coef[coef['C_FUNDNAME'].isin(ptf_names)].copy()

            if add_code:
                coef['portfolio_code'] = coef['C_FUNDNAME'].map(self.name_to_code)

            coef = _format_string(coef)
            tablename = self.table_dict[sec_type.replace('all', '') + file_type.lower()]  # 数据库标名
            t = pd.to_datetime(folder).strftime('%Y-%m-%d')
            self.delete_table(tablename, t=t, ptf_names=ptf_names)
            self.insert_table(tablename, coef, t=t)

    def insert_temp_t(self, folder_path, t, file_list=None):
        '''批量更新当日指标的temp data，数据库仅保留当日结果'''
        file_path = folder_path + t.replace('-', '') + '\\'
        file_list = ['data_benchmark', 'data_db'] if file_list is None else file_list
        for file_name in file_list:
            self.insert_tmpdata(file_path, file_name)

    def insert_tmpdata(self, file_path, file_name):
        dict_cols = pd.read_excel(os.path.join(DIR_OF_MAIN_PROG, 'data', 'SheetColumns_%s.xlsx' % file_name),
                                  sheet_name=None, engine='openpyxl')
        data = pd.read_excel(file_path + '%s.xlsx' % file_name, sheet_name=None, engine='openpyxl')

        for sheet in dict_cols.keys():
            data_temp = _format_string(data[sheet])
            data_temp.columns = dict_cols[sheet]['Sheet_columns']
            table_name = self.table_dict[sheet]
            self.replace_table(table_name, data_temp)

    def delete_table(self, table, t, t_colname='D_DATE', ptf_names=None, name_colname='C_FUNDNAME', schema='quant'):
        if ptf_names is not None:
            self._delete_table_in(table, t=t, t_colname=t_colname, ptf_names=ptf_names, name_colname=name_colname, schema=schema)
            return None

        condition = column(t_colname) == t
        try:
            self.db_quant.delete(table.lower(), condition, schema)
            logger.info('%s删除成功，table: %s deleted from database.' % (t, table))
        except (DoesNotExist, exc.NoSuchTableError):
            logger.warning('%s删除失败，table: %s data not found.' % (t, table))
            pass

    def _delete_table_in(self, table, t, ptf_names, t_colname='D_DATE', name_colname='C_FUNDNAME', schema='quant'):
        condition = and_(column(t_colname) == t, column(name_colname).in_(ptf_names))
        cond_str = '%s = %s, %s in (%s)' % (t_colname, t, name_colname, ', '.join(ptf_names))

        try:
            self.db_quant.delete(table.lower(), condition, schema)
            logger.info('删除成功，table: %s - %s deleted from database.' % (table, cond_str))
        except (DoesNotExist, exc.NoSuchTableError):
            logger.warning('删除失败，table: %s - %s not found.' % (table, cond_str))

    def insert_table(self, table, data, t, schema='quant', if_exists='append'):
        if data.shape[0] == 0:
            logger.info('%s 数据为空，无需插入' % t)
            return None
        self.db_quant.insert_dataframe(table=table.lower(), data=data, schema=schema, if_exists=if_exists)
        logger.info('插入成功，table: %s - %s inserted to database.' % (table, t))

    def replace_table(self, table, data, schema='quant'):
        db_quant.insert_dataframe(table=table.lower(), data=data, schema=schema, if_exists='replace')
        logger.info('更新成功，table: %s replaced.' % table)

if __name__ == '__main__':
    pass
    # # gitlab
    # run_my_function()
    #
    # # github test
    # run_my_function()
