'''
@Description: 组合风险指标
@Author: Wangp
@Date: 2020-03-26 14:00:25
LastEditTime: 2021-05-17 18:01:45
LastEditors: Wangp
'''

import os
import openpyxl
import pandas as pd
import numpy as np
from WindPy import w
from datetime import datetime

from scripts.utils.log_utils import logger
from scripts.settings import config, DIR_OF_MAIN_PROG
from scripts.db import OracleDB, sqls_config
from scripts.db.db_utils import convert_columns
from scripts import demo_code
from scripts.utils_ri.RiskIndicators import RiskIndicators

from scripts.data_check import DataCheck
from scripts.BasicIndicators import BasicIndicators
from scripts.ConcentrationIndicators import ConcentrationIndicators
from scripts.MarketIndicators import MarketIndicators
from scripts.MarketIndicators_b import BenchmarkRiskReturn
from scripts.CreditIndicators import CreditIndicators
from scripts.LiquidityIndicators import LiquidityIndicators
from scripts.LiquidityIndicators_core import LiquidityIndicators_core
from scripts.RiskIndicator_mg import RiskIndicator_mg, RiskIndicator_cmt
from scripts.RiskIndicator_mismatch import RiskIndicator_mismch
from scripts.CBIndicators import CBIndicators
from scripts.cbond_indicators import CBondIndicators
from scripts.derivatives import Derivatives, Derivt_CRMW
from scripts.risk_warning import PortfolioWarning
from scripts.risk_alert_additional import PortfolioWarning_add
from scripts.utils_ri.IndexFund_deviation import IndexFundIndicators
from scripts.LiquidityHolders import LiquidityLiablility, LiquidityAsset
from scripts.RiskIndicator_TN import tn_indicators


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

    Indicators.data_wind_equity.to_excel(os.path.join(data_path, 'data_wind_equity.xlsx'), index=False)

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


def retrieve_time_series(save_path, startdate, enddate, file_name='RiskIndicators-v3.xlsx'):
    '''
    生成风险指标的时间序列数据，从数据库取数并保存至本地\n
    :param save_path: string, 保存的文件夹路径
    :param startdate: string, yyyy-mm-dd格式, 起始日期(含)
    :param enddate: string, yyyy-mm-dd格式, 结束日期(含)
    :param file_name: string, 保存的文件名及格式
    :return: None
    '''
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    q = sqls_config['rc_series']['Sql']

    writer = pd.ExcelWriter(os.path.join(save_path, file_name), engine='openpyxl')
    sheets = ['Concentration', 'Basic', 'Market_return', 'Market_return_bch', 'Market_Holding', 'Credit', 'Liquidity', 'Liquidity_core']
    tables = ['RC_CONCENTRATION', 'RC_STYLE', 'RC_MR_RETURN', 'RC_MR_RETURN_BCH', 'RC_MR_HOLDING', 'RC_CREDIT', 'RC_LIQUIDITY', 'RC_LR_CORE']
    dict_cols = pd.read_excel(os.path.join(save_path, '20220805_RiskIndicators.xlsx'), sheet_name=None, engine='openpyxl')
    for (sheet, table) in zip(sheets, tables):
        idx_temp = convert_columns(db_risk.read_sql(q%(table, startdate, enddate)).rename(columns={'PAIN_INDEX': 'pain_index', 'HIT_RATE': 'hit_rate'}))
        idx_temp['D_DATE'] = pd.to_datetime(idx_temp['D_DATE'])
        idx_temp = idx_temp.sort_values(by=['D_DATE', 'C_FUNDNAME'])
        if 'insert_time'.upper() in idx_temp:
            idx_temp = idx_temp.drop(columns=['insert_time'.upper()])
        idx_temp.columns = dict_cols[sheet].columns
        idx_temp.to_excel(writer, sheet_name=sheet, index=False)
    writer.save()


def retrieve_idc_repo(t):
    '''
    回购明细表(idc_repo)\n
    :param t: string, yyyy-mm-dd格式, 取数日期
    :return: None
    '''
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    q = sqls_config['idc_repo_t']['Sql']%t
    data_repo = db_risk.read_sql(q)
    data_repo = data_repo.loc[(data_repo['spot'] == '银行间') & (data_repo['direction'] == '融资回购') & (data_repo['actual_maturity_date'] > t), :].copy()
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
    t1 = w.tdaysoffset(-1, t, "").Data[0][0].strftime('%Y-%m-%d')
    idx_t1 = convert_columns(db_risk.read_sql(q%t1)).rename(columns={'PAIN_INDEX': 'pain_index', 'HIT_RATE': 'hit_rate'}).drop(columns=['INSERT_TIME'])
    idx_t1['D_DATE'] = pd.to_datetime(t)
    return idx_t1


def check_path(folder_path):
    '''
    检查文件夹是否存在, 若不存在则创建该文件夹\n
    :param folder_path: string, 文件夹路径
    :return: None
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == '__main__':
    # t = demo_code.retrieve_n_tradeday(datetime.today(), 0)
    t = '2025-06-23'
    # print(t)
    save_path_rc = config['shared_drive_data']['risk_indicator_daily']['path']
    save_path = os.path.join(DIR_OF_MAIN_PROG, 'data') + '\\'
    data_path_out = save_path + '%s\\' % t.replace('-', '')
    demo_code.check_path(data_path_out)
    logger.info('开始指标计算...')

    # InsertData = demo_code.InsertData()
    # InsertData.insert_dpe_t(save_path, t)

    # # 运行t+n估值指标
    # tn_indicators(t)
    # # # 从数据库保存指标
    # # demo_code.retrieve_idc_repo(t)
    # # demo_code.retrieve_time_series(save_path_rc, t, t, '%s_RiskIndicators.xlsx' % t.replace('-', ''))
    # #
    # # # 风险指标的时间序列数据
    # # demo_code.retrieve_time_series(save_path_rc, '2025-01-01', t)
    # # logger.info('Integration done.')
    #
    # # # 风险预警
    # # rw = PortfolioWarning(t, data_path_out)
    # # rw.drawdown_warning()
    # # rw.credit_warning()
    # # rw.calc_all_warnings()
    # # logger.info('Warnings done.')
    #
    # # 管理层风险指标，因管理层的信用风险预警需取预警模块的数据，故需在风险预警之后
    # # mg = RiskIndicator_mg(t, data_path_out)
    # # mg.calcAllIdx()
    # # mg.saveAllIdx(save_path_rc)
    # # mg.insert2db()
    # # logger.info('mg Indicators done.')
    #
    # # # 风险预警，因流动性预警需要管理层指标的数据，故需重跑一次
    # # rw.liquidity_warning()
    # # rw.saveAll(save_path_rc)
    # #
    # # # 投委会决议相关指标监控预警
    # # rw_add = PortfolioWarning_add(t)
    # # rw_add.calc_all_alert()
    # # rw_add.saveAll(save_path_rc)
    # #
    # # cmt = RiskIndicator_cmt(t, data_path_out)
    # # cmt.calculate_and_save(save_path_rc)
    #
    #
    # #
    # # #
    # # # # 衍生品相关指标(包含期货、CRMW)
    # # # derivt = Derivatives(t)
    # # # derivt.run()
    # # # logger.info('%s - Derivatives done.' % t)
    # # #
    #
    # # w.start()
    # # tdays = w.tdays('2025-01-01', '2025-05-12')
    # # tds = [t.strftime('%Y-%m-%d') for t in tdays.Data[0]]
    # # for t in tds:
    # #     print(t)
    # #     data_path_out = save_path + '%s\\' % t.replace('-', '')
    #
    # # # 集中度指标, rc_concentration
    # # ConcenIdx = ConcentrationIndicators(t)
    # # ConcenIdx.SaveCleaningData(folder_path=data_path_out)
    # # ConcenIdx.res = ConcenIdx.CalculateAll()
    # # ConcenIdx.insert2db('RiskIndicators', 'rc_concentration', 'Concentration', ConcenIdx.res)
    # # logger.info('%s - Concentration done.' % t)
    # #
    # # # 将当日风险指标的中间数据插入数据库
    # # InsertData = demo_code.InsertData()
    # # InsertData.insert_dpe_t(save_path, t)
    # # InsertData.insert_temp_t(save_path, t)
    # # ConcenIdx.classify_credit_bond_info()
    #
    # # # 产品基础信息, rc_style
    # # BasicIdx = BasicIndicators(t, data_path_out)
    # # BasicIdx.res = BasicIdx.CalcAllIndices()
    # # BasicIdx.insert2db('RiskIndicators', 'rc_style', 'Basic', BasicIdx.res)
    # # logger.info('%s - Basic Indicators Done.' % t)
    # #
    # # # 市场风险指标, 持仓相关rc_mr_holding及净值相关指标rc_mr_return
    # MarketIdx = MarketIndicators(t, data_path_out)
    # # MarketIdx.calc_portytm_component()
    # # MarketIdx.calcHoldingRelated()
    # # MarketIdx.insert2db('RiskIndicators', 'rc_mr_holding', 'Market_Holding', MarketIdx.res_holding)
    # # MarketIdx.calcNavRelated()
    # # MarketIdx.insert2db_dura()
    # # MarketIdx.insert2db('RiskIndicators', 'rc_mr_return', 'Market_return', MarketIdx.res_nav)

    # 特殊产品单独计算特殊关键久期：永赢邦利
    # key_years = [-1, 0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 7, 10, 30, np.inf]
    # prods = ['永赢邦利债券', '永赢坤益债券', '永赢中债-3-5年政金债指数', '永赢迅利中高等级短债', '永赢中债-1-3年政金债指数']
    # MarketIdx.define_key_years(key_years)
    # res_key_dura = MarketIdx.calc_key_duration_special(prods)
    # res_key_dura.to_excel(os.path.join(save_path_rc, '%s_特殊产品关键久期.xlsx' % (t.replace('-', ''))), index=False)
    # logger.info('%s - Market Indicators Done.' % t)
    #
    #
    # # w.start()
    # # tdays = w.tdays('2025-01-01', '2025-05-30')
    # # tds = [t.strftime('%Y-%m-%d') for t in tdays.Data[0]]
    # # for t in tds:
    # #     print(t)
    # #     data_path_out = save_path + '%s\\' % t.replace('-', '')
    # #     mg = RiskIndicator_mg(t, data_path_out)
    # #     mg.calcAllIdx()
    # #     mg.insert2db()
    #
    # #     MarketIdx = MarketIndicators(t, data_path_out)
    # #     MarketIdx.calcHoldingRelated()
    # #     MarketIdx.insert2db('RiskIndicators', 'rc_mr_holding', 'Market_Holding', MarketIdx.res_holding)
    # #
    # # # 特殊产品单独计算特殊关键久期：永赢邦利
    # # key_years = [-1, 0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 7, 10, 30, np.inf]
    # # prods = ['永赢邦利债券', '永赢坤益债券', '永赢中债-3-5年政金债指数', '永赢迅利中高等级短债', '永赢中债-1-3年政金债指数']
    # # MarketIdx.define_key_years(key_years)
    # # res_key_dura = MarketIdx.calc_key_duration_special(prods)
    # # res_key_dura.to_excel(os.path.join(save_path_rc, '%s_特殊产品关键久期.xlsx'%(t.replace('-', ''))), index=False)
    # # logger.info('%s - Market Indicators Done.' % t)
    # #
    # # # 风险收益指标：比较基准, rc_mr_return_bch
    # # mkt_b = BenchmarkRiskReturn(t)
    # # mkt_b.calcNavRelated()
    # # mkt_b.insert2db()
    # # logger.info('%s - Benchmark indicators done.' % t)
    # #
    # # 信用风险指标, rc_credit
    # # CreditIdx = CreditIndicators(t, data_path_out)
    # # CreditIdx.CalculateAll()
    # # CreditIdx.insert2db('RiskIndicators', 'rc_credit', 'Credit', CreditIdx.resAll)
    # # logger.info('%s - Credit Indicators Done.' % t)
    #
    # 流动性指标, rc_liquidity
    # LiqIdx = LiquidityIndicators(t, data_path_out)
    # LiqIdx.res = LiqIdx.CalculateAll()
    # # LiqIdx.insert2db('RiskIndicators', 'rc_liquidity', 'Liquidity', LiqIdx.res)
    # # LiqIdx.supply_for_rc_liquidity()
    # # LiqIdx.getLeverageDist()
    # # LiqIdx.holdings.to_excel(data_path_out + 'Liq_holdings.xlsx', index=False)
    #
    # # 核心流动性指标, rc_lr_core
    # # LiqIdx_core = LiquidityIndicators_core(t, data_path_out)
    # # res_core = LiqIdx_core.getCertainIdx(interval=2)
    # # LiqIdx_core.insert2db('RiskIndicators', 'rc_lr_core', 'Liquidity_core', res_core)
    # # LiqIdx_core.CalculateAll()
    # # LiqIdx_core.save_all(save_path_rc, t)
    # # logger.info('Liquidity Indicators Done.')
    #
    # # logger.info('%s - CBond 开始计算.' % t)
    CBIdx = CBondIndicators(t, data_path_out)
    CBIdx.CalculateAll()
    # # 插入到数据库
    CBIdx.insert_2_db()
    # # # 保存到excel
    # # CBIdx.saveAll(os.path.join(save_path_rc, '%s_RiskIndicators_CBOND.xlsx' % t.replace('-', '')))
    # # logger.info('%s - CBond done.' % t)
    # #
    # # # 错配相关指标
    # # ri_mm = RiskIndicator_mismch(t, data_path_out)
    # # ri_mm.calc_all()
    # # ri_mm.insert2db()
    # # logger.info('%s - Mismatch Indicators Done.' % t)
    # #
    # # # 从数据库保存指标
    # # demo_code.retrieve_time_series(save_path_rc, t, t, '%s_RiskIndicators.xlsx' % t.replace('-', ''))
    # # #
    # # # 风险指标的时间序列数据
    # # demo_code.retrieve_idc_repo(t)
    # # demo_code.retrieve_time_series(save_path_rc, '2024-01-01', t)
    # # logger.info('Integration done.')
    # # #
    # # #
    # 风险预警
    # rw = PortfolioWarning(t, data_path_out)
    # rw.drawdown_warning()
    # rw.credit_warning()
    # rw.calc_all_warnings()
    # #
    # 管理层风险指标，因管理层的信用风险预警需取预警模块的数据，故需在风险预警之后
    # mg = RiskIndicator_mg(t, data_path_out)
    # mg.calcAllIdx()
    # mg.saveAllIdx(save_path_rc)
    # mg.insert2db()
    # logger.info('mg Indicators done.')
    #
    # # 风险预警，因流动性预警需要管理层指标的数据，故需重跑一次
    # rw.liquidity_warning()
    # rw.saveAll(save_path_rc)
    # #
    # # 投委会决议相关指标监控预警
    # rw_add = PortfolioWarning_add(t)
    # rw_add.alert_position()
    # rw_add.calc_all_alert()
    # rw_add.saveAll(save_path_rc)
    # #
    # cmt = RiskIndicator_cmt(t, data_path_out)
    # cmt.calculate_and_save(save_path_rc)