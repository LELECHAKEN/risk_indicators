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
from scripts.RiskIndicator_mg import RiskIndicator_mg
from scripts.RiskIndicator_mismatch import RiskIndicator_mismch
from scripts.CBIndicators import CBIndicators
from scripts.derivatives import Derivatives
from scripts.risk_warning import PortfolioWarning
from scripts.risk_alert_additional import PortfolioWarning_add
from scripts.utils_ri.IndexFund_deviation import IndexFundIndicators
from scripts.cbond_indicators import CBondIndicators

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
    t = (datetime.today() - pd.Timedelta(1, 'd')).strftime('%Y-%m-%d')
    # t = '2025-04-14'
    # print(t)
    save_path_rc = config['shared_drive_data']['risk_indicator_daily']['path']
    save_path = os.path.join(DIR_OF_MAIN_PROG, 'data') + '\\'
    # data_path_out = save_path + '%s\\' % t.replace('-', '')
    data_path_out = save_path + '%s\\' % t.replace('-', '')
    demo_code.check_path(data_path_out)
    # check_path(data_path_out)

    # RiskIdx = RiskIndicators(t)
    # RiskIdx.RepoCleaning()
    # RiskIdx.SaveCleaningData(folder_path=data_path_out)
    # ConcenIdx = ConcentrationIndicators(t)
    # ConcenIdx.SaveCleaningData(folder_path=data_path_out)
    # InsertData = demo_code.InsertData()
    # InsertData.insert_dpe_t(save_path, t)

    # ConcenIdx = ConcentrationIndicators(t)
    # demo_code.saveData(data_path_out, ConcenIdx)
    # ConcenIdx.res = ConcenIdx.CalculateAll()
    #
    # demo_code.insert_dpe_t(save_path, t)
    # demo_code.insert_temp_t(save_path, t, 'quant')

    w.start()
    tdays = w.tdays('2025-06-23', '2025-07-02')
    tds = [t.strftime('%Y-%m-%d') for t in tdays.Data[0]]
    # tds = ['2022-05-13']
    for td in tds:

        print(td)
        data_path_out = save_path + '%s\\' % td.replace('-', '')
        CBIdx = CBondIndicators(td, data_path_out)
        CBIdx.CalculateAll()
        # # 插入到数据库
        CBIdx.insert_2_db()

        # RiskIdx = RiskIndicators(t)
        # RiskIdx.classify_credit_bond_info()
        # data_path_out = save_path + '%s\\' % td.replace('-', '')
        # RiskIdx.add_portfolio_code(table='dpe_portfoliobond', t=td)
        # ConcenIdx = ConcentrationIndicators(td)
        # ConcenIdx.SaveCleaningData(folder_path=data_path_out)
        #
        # LiqIdx = LiquidityIndicators(td, data_path_out)
        # LiqIdx.res = LiqIdx.CalculateAll()
        # LiqIdx.insert2db('RiskIndicators', 'rc_liquidity', 'Liquidity', LiqIdx.res)
        # LiqIdx.supply_for_rc_liquidity()
        # LiqIdx.getLeverageDist()
        #
        # LiqIdx_core = LiquidityIndicators_core(td, data_path_out)
        # res_core = LiqIdx_core.getCertainIdx(interval=2)
        # LiqIdx_core.insert2db('RiskIndicators', 'rc_lr_core', 'Liquidity_core', res_core)
        # LiqIdx_core.CalculateAll()
        # LiqIdx_core.save_all(save_path_rc, td)
        # logger.info('Liquidity Indicators Done.')
        # mg = RiskIndicator_mg(td, data_path_out)
        # mg.calcAllIdx()
        # mg.insert2db()
    # mg.insert2db()

        # MarketIdx = MarketIndicators(td, data_path_out)
        # MarketIdx.calcHoldingRelated()
        # MarketIdx.insert2db('RiskIndicators', 'rc_mr_holding', 'Market_Holding', MarketIdx.res_holding)

    # 产品基础信息, rc_style
    # BasicIdx = BasicIndicators(t, data_path_out)
    # BasicIdx.res = BasicIdx.CalcAllIndices()
    # BasicIdx.res.to_excel(r"E:\risk_indicators\data\20240306\rc_style.xlsx")
    # BasicIdx.insert2db('RiskIndicators', 'rc_style', 'Basic', BasicIdx.res)
    # logger.info('Basic Indicators Done.')

    # 市场风险指标, 持仓相关rc_holding及净值相关指标rc_mr_holding
    # MarketIdx = MarketIndicators(t, data_path_out)
    # MarketIdx.calc_portytm_component()
    # MarketIdx.calcHoldingRelated()
    # MarketIdx.res_holding.to_excel(r"E:\risk_indicators\data\20240306\rc_mr_holding.xlsx")
    # MarketIdx.calcNavRelated(startDate='2022-12-31')  # TODO: 已进入2024年，暂保留2023年以来的全部净值
    # MarketIdx.calc_winning_ratio_t()

    # 风险收益指标：比较基准, rc_mr_return_bch
    # mkt_b = BenchmarkRiskReturn(t)
    # mkt_b.calcNavRelated()
    # mkt_b.insert2db()
    # bch_t1 = mkt_b.res_nav.drop(columns=['insert_time'])
    # bch_t1['D_DATE'] = pd.to_datetime(bch_t1['D_DATE'])
    # # bch_t1 = retrieveBchIdxLastDay(t)
    # logger.info('Benchmark indicators done.')

    # # 信用风险指标, rc_credit
    # CreditIdx = CreditIndicators(t, data_path_out)
    # CreditIdx.CalculateAll()
    # CreditIdx.resAll = CreditIdx.IntegrateAll()
    # CreditIdx.resAll.to_excel(r"E:\risk_indicators\data\20240306\rc_credit.xlsx")
    # CreditIdx.insert2db('RiskIndicators', 'rc_credit', 'Credit', CreditIdx.resAll)
    # logger.info('Credit Indicators Done.')

    # 流动性指标, rc_liquidity
    # LiqIdx = LiquidityIndicators(t, data_path_out)
    # LiqIdx.res = LiqIdx.CalculateAll()
    # LiqIdx.res.to_excel(r"E:\risk_indicators\data\20240306\rc_liquidity.xlsx")
    # LiqIdx.insert2db('RiskIndicators', 'rc_liquidity', 'Liquidity', LiqIdx.res)
    # LiqIdx.insert_repo_ttm()
    # LiqIdx.holdings.to_excel(data_path_out + 'Liq_holdings.xlsx', index=False)

    # 核心流动性指标, rc_lr_core
    # LiqIdx_core = LiquidityIndicators_core(t, data_path_out)
    # res_core = LiqIdx_core.getCertainIdx(interval=2)
    # LiqIdx_core.insert2db('RiskIndicators', 'rc_lr_core', 'Liquidity_core', res_core)
    # LiqIdx_core.CalculateAll()
    # LiqIdx_core.save_all(save_path_rc, t)
    # logger.info('Liquidity Indicators Done.')

    # 错配相关指标
    # ri_mm = RiskIndicator_mismch(t, data_path_out)
    # ri_mm.calc_all()
    # ri_mm.insert2db()

    # # 风险预警
    # rw = PortfolioWarning(t, data_path_out)
    # rw.calc_all_warnings()

    # 管理层风险指标，因管理层的信用风险预警需取预警模块的数据，故需在风险预警之后
    # mg = RiskIndicator_mg(t, data_path_out)
    # mg.calcAllIdx(save_path)
    # mg.saveAllIdx(save_path)
    # mg.saveAllIdx(save_path_rc)
    # mg.insert2db()
    # logger.info('mg Indicators done.')

    # 投委会决议相关指标监控预警
    # rw_add = PortfolioWarning_add(t)
    # rw_add.calc_all_alert()
    # rw_add.saveAll(save_path)
    # rw_add.saveAll(save_path_rc)

    # ri.HoldingsCleaning()
    # w.start()
    # date_list = w.tdays("2022-09-01", "2022-12-31", "").Data[0]
    # date_list = [x.strftime('%Y-%m-%d') for x in date_list]
    # for t in date_list:
    #     print(t)
    #     data_path_out = save_path + '%s\\' % t.replace('-', '')
    #     MarketIdx = MarketIndicators(t, data_path_out)
    #     MarketIdx.calc_portytm_component()

    # # 从数据库保存指标
    # retrieve_time_series(save_path_rc, t, t, '%s_RiskIndicators.xlsx' % t.replace('-', ''))
    #
    # # 风险指标的时间序列数据
    # retrieve_idc_repo(t)
    # retrieve_time_series(save_path_rc, '2023-01-01', t)
    # logger.info('Integration done.')

    # rw = PortfolioWarning(t, data_path_out)
    # rw.drawdown_warning()
    # rw.credit_warning()
    # rw.calc_all_warnings()

    # # 管理层风险指标，因管理层的信用风险预警需取预警模块的数据，故需在风险预警之后
    # mg = RiskIndicator_mg(t, data_path_out)
    # mg.calcAllIdx(save_path)
    # # mg.saveAllIdx(save_path)
    # mg.saveAllIdx(save_path_rc)
    # mg.insert2db()
    # logger.info('mg Indicators done.')

    # rw_add = PortfolioWarning_add(t)
    # rw_add.alert_postion()
