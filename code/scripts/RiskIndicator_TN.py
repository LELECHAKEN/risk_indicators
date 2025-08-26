#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time   : 2024/4/1 17:04
# !@File   : RiskIndicator_TN.py
# !@Author : shiyue
import datetime
import os
from scripts.settings import config, DIR_OF_MAIN_PROG
from scripts.utils.log_utils import logger
from scripts import demo_code

from scripts.BasicIndicators import BasicIndicators
from scripts.ConcentrationIndicators import ConcentrationIndicators
from scripts.MarketIndicators import MarketIndicators
from scripts.MarketIndicators_b import BenchmarkRiskReturn
from scripts.CreditIndicators import CreditIndicators
from scripts.LiquidityIndicators import LiquidityIndicators
from scripts.LiquidityIndicators_core import LiquidityIndicators_core
from scripts.RiskIndicator_mismatch import RiskIndicator_mismch
from scripts.derivatives import Derivatives


def tn_indicators(baseDate):
    tn_ptf = demo_code.tn_ptf_codes()
    for n, ptf_codes in tn_ptf.items():
        real_t = demo_code.retrieve_n_tradeday(baseDate, n)
        run_all_indicators(baseDate, real_t, ptf_codes)

    # 单独跑数使用
    # n = 2
    # ptf_codes = tn_ptf[n]
    # real_t = demo_code.retrieve_n_tradeday(baseDate, n)
    # run_all_indicators(baseDate, real_t, ptf_codes)


def run_all_indicators(baseDate, real_t, ptf_codes):
    logger.info('%s - Indicators Calculation Begin: %s' % (real_t, ', '.join(ptf_codes)))
    save_path_rc = config['shared_drive_data']['risk_indicator_daily']['path']
    save_path = os.path.join(DIR_OF_MAIN_PROG, 'data') + '\\'
    data_path_out = save_path + '%s\\' % real_t.replace('-', '')

    # 衍生品相关指标
    derivt = Derivatives(real_t, ptf_codes)
    derivt.run()
    logger.info('%s - Derivatives  done.' % real_t)

    # 集中度指标, rc_concentration
    ConcenIdx = ConcentrationIndicators(real_t, ptf_codes)
    ConcenIdx.SaveCleaningData(folder_path=data_path_out)
    ConcenIdx.res = ConcenIdx.CalculateAll()
    ConcenIdx.insert2db('RiskIndicators', 'rc_concentration', 'Concentration', ConcenIdx.res)
    logger.info('%s - Concentration done.' % real_t)

    # 将当日风险指标的中间数据插入数据库
    InsertData = demo_code.InsertData()
    InsertData.insert_dpe_t(save_path, real_t, ptf_codes)
    InsertData.insert_temp_t(save_path, real_t, file_list=['data_db'])
    ConcenIdx.classify_credit_bond_info()

    # 产品基础信息, rc_style
    BasicIdx = BasicIndicators(real_t, data_path_out, ptf_codes)
    BasicIdx.res = BasicIdx.CalcAllIndices()
    BasicIdx.insert2db('RiskIndicators', 'rc_style', 'Basic', BasicIdx.res)
    logger.info('%s - Basic Indicators Done.' % real_t)

    # 市场风险指标, 持仓相关rc_holding及净值相关指标rc_mr_holding
    MarketIdx = MarketIndicators(real_t, data_path_out, ptf_codes)
    MarketIdx.calc_portytm_component()
    MarketIdx.calcHoldingRelated()
    MarketIdx.calcNavRelated()
    MarketIdx.insert2db_dura()
    MarketIdx.insert2db('RiskIndicators', 'rc_mr_return', 'Market_return', MarketIdx.res_nav)
    MarketIdx.insert2db('RiskIndicators', 'rc_mr_holding', 'Market_Holding', MarketIdx.res_holding)
    logger.info('%s - Market indicators done.' % real_t)

    # 风险收益指标：比较基准, rc_mr_return_bch
    mkt_b = BenchmarkRiskReturn(real_t, ptf_codes)
    mkt_b.calcNavRelated()
    mkt_b.insert2db()
    logger.info('%s - Benchmark indicators done.' % real_t)

    # 信用风险指标, rc_credit
    CreditIdx = CreditIndicators(real_t, data_path_out, ptf_codes)
    CreditIdx.CalculateAll()
    CreditIdx.insert2db('RiskIndicators', 'rc_credit', 'Credit', CreditIdx.resAll)
    logger.info('%s - Credit Indicators Done.' % real_t)

    # 流动性指标, rc_liquidity
    LiqIdx = LiquidityIndicators(real_t, data_path_out, ptf_codes)
    LiqIdx.res = LiqIdx.CalculateAll()
    LiqIdx.insert2db('RiskIndicators', 'rc_liquidity', 'Liquidity', LiqIdx.res)
    LiqIdx.holdings.to_excel(data_path_out + 'Liq_holdings.xlsx', index=False)
    LiqIdx.supply_for_rc_liquidity()

    # 核心流动性指标, rc_lr_core
    LiqIdx_core = LiquidityIndicators_core(real_t, data_path_out, ptf_codes)
    res_core = LiqIdx_core.getCertainIdx(interval=2)
    LiqIdx_core.insert2db('RiskIndicators', 'rc_lr_core', 'Liquidity_core', res_core)
    LiqIdx_core.CalculateAll()
    LiqIdx_core.save_all(save_path_rc, baseDate)  # 存储在公盘的基期文件中
    logger.info('%s - Liquidity Indicators Done.' % real_t)

    # 错配相关指标
    ri_mm = RiskIndicator_mismch(real_t, data_path_out, ptf_codes)
    ri_mm.calc_all()
    ri_mm.insert2db()
    logger.info('%s - Mismatch Indicators Done.' % real_t)




