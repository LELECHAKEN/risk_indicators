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
import traceback
from datetime import datetime

from scripts.utils.log_utils import logger
from scripts.settings import config, DIR_OF_MAIN_PROG
from scripts.db import OracleDB, sqls_config
from scripts.db.db_utils import convert_columns
from scripts import demo_code

from scripts.BasicIndicators import BasicIndicators
from scripts.ConcentrationIndicators import ConcentrationIndicators
from scripts.MarketIndicators import MarketIndicators
from scripts.MarketIndicators_b import BenchmarkRiskReturn
from scripts.CreditIndicators import CreditIndicators
from scripts.LiquidityIndicators import LiquidityIndicators
from scripts.LiquidityIndicators_core import LiquidityIndicators_core
from scripts.RiskIndicator_mg import RiskIndicator_mg, RiskIndicator_cmt
from scripts.RiskIndicator_mismatch import RiskIndicator_mismch
from scripts.cbond_indicators import CBondIndicators
from scripts.derivatives import Derivatives
from scripts.risk_warning import PortfolioWarning
from scripts.risk_alert_additional import PortfolioWarning_add
from scripts.RiskIndicator_TN import tn_indicators


if __name__ == '__main__':
    t = demo_code.retrieve_n_tradeday(datetime.today(), 0)
    # t = '2024-03-06'
    print(t)
    save_path_rc = config['shared_drive_data']['risk_indicator_daily']['path']
    save_path = os.path.join(DIR_OF_MAIN_PROG, 'data') + '\\'
    data_path_out = save_path + '%s\\' % t.replace('-', '')
    demo_code.check_path(data_path_out)
    logger.info('开始指标计算...')

    try:
        # 衍生品相关指标(包含期货、CRMW)
        derivt = Derivatives(t)
        derivt.run()
        logger.info('%s - Derivatives done.' % t)

        # 集中度指标, rc_concentration
        ConcenIdx = ConcentrationIndicators(t)
        ConcenIdx.SaveCleaningData(folder_path=data_path_out)
        ConcenIdx.res = ConcenIdx.CalculateAll()
        ConcenIdx.insert2db('RiskIndicators', 'rc_concentration', 'Concentration', ConcenIdx.res)
        logger.info('%s - Concentration done.' % t)

        # # 将当日风险指标的中间数据插入数据库
        InsertData = demo_code.InsertData()
        InsertData.insert_dpe_t(save_path, t)
        InsertData.insert_temp_t(save_path, t)
        ConcenIdx.classify_credit_bond_info()

        # 产品基础信息, rc_style
        BasicIdx = BasicIndicators(t, data_path_out)
        BasicIdx.res = BasicIdx.CalcAllIndices()
        BasicIdx.insert2db('RiskIndicators', 'rc_style', 'Basic', BasicIdx.res)
        logger.info('%s - Basic Indicators Done.' % t)

        # # 市场风险指标, 持仓相关rc_holding及净值相关指标rc_mr_holding
        MarketIdx = MarketIndicators(t, data_path_out)
        MarketIdx.calc_portytm_component()
        MarketIdx.calcHoldingRelated()
        MarketIdx.calcNavRelated()
        MarketIdx.insert2db_dura()
        MarketIdx.insert2db('RiskIndicators', 'rc_mr_return', 'Market_return', MarketIdx.res_nav)
        MarketIdx.insert2db('RiskIndicators', 'rc_mr_holding', 'Market_Holding', MarketIdx.res_holding)

        # 特殊产品单独计算特殊关键久期：永赢邦利
        key_years = [-1, 0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 7, 10, 30, np.inf]
        prods = ['永赢邦利债券', '永赢坤益债券', '永赢中债-3-5年政金债指数', '永赢迅利中高等级短债', '永赢中债-1-3年政金债指数']
        MarketIdx.define_key_years(key_years)
        res_key_dura = MarketIdx.calc_key_duration_special(prods)
        res_key_dura.to_excel(os.path.join(save_path_rc, '%s_特殊产品关键久期.xlsx'%(t.replace('-', ''))), index=False)
        logger.info('%s - Market Indicators Done.' % t)

        # 风险收益指标：比较基准, rc_mr_return_bch
        mkt_b = BenchmarkRiskReturn(t)
        mkt_b.calcNavRelated()
        mkt_b.insert2db()
        logger.info('%s - Benchmark indicators done.' % t)

        # 信用风险指标, rc_credit
        CreditIdx = CreditIndicators(t, data_path_out)
        CreditIdx.CalculateAll()
        CreditIdx.insert2db('RiskIndicators', 'rc_credit', 'Credit', CreditIdx.resAll)
        logger.info('%s - Credit Indicators Done.' % t)

        # 流动性指标, rc_liquidity
        LiqIdx = LiquidityIndicators(t, data_path_out)
        LiqIdx.res = LiqIdx.CalculateAll()
        LiqIdx.insert2db('RiskIndicators', 'rc_liquidity', 'Liquidity', LiqIdx.res)
        LiqIdx.holdings.to_excel(data_path_out + 'Liq_holdings.xlsx', index=False)
        # 补充rc_liquidity分期限杠杆成本字段
        LiqIdx.supply_for_rc_liquidity()

        # 核心流动性指标, rc_lr_core
        LiqIdx_core = LiquidityIndicators_core(t, data_path_out)
        res_core = LiqIdx_core.getCertainIdx(interval=2)
        LiqIdx_core.insert2db('RiskIndicators', 'rc_lr_core', 'Liquidity_core', res_core)
        LiqIdx_core.CalculateAll()
        LiqIdx_core.save_all(save_path_rc, t)
        logger.info('Liquidity Indicators Done.')

        # 错配相关指标
        ri_mm = RiskIndicator_mismch(t, data_path_out)
        ri_mm.calc_all()
        ri_mm.insert2db()
        logger.info('%s - Mismatch Indicators Done.' % t)

        # 运行t+n估值指标
        tn_indicators(t)

        # 从数据库保存指标
        demo_code.retrieve_idc_repo(t)
        demo_code.retrieve_time_series(save_path_rc, t, t, '%s_RiskIndicators.xlsx' % t.replace('-', ''))

        # 风险指标的时间序列数据
        demo_code.retrieve_time_series(save_path_rc, '2025-01-01', t)
        logger.info('Integration done.')

        # 风险预警
        rw = PortfolioWarning(t, data_path_out)
        rw.drawdown_warning()
        rw.credit_warning()
        rw.calc_all_warnings()
        logger.info('Warnings done.')

        # 风险预警，因流动性预警需要管理层指标的数据，故需重跑一次
        rw.liquidity_warning()
        rw.saveAll(save_path_rc)

        # 管理层风险指标，因管理层的信用风险预警需取预警模块的数据，故需在风险预警之后
        mg = RiskIndicator_mg(t, data_path_out)
        mg.calcAllIdx()
        mg.saveAllIdx(save_path_rc)
        mg.insert2db()
        logger.info('mg Indicators done.')

        cmt = RiskIndicator_cmt(t, data_path_out)
        cmt.calculate_and_save(save_path_rc)

    except Exception as e:
        logger.error("err_msg：%s\t%s" % (str(e), traceback.format_exc().replace("\n", "")))
        exit(1)



