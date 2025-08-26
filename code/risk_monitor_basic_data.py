# @Time : 2022/2/10 14:34 
# @Author : for wangp
# @File : risk_monitor_basic_data.py 
# @Software: PyCharm
# @Description: 投委会监控相关指标
import os
from datetime import datetime
from WindPy import w
import pandas as pd

from scripts import demo_code
from scripts.db import OracleDB, sqls_config
from scripts.utils.log_utils import logger
from scripts.settings import config, DIR_OF_MAIN_PROG
from scripts.MarketIndicators import MarketIndicators, IndexFundIndicators
from scripts.risk_alert_additional import PortfolioWarning_add

if __name__ == '__main__':
    t = demo_code.retrieve_n_tradeday(datetime.today(), 0)
    # t = '2024-03-06'

    save_path = os.path.join(DIR_OF_MAIN_PROG, 'data') + '\\'
    data_path_out = os.path.join(save_path, '%s\\' % t.replace('-', ''))
    demo_code.check_path(save_path)
    logger.info('开始运行%s日的投委会指标监控基础数据'%t)
    MktIdx = MarketIndicators(t, data_path_out)
    MktIdx.calc_maxdd_lastyear()
    MktIdx.calc_winning_ratio_t()
    MktIdx.Indexfund_BchPerformance()
    logger.info('完成运行%s日的投委会指标监控基础数据' % t)

    # 单独跑指数跟踪误差
    # w.start()
    # tdays = w.tdays('2025-01-01', '2025-05-29')
    # tds = [t.strftime('%Y-%m-%d') for t in tdays.Data[0]]
    # for t in tds:
    #     indexfund = IndexFundIndicators(t=t)
    #     indexfund.index_fund_deviation()

