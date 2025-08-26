import os
from datetime import datetime
from scripts import demo_code
from scripts.utils.log_utils import logger
from scripts.settings import config, DIR_OF_MAIN_PROG
from scripts.cbond_indicators import CBondIndicators
from scripts.risk_alert_additional import PortfolioWarning_add

if __name__ == '__main__':
    t = demo_code.retrieve_n_tradeday(datetime.today(), 0)
    # t = '2025-06-25'
    print(t)
    save_path_rc = config['shared_drive_data']['risk_indicator_daily']['path']
    save_path = os.path.join(DIR_OF_MAIN_PROG, 'data') + '\\'
    data_path_out = save_path + '%s\\' % t.replace('-', '')
    demo_code.check_path(data_path_out)

    logger.info('%s - CBond 开始计算.' % t)
    CBIdx = CBondIndicators(t, data_path_out)
    CBIdx.CalculateAll()
    # 插入到数据库
    CBIdx.insert_2_db()
    # 保存到excel
    CBIdx.saveAll(os.path.join(save_path_rc, '%s_RiskIndicators_CBOND.xlsx' % t.replace('-', '')))
    logger.info('%s - CBond done.' % t)

    # 投委会决议相关指标监控预警
    rw_add = PortfolioWarning_add(t)
    rw_add.calc_all_alert()
    rw_add.saveAll(save_path_rc)