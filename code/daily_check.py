# @Time : 2021/7/14 15:23 
# @Author : for wangp
# @File : daily_check.py 
# @Software: PyCharm

import os
import datetime
import traceback
import pandas as pd
from WindPy import w
w.start()

from scripts.utils.log_utils import logger
from scripts.settings import config, DIR_OF_MAIN_PROG
from scripts.data_check import DataCheck
from scripts.updateFundNav import PtfNav


def check_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == '__main__':
    ptfnav = PtfNav()
    t = ptfnav.retrieve_n_tradeday(datetime.datetime.today(), 0)
    print(t)

    save_path = os.path.join(DIR_OF_MAIN_PROG, 'data', t.replace('-', '')) + '\\'
    check_path(save_path)

    try:
        ptfnav.integrate_nav(t, plus_n=0)
        ptfnav.integrate_nav(t, plus_n=1)
        ptfnav.integrate_nav(t, plus_n=2)

        dc = DataCheck(t)
        res_1 = dc.check_nav()
        res_2_data, res_2_prod = dc.check_ret_cum()   # 检查全量
        res_3 = dc.check_ret_daily()

        if res_1.shape[0] > 0:
            logger.error('【alert】期初净值检查')
            print(res_1)
        else:
            logger.info('【无误】期初净值检查')
        if res_2_prod.shape[0] > 0:
            logger.error('【alert】每日收益率检查有误')
            print(res_2_prod)
        else:
            logger.info('【无误】每日收益率检查')
        if res_3.shape[0] > 0:
            logger.error('【alert】当日收益率检查')
            print(res_3)
        else:
            logger.info('【无误】当日收益率检查')

        writer = pd.ExcelWriter(save_path + 'check_result.xlsx')
        res_1.to_excel(writer, sheet_name='期初净值检查', index=False)
        res_2_data.to_excel(writer, sheet_name='每日收益率检查-data', index=False)
        res_2_prod.to_excel(writer, sheet_name='每日收益率检查', index=False)
        res_3.to_excel(writer, sheet_name='当日收益率检查', index=False)
        writer.save()

    except Exception as e:
        logger.error("err_msg：%s\t%s" % (str(e), traceback.format_exc().replace("\n", "")))
        exit(1)

    # # 用于批量跑数
    # w.start()
    # date_list = w.tdays("2024-12-26", "2025-01-10", "").Data[0]
    # date_list = [x.strftime('%Y-%m-%d') for x in date_list]
    # for t in date_list:
    #     ptfnav.integrate_nav(t, ptf_codes=['SASS20.SMA'])