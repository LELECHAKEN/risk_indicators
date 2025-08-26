# @Time : 2021/12/16 10:40 
# @Author : for wangp
# @File : daily_check_mail.py
# @Software: PyCharm

import os
import datetime
from WindPy import w
w.start()

from scripts.settings import config, DIR_OF_MAIN_PROG
from scripts.utils.log_utils import logger
from scripts.db import OracleDB, sqls_config
from scripts.utils.send_email_standard import format_body, send_email

db_risk = OracleDB(config['data_base']['QUANT']['url'])


def get_nav_check(t):
    q_nav = sqls_config['return_check']['Sql'] % t
    nav_check = db_risk.read_sql(q_nav)
    return nav_check

def get_fund_divid(t):
    q_dvd = sqls_config['dividend_info']['Sql']
    fund_dvd = db_risk.read_sql(q_dvd)
    fund_dvd = fund_dvd[fund_dvd['d_date'] == t].copy()
    fund_dvd['dividend'] = [round(x, 4) for x in fund_dvd['dividend']]
    return fund_dvd


if __name__ == '__main__':
    receiver = 'wangp@maxwealthfund.com'
    c_receiver = 'zhangyf@maxwealthfund.com;zhengx@maxwealthfund.com;zhouyy@maxwealthfund.com;daisj@maxwealthfund.com;shiy02@maxwealthfund.com'
    t_today = datetime.datetime.now().strftime('%Y-%m-%d')
    t = w.tdaysoffset(-1, t_today, "").Data[0][0].strftime('%Y-%m-%d')
    title = '%s单位净值数据检查'%t

    cols_map = {'c_fundname': '组合名称', 'd_date': '日期',
                'ret_daily_val': '复权净值(处理后)', 'NAV_orig': '单位净值(估值)',
                'ret': '单日收益率', 'nav': '单位净值(复权)',
                'dividend': '当日分红', 'l_fundtype': '基金类型'}
    nav_check = get_nav_check(t).rename(columns=cols_map)
    fund_dvd = get_fund_divid(t).rename(columns=cols_map)
    # if nav_check.shape[0] > 0:
    body = format_body([nav_check, fund_dvd], ['（1）复权单位净值检查结果如下', '（2）当日组合分红信息如下'])
    send_email(receiver, c_receiver, title, body)