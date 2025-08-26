#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time   : 2024/3/26 17:10
# !@File   : DailyRunTN.py
# !@Author : shiyue

from datetime import datetime
from scripts import demo_code
from scripts.RiskIndicator_TN import tn_indicators

if __name__ == '__main__':
    t = demo_code.retrieve_n_tradeday(datetime.today(), 0)
    # t = '2024-03-06'
    print(t)

    # 运行t+n估值指标
    tn_indicators(t)

