#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @Desc    :
# @Author  : zhouyy
# @Time    : 2021/3/30 14:55
'''
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import exc, and_, column
import traceback
from functools import reduce
import os
from WindPy import w

w.start()

from scripts.utils_ri import RiskIndicators
from scripts.trade_dt import date_utils
from scripts.db import OracleDB, column, sqls_config, db_factors
from scripts.settings import config, DIR_OF_MAIN_PROG
from .utils.log_utils import logger
from .db.util import DoesNotExist

dq = OracleDB(config['data_base']['JY']['url'])
DbFactor = db_factors.DbFactor
TradeDays = date_utils.TradeDays

WINDOW = [2, 9, 29]
UP_BIAS = [0.25, 0.9, 1.8]
DOWN_BIAS = [-0.25, -0.4, -0.6]


def save_all(save_path: str, data: dict, index=False):
    """
    存储data中的数据到文件中，key为sheet名，values为dataframe
    :param save_path:
    :param data:
    :return:
    """
    writer = pd.ExcelWriter(save_path)
    for k, v in data.items():
        v.to_excel(writer, sheet_name=k, index=index)
    writer.save()
    # writer.close()

def insert2db(table_name: list, res_list: list, basedate: str):
    """
    插入数据表
    """
    for table, data in zip(table_name, res_list):
        delete_table(table, basedate, 'quant')
        insert_table(table, data, 'quant')

def delete_table(table, t, schema):
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    condition = column('D_DATE') == t
    try:
        db_risk.delete(table.lower(), condition, schema)
        logger.info(f"{table} deleted，条件为：'D_DATE' == {t}")
    except (DoesNotExist, exc.NoSuchTableError):
        logger.info(f"删除失败(数据可能不存在)：{table} ，条件为：'D_DATE' == {t}, 错误为：{traceback.format_exc()} ")
        pass

def insert_table(table, data, schema, if_exists='append'):
    # TODO: insert_time字段
    try:
        db_risk = OracleDB(config['data_base']['QUANT']['url'])
        data['insert_time'] = datetime.datetime.now()
        db_risk.insert_dataframe(table=table.lower(), data=data, schema=schema, if_exists=if_exists)
        logger.info(f"插入数据成功，table: {table}")
    except Exception as e:
        logger.error(f"数据插入失败，table：{table}, 错误为：{traceback.format_exc()}")


def xldate_to_datetime(xldate):
    temp = datetime.datetime(1899, 12, 30)
    delta = datetime.timedelta(days=xldate)
    return temp + delta

def convert_market_code(market_code: np.array):
    """
    JY的SECUMARKET 转换为wind代码后缀
    :param market_code:
    :return:
    """
    return np.where(market_code == 83, '.SH', np.where(market_code == 90, '.SZ', ''))

def split_market_code(sec_code: np.array):
    """
    把wind代码分开， 如 000001.SH 拆成000001
    :param sec_code:
    :return: list
    """
    return list(map(lambda x: x.split('.')[0], sec_code))

def convert_sec_code(sec_code, from_='Wind', to_=None, dict={'sec_code': 'bond_code', 'jy_market': 'bond_market'}):
    """
    代码切换
    :param sec_code:
    :param from_:
    :param to_:
    :param dict: 将6位数字代码‘bond_code’ 拼接jy市场代码（89,90）成wind代码
    :return:
    """

    sec_code = sec_code.copy()
    # 把wind代码分开， 如 000001.SH 拆成000001
    if (from_ == 'Wind') & (to_ is None):
        return list(map(lambda x: x.split('.')[0], sec_code))

    # JY的证券代码（000001）和市场代码（89，90）拼接成wind代码（000001.SH）
    if (from_ == 'JY') & (to_== 'Wind'):
        sec_code['wind_market'] = convert_market_code(sec_code[dict['jy_market']])
        return sec_code.apply(lambda x: x[dict['sec_code']] + x['wind_market'], axis=1).to_list()

def _cal_turnover_avg(turnover_mat, list_days, n):
    """
    计算不同参数下两种平均换手率
    :param turnover_mat:
    :param n:
    :return:
    """
    turnover_mean_mat = turnover_mat.rolling(window=n, min_periods=n).mean()
    # 剔除连续上市不足n日、或行情数据不足n日的（避免rolling_sum报错）
    turnover_mat_notnull = turnover_mat.loc[:, turnover_mean_mat.notnull().sum() > 0]

    # 1. 简单移动平均
    turnover_mean_n_mat = turnover_mat_notnull.apply(
        lambda x: x.dropna().rolling(window=n).mean().dropna()).fillna(method='ffill')
    turnover_mean_n = turnover_mean_n_mat.stack().to_frame('mean').reset_index()
    turnover_mean_n['n'] = n
    turnover_mean_n['method'] = 'SMA'

    # 2. 时间加权平均
    turnover_ewm_n_mat = turnover_mat_notnull.apply(
        lambda x: x.dropna().ewm(span=n, ignore_na=True).mean().dropna()).fillna(method='ffill')
    turnover_ewm_n = turnover_ewm_n_mat.stack().to_frame('mean').reset_index()
    turnover_ewm_n['n'] = n
    turnover_ewm_n['method'] = 'EWMA'

    # 3. 上市超过10个交易日，但行情数据不足10日的（如私募EB, 或发生停牌），换手率数据0填充
    cond = (turnover_mat.notnull().cumsum() < 10) & (list_days > 10)
    turnover_mean_n_temp = turnover_mat.fillna(0).rolling(window=10, min_periods=1).mean().where(cond).dropna(how='all',
                                                                                                              axis=1)
    turnover_mean_n_temp = turnover_mean_n_temp.stack().to_frame('mean').reset_index()
    turnover_mean_n_temp['method'] = 'SMA'
    turnover_mean_n_temp['n'] = n

    turnover_ewm_n_temp = turnover_mat.fillna(0).apply(
        lambda x: x.dropna().ewm(span=n, ignore_na=True).mean().dropna()).fillna(method='ffill').where(cond).dropna(
        how='all', axis=1)
    turnover_ewm_n_temp = turnover_ewm_n_temp.stack().to_frame('mean').reset_index()
    turnover_ewm_n_temp['method'] = 'EWMA'
    turnover_ewm_n_temp['n'] = n

    turnover_mean_n = pd.concat([turnover_mean_n, turnover_mean_n_temp])
    turnover_ewm_n = pd.concat([turnover_ewm_n, turnover_ewm_n_temp])
    return turnover_mean_n, turnover_ewm_n

def cal_insert_cbond_turnover_avg(cbond_windcode_list: list, beg_date: str, end_date: str, n_days=[2, 5, 10, 20]):
    """
    计算转债市场换手率均值，分别采用简单平均、时间加权，回看天数 n_days
    cbond_windcode_list: 转债wind代码
    :param beg_date:
    :param end_date:
    :param n_days:
    :return:
    """
    logger.info(f"计算持仓转债不同区间换手率")
    cbond_list = pd.DataFrame({'cbond_windcode': cbond_windcode_list})
    cbond_list['sec_code'] = cbond_list['cbond_windcode'].apply(lambda x: x.split('.')[0])
    # 组合持仓明细
    db_jy = DbFactor(config['data_base']['JY']['url'])
    basic_info = db_jy.get_factor('cbond_basicinfo_field', code_list=list(set(cbond_list['sec_code'])),
                                  field='b.SecuCode')
    basic_info['cbond_windcode'] = convert_sec_code(basic_info[['cbond_code', 'cbond_market']],
                                                        from_='JY', to_='Wind',
                                                        dict={'sec_code': 'cbond_code', 'jy_market': 'cbond_market'})
    basic_info.loc[basic_info['list_date'].notnull(), 'list_datenum'] = basic_info['list_date'].dropna().apply(
        lambda x: int(datetime.datetime.strftime(x, '%Y%m%d')))
    basic_info.loc[basic_info['list_date'].notnull(), 'list_datenum'] = \
        basic_info['list_date'].dropna().apply(lambda x: int(datetime.datetime.strftime(x, '%Y%m%d')))
    # 获取最早上市日期
    list_date_min = int(basic_info['list_date'].min().strftime('%Y%m%d'))
    duration_days = TradeDays.get_adjustdate(list_date_min, int(end_date.replace('-', '')), ('D', 1))
    turnover_daily = db_jy.get_factor('cbond_turnover_field', code_list=list(set(cbond_list['sec_code'])),
                                      beg_date=str(list_date_min).replace('-', ''), end_date=end_date.replace('-', ''),
                                      field='b.SecuCode')
    turnover_daily.drop_duplicates(inplace=True)
    turnover_daily['cbond_windcode'] = convert_sec_code(turnover_daily[['cbond_code', 'secumarket']],
                                                        from_='JY', to_='Wind',
                                                        dict={'sec_code': 'cbond_code', 'jy_market': 'secumarket'})
    turnover_mat = turnover_daily.pivot_table(index='tradingday', columns='cbond_windcode', values='turnoverrate')
    turnover_mat = turnover_mat.reindex(index=pd.to_datetime(duration_days['trade_dt'].astype('str')),
                                        columns=np.sort(list(set(cbond_list['cbond_windcode']))))

    # 成交量
    volume_mat = turnover_daily.pivot_table(index='tradingday', columns='cbond_windcode', values='turnovervolume')
    volume_mat = volume_mat.reindex(index=turnover_mat.index, columns=turnover_mat.columns)

    turnover_mean = pd.DataFrame()
    volume_mean = pd.DataFrame()
    # 上市天数（交易日）
    # 20240223： 退市后为-1
    list_days = basic_info.pivot(index='list_date', columns='cbond_windcode', values='cbond_code').reindex(
                                    index=turnover_mat.index,
                                    columns=turnover_mat.columns).fillna(method='ffill').notnull().cumsum()
    # 20240223： 退市后为-1，未退市为1
    if_delist = basic_info.pivot(index='trade_enddate', columns='cbond_windcode', values='cbond_code').reindex(
        index=turnover_mat.index,
        columns=turnover_mat.columns).fillna(method='ffill').notnull() * -2 + 1
    list_days = list_days * if_delist.values

    for n in n_days:
        turnover_sma_n, turnover_ewm_n = _cal_turnover_avg(turnover_mat, list_days, n)
        volume_sma_n, volume_ewm_n = _cal_turnover_avg(volume_mat, list_days, n)
        # 合并数据
        turnover_mean = reduce(lambda x, y: pd.concat([x, y]), [turnover_mean, turnover_sma_n, turnover_ewm_n])
        volume_mean = reduce(lambda x, y: pd.concat([x, y]), [volume_mean, volume_sma_n, volume_ewm_n])

    # 上市10个交易日内不做监控
    turnover_mean = pd.merge(turnover_mean, list_days.stack().to_frame('list_days').reset_index(),
             left_on=['trade_dt', 'cbond_windcode'], right_on=['trade_dt', 'cbond_windcode'], how='left')
    turnover_mean = turnover_mean[turnover_mean['list_days'] > 10]
    turnover_mean = turnover_mean.rename(columns={'mean': 'turnover_mean'})
    volume_mean = volume_mean.rename(columns={'mean': 'volume_mean'})
    turnover_mean = pd.merge(turnover_mean, volume_mean, left_on=['trade_dt', 'cbond_windcode', 'n', 'method'],
                             right_on=['trade_dt', 'cbond_windcode', 'n', 'method'], how='left')

    # 格式调整
    turnover_mean['trade_dt'] = turnover_mean['trade_dt'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
    turnover_mean = turnover_mean.drop(columns=['list_days'])
    turnover_mean = turnover_mean[(turnover_mean['trade_dt'] >= beg_date) & (turnover_mean['trade_dt'] <= end_date)]
    turnover_mean = pd.merge(basic_info[['cbond_windcode', 'cbond_name']], turnover_mean, left_on='cbond_windcode',
                             right_on='cbond_windcode', how='right')
    turnover_mean['turnover_mean'] = turnover_mean['turnover_mean'].round(2)
    turnover_mean['volume_mean'] = turnover_mean['volume_mean'].round(2)
    # 插入数据库
    table_names = ['RC_CB_TURNOVER_MEAN']
    datas = [turnover_mean]
    cond = and_(column("trade_dt") >= beg_date, column("trade_dt") <= end_date)
    logger.info(f"开始落库，表名{table_names}...")
    db_factor = DbFactor(config['data_base']['QUANT']['url'])
    db_factor.insert2db(table_name=table_names, res_list=datas, cond=cond)
    db_factor.close()
    return turnover_mean

class CBIndicators(RiskIndicators):
    def __init__(self, t, save_path, ptf_codes=None):
        self.basedate = t
        self.save_path = save_path
        self._format_ptf_codes(ptf_codes)
        # 2022/11/30： 参与率调整为15%
        self.participate = 0.15
        self._InterestBond = ['政策银行债', '国债', '央行票据', '地方政府债', '政府支持机构债']
        self._CBBond_type = ['可转债', '可交换债']
        self._loadFile()
        self._loadHoldings(self.save_path)

        # 产品表（获取专户剩余期限）TODO: 路径更改 r'\\shaoafile01\RiskManagement\'
        self.product_outline = pd.read_excel(config['shared_drive_data']['product_outline']['path'],
                                             sheet_name=config['shared_drive_data']['product_outline']['sheet']
                                             , engine='openpyxl').drop_duplicates()

        # 提取转债持仓(F_ASSETRATIO为占净资产比例（%）)
        self.CBbond_holdings = self.bond_holdings[
                            self.bond_holdings['WINDL2TYPE'].isin(self._CBBond_type)].drop_duplicates()
        self.CBbond_holdings.columns = list(map(lambda x: x.lower(), self.CBbond_holdings.columns))
        print('Holdings Cleaning Done.')
        # 获取基本信息、赎回条款
        self.get_basic_info()
        self.get_call_clause()

        # 计算一些中间数据
        cal_insert_cbond_turnover_avg(self.CBbond_holdings['code'].to_list(), t, t, n_days=[2, 5, 10, 20])
        self.get_remain_amount() #转债余额
        self.get_turnover() # 换手率
        self.cal_monetize_days() # 变现天数
        self.get_deriv_indicators() # 转股溢价率、转换价值等
        #特殊条款(对新增转债)
        # self.get_special_clause()


    def get_last_holdings(self, t):
        """
        获取上一日持仓8(用于监控新增转债特殊条款）
        :return:
        """
        # 上一日持仓
        date_this = t.replace('-', '')
        date_last = str(date_utils.TradeDays.offset_date(int(date_this), -1)[0])
        save_path_last = self.save_path.replace(date_this, date_last)
        try:
            bond_holdings_last = pd.read_excel(save_path_last + 'Holdings.xlsx',
                                          sheet_name='bond_holdings', engine='openpyxl').drop_duplicates()
            self.CBbond_holdings_last = bond_holdings_last[bond_holdings_last['WINDL2TYPE'].isin(self._CBBond_type)]
        except Exception as e:
            # raise ValueError(f"读取上一日持仓失败， {e}")
            self.CBbond_holdings_last = None
            print(f"读取上一日持仓失败， {e}")


    def get_basic_info(self):
        """
        转债基本信息：转债内部代码、转债代码、转债简称、正股代码、正股简称、上市起始日、上市截止日、转换起始日、转换截止日、停止转股日(提前到期）
        :return:
        """
        print("获取转债基本信息...")
        # 1. 基本信息
        CBbond_holdings = self.CBbond_holdings
        basic_info = pd.DataFrame()
        sec_list = CBbond_holdings['code'].unique().tolist()
        loop_num = int(np.ceil(len(sec_list) / 1000))
        for i in range(loop_num):
            secs_temp = sec_list[i * 1000: min((i + 1) * 1000, len(sec_list))]
            secs_temp_str = ','.join('\'' + str(x.split('.')[0]) + '\'' for x in secs_temp)
            q = sqls_config['cbond_basicinfo']['Sql']
            q = q % (secs_temp_str)
            basic_info = dq.read_sql(q)
            basic_info.drop_duplicates(inplace=True)
        basic_info['cbond_windcode'] = convert_sec_code(basic_info, from_='JY', to_='Wind',
                                                        dict={'sec_code': 'cbond_code', 'jy_market': 'cbond_market'})
        self.basic_info = basic_info
        # 未上市转债
        new_list = basic_info[(basic_info['list_date'].isnull()) | (basic_info['list_date'] > pd.to_datetime(self.basedate))]
        if new_list.shape[0] > 0:
            print(f"持仓中存在未上市转债{new_list['cbond_name'].to_list()} \n {new_list['cbond_windcode'].to_list()}，预警中将剔除...")
            self.CBbond_holdings = CBbond_holdings.loc[~CBbond_holdings['code'].isin(new_list['cbond_windcode'].to_list())]
        pass

    def get_special_clause(self):
        """
        从wind api 中获取转债特殊条款: 赎回、下修、条件回售
        :return:
        """
        CBbond_holdings = self.CBbond_holdings.copy()
        if not (self.CBbond_holdings_last is None):
            CBbond_holdings_last = self.CBbond_holdings_last.copy()
            sec_list_this = CBbond_holdings['code'].unique().tolist()
            sec_list_last = CBbond_holdings_last['code'].unique().tolist()
            # 新增转债
            sec_list = list(set(sec_list_this) - set(sec_list_last))
            if len(sec_list) > 0:
                try:
                    data_wind = w.wss(sec_list,
                                      "clause_calloption_redeemitem,clause_reset_item,clause_putoption_sellbackitem,"
                                      "clause_calloption_redeemclause,clause_reset_timepointclause,clause_putoption_timeputbackclause")
                except NameError:
                    raise NameError("wind接口未连接! 无法获取转债特殊条款数据")
                except Exception as e:
                    raise ValueError(f"使用wind api提取转债特殊条款出错！ErrorCode:{e}")
                else:
                    pass

                if not (data_wind.ErrorCode == 0):
                    raise ValueError(f"使用wind api提取转债特殊条款出错！ErrorCode:{data_wind.ErrorCode},"
                                     f"Data:{data_wind.Data}")
                special_clause = pd.DataFrame(data_wind.Data, columns=data_wind.Codes, index=data_wind.Fields).T
                special_clause = special_clause.reset_index().rename(
                                                        columns={'index': 'cbond_windcode',
                                                                 'clause_calloption_redeemitem'.upper(): '赎回条款',
                                                                 'clause_reset_item'.upper(): '特别下修条款',
                                                                 'clause_putoption_sellbackitem'.upper(): '条件回售条款',
                                                                 'clause_calloption_redeemclause'.upper(): '时点赎回条款',
                                                                 'clause_reset_timepointclause'.upper(): '时点回售条款',
                                                                 'clause_putoption_timeputbackclause'.upper(): '时点下修条款'})
                special_clause['cbond_code'] = split_market_code(special_clause['cbond_windcode'])
                self.special_clause = special_clause
        pass


    def get_call_clause(self):
        """
        提取转债赎回条款信息
        :param date:
        :return:
        """
        print("获取转债赎回条款信息...")
        #
        # w.wss("128125.SZ", "clause_calloption_redeemitem,clause_reset_item,clause_putoption_sellbackitem")
        # 1. 赎回条款: 赎回触发条件
        CBbond_holdings = self.CBbond_holdings
        date_list = CBbond_holdings['d_date'].unique()
        call_clause = pd.DataFrame()
        sec_list = CBbond_holdings['code'].unique().tolist()
        loop_num = int(np.ceil(len(sec_list) / 1000))
        for i in range(loop_num):
            secs_temp = sec_list[i * 1000: min((i + 1) * 1000, len(sec_list))]
            secs_temp_str = ','.join('\'' + str(x.split('.')[0]) + '\'' for x in secs_temp)
            q = sqls_config['cbond_call_clause']['Sql']
            q = q % (secs_temp_str)
            call_clause_temp = dq.read_sql(q)
            call_clause_temp.drop_duplicates(inplace=True)
            call_clause = pd.concat([call_clause, call_clause_temp])
        self.call_clause = call_clause

    def get_remain_amount(self):
        """
        TODO:
        提取可转债未转股余额(万元)，该数据不定期公布需要做填充
        若数据为空说明上市以来未发生过转股及规模变动，则余额取发行额
        :return:
        """
        print("获取转债余额...")
        CBbond_holdings = self.CBbond_holdings # 持仓量：F_MOUNT
        # date_list = CBbond_holdings['D_DATE'].unique()
        remain_amount = pd.DataFrame()
        # 获取最早上市日期
        list_date_min = int(self.basic_info['list_date'].min().strftime('%Y%m%d'))
        # 最后持仓日期
        end_date = CBbond_holdings['d_date'].apply(lambda x: int(date_utils.strftime(x, '%Y%m%d'))).min()
        duration_days = date_utils.TradeDays.get_adjustdate(list_date_min, end_date, ('D', 1))
        sec_list = CBbond_holdings['code'].unique().tolist()

        # for temp_date in date_list:
        # sec_list = CBbond_holdings.loc[CBbond_holdings['D_DATE'] == temp_date, 'code'].unique().tolist()
        loop_num = int(np.ceil(len(sec_list) / 1000))
        for i in range(loop_num):
            secs_temp = sec_list[i * 1000: min((i + 1) * 1000, len(sec_list))]
            secs_temp_str = ','.join('\'' + str(x.split('.')[0]) + '\'' for x in secs_temp)
            # 可转债代码, 可转债名称， 证券市场, 可转债剩余金额， 截止日期
            q = sqls_config['cbond_remain_amount']['Sql']
            q = q.format(secs_temp_str=secs_temp_str, list_date_min=list_date_min, end_date=end_date)
            remain_amount_temp = dq.read_sql(q)
            remain_amount_temp.drop_duplicates(inplace=True)
            # remain_amount_temp['date'] = temp_date
            remain_amount = pd.concat([remain_amount, remain_amount_temp])
        # 填充
        remain_amount_mat = remain_amount.pivot_table(values='remainingamount', index='enddate', columns='cbond_code')
        remain_amount_mat.fillna(method='ffill', inplace=True)
        remain_amount_holdings = remain_amount_mat.reindex(index=pd.to_datetime(duration_days['trade_dt'].astype('str')),
                                                      columns = split_market_code(sec_list)).iloc[-1, :].to_frame('remain')
        # 未发生过转股的，转债余额用发行规模替代
        issue_size = self.basic_info[['cbond_innercode', 'cbond_code', 'actualissuesize']]
        remain_amount_holdings = pd.merge(remain_amount_holdings, issue_size, left_index=True, right_on='cbond_code', how='left')
        remain_amount_holdings['remain_amount'] = remain_amount_holdings['remain'].fillna(remain_amount_holdings['actualissuesize'])
        remain_amount_holdings = remain_amount_holdings.reindex(columns=['cbond_innercode', 'remain_amount'])
        remain_amount_holdings = pd.merge(self.basic_info[['cbond_innercode', 'cbond_code', 'cbond_market']],
                                            remain_amount_holdings, on='cbond_innercode', how='left')
        remain_amount_holdings['end_date'] = date_utils.strptime(str(end_date), '%Y%m%d')
        self.remain_amount_holdings = remain_amount_holdings

    def get_turnover(self):
        """
        提取可转债换手率
        专户到期天数在 X:\1. 基础数据\产品一览表
        wind turnover: 成交量（单位是元）/债券余额（元）
        :return:
        """
        print("获取换手率...")
        CBbond_holdings = self.CBbond_holdings
        date_list = CBbond_holdings['d_date'].unique()
        if len(date_list) > 1:
            raise ValueError("持仓数据中D_DATE超过一天！")
        turnover = pd.DataFrame()
        # 获取最早上市日期
        list_date_min = int(self.basic_info['list_date'].min().strftime('%Y%m%d'))
        # 最后持仓日期
        end_date = CBbond_holdings['d_date'].apply(lambda x: int(date_utils.strftime(x, '%Y%m%d'))).min()
        sec_list = CBbond_holdings['code'].unique().tolist()
        # 最近十个交易日日期
        # temp_datenum = int(pd.to_datetime(temp_date).strftime('%Y%m%d'))
        # temp_datenum_offset = date_utils.TradeDays.offset_date(int(temp_datenum),
        #                                                     None, -10)[0]
        duration_days = date_utils.TradeDays.get_adjustdate(list_date_min, end_date, ('D', 1))
        loop_num = int(np.ceil(len(sec_list) / 1000))
        for i in range(loop_num):
            secs_temp = sec_list[i * 1000: min((i + 1) * 1000, len(sec_list))]
            secs_temp_str = ','.join('\'' + str(x.split('.')[0]) + '\'' for x in secs_temp)
            # 可转债内部代码、代码, 可转债名称， 证券市场, 换手率（%），成交量（张）， 交易日
            q = sqls_config['cbond_turnover']['Sql']
            q = q.format(secs_temp_str=secs_temp_str, list_date_min=list_date_min, end_date=end_date)
            turnover_temp = dq.read_sql(q)
            turnover_temp.drop_duplicates(inplace=True)
            turnover = pd.concat([turnover, turnover_temp])
        # 计算近10个交易日均值
        turnover_mat = turnover.pivot_table(index='tradingday', columns='cbond_code', values='turnoverrate')
        turnover_mat = turnover_mat.reindex(index=pd.to_datetime(duration_days['trade_dt'].astype('str')),
                                            columns=split_market_code(sec_list))
        turnover_mean_mat = turnover_mat.rolling(window=10, min_periods=10).mean() # 取最后一个非nan即为最近10个交易日换手
        # 成交量
        volume_mat = turnover.pivot_table(index='tradingday', columns='cbond_code', values='turnovervolume')
        volume_mat = volume_mat.reindex(index=pd.to_datetime(duration_days['trade_dt'].astype('str')),
                                            columns=split_market_code(sec_list))
        volume_mean_mat = volume_mat.rolling(window=10, min_periods=10).mean()
        
        # 算法1： 取最近10个交易日均值(复牌不足10日的为nan)
        # turnover_mean = turnover_mean_mat.dropna(how='all', axis=1).apply(
        #                                         lambda x: x.dropna().iloc[-1]).to_frame('turnover_mean')
        # 特别处理：复牌超过5天不足10天的情况，turnover_mean为nan
        # last_tradedays_count = turnover_mat.notnull().rolling(window=10).sum(min=1)  # 最近10个交易日天数
        # turnover_mat_fp = turnover_mat.loc[:,
        #                   turnover_mat.iloc[-1, :].notnull() & (last_tradedays_count.iloc[-1, :] >= 5) & (
        #                       turnover_mean.iloc[-1, :].isnull())]

        # turnover.columns = ['InnerCode', 'SecuCode', 'SecuAbbr', 'SecuMarket','BondNature', 'TurnoverRate', 'EndDate']

        # 算法2： 取有交易的10个连续交易日换手(%)均值
        turnover_mat_notnull = turnover_mat.loc[:, turnover_mean_mat.notnull().sum() > 0] # 剔除连续上市不足10日、行情数据不足10日的
        turnover_mean = turnover_mat_notnull.apply(
            lambda x: x.dropna().rolling(window=10).mean().dropna().iloc[-1]).to_frame('turnover_mean').reset_index()
        # 已上市但无行情（如私募EB），换手率数据为0
        logger.info(f"以下转债行情数据不足10日:{list(turnover_mat.loc[:, turnover_mat.notnull().sum() < 10].columns)},"
              f"将取可得行情计算换手率10日均值")
        turnover_temp = turnover_mat.loc[:,
                             turnover_mat.notnull().sum() < 10].mean().fillna(0).to_frame('turnover_mean').reset_index()
        # 2021/4/29：上市10个交易日内不做监控
        if turnover_temp.shape[0] > 0:
            AShareCalendar = date_utils.TradeDays.get_AShareCalendar()
            AShareCalendar['trade_datetime'] = AShareCalendar['trade_dt'].apply(
                lambda x: date_utils.strptime(str(int(x)), '%Y%m%d'))
            turnover_temp = pd.merge(turnover_temp, self.basic_info[['cbond_code', 'list_date']], on='cbond_code', how='left')
            turnover_temp['list_date_idx'] = pd.Index(AShareCalendar['trade_datetime']).get_indexer(turnover_temp['list_date'])
            turnover_temp['d_date_idx'] = pd.Index(AShareCalendar['trade_datetime']).get_indexer(date_list)[0]
            turnover_temp['list_duration'] = turnover_temp['d_date_idx'] - turnover_temp['list_date_idx']
            turnover_mean = pd.concat(
                [turnover_mean, turnover_temp.loc[turnover_temp['list_duration'] >= 10, ['cbond_code', 'turnover_mean']]])
        
        # 成交量
        volume_mat_notnull = volume_mat.loc[:, volume_mean_mat.notnull().sum() > 0] # 剔除连续上市不足10日、行情数据不足10日的
        volume_mean = volume_mat_notnull.apply(
            lambda x: x.dropna().rolling(window=10).mean().dropna().iloc[-1]).to_frame('volume_mean').reset_index()
        volume_temp = volume_mat.loc[:,
                             volume_mat.notnull().sum() < 10].mean().fillna(0).to_frame('volume_mean').reset_index()
        volume_mean = pd.concat([volume_mean, volume_temp])
        self.volume_mean = volume_mean.reindex(columns=['cbond_code', 'volume_mean'])
        self.turnover_mean = turnover_mean

    def cal_monetize_days(self):
        """
        计算变现天数 = 持仓量 / （流通股本 * 换手率均值 * 0.05）
        :return:
        """
        print("计算变现天数...")
        CBbond_holdings = self.CBbond_holdings[['d_date', 'c_fullname', 'c_fundname', 'c_subname_bsh', 'f_mount',
                                                'f_asset', 'f_assetratio', 'code']]
        CBbond_holdings = CBbond_holdings.rename(columns={'code': 'cbond_windcode'})
        CBbond_holdings['cbond_code'] = split_market_code(CBbond_holdings['cbond_windcode'].values)

        remain_amount = self.remain_amount_holdings
        turnover_mean = self.turnover_mean
        volume_mean = self.volume_mean
        # 转债余额（元）
        df = pd.merge(CBbond_holdings, remain_amount[['cbond_code', 'end_date', 'remain_amount']],
                      left_on=['d_date', 'cbond_code'], right_on=['end_date', 'cbond_code'], how='left').drop(columns=['end_date'])
        # 换手率（%)
        df = pd.merge(df, turnover_mean, on='cbond_code', how='left')
        df = pd.merge(df, volume_mean, on='cbond_code', how='left')
        # 计算变现天数 = 持仓量 / （流通股本 * 换手率均值 * 0.05）
        # 2022/11/30: 参与度5%调整为15%
        participate = self.participate
        df['monetize_days'] = df['f_mount'] / ((df['remain_amount']/100) * df['turnover_mean'] / 100 * participate)
        df['monetize_days'] = df['monetize_days'].replace(float('inf'), 999) # 将换手率为0的变现天数设为999
        df.sort_values(by=['d_date', 'c_fundname', 'c_subname_bsh'], inplace=True)
        self.monetize_days = df

    def get_deriv_indicators(self):
        """
        获取可转债衍生指标（转股溢价率、平价、转股价值等）
        :return:
        """
        print("获取可转债衍生指标（转股溢价率、转股价值等）...")
        CBbond_holdings = self.CBbond_holdings
        date_list = CBbond_holdings['d_date'].unique()
        cb_deriv_indicators = pd.DataFrame()
        for temp_date in date_list:
            sec_list = CBbond_holdings.loc[CBbond_holdings['d_date'] == temp_date, 'code'].unique().tolist()
            loop_num = int(np.ceil(len(sec_list) / 1000))
            for i in range(loop_num):
                secs_temp = sec_list[i * 1000: min((i + 1) * 1000, len(sec_list))]
                secs_temp_str = ','.join('\'' + str(x.split('.')[0]) + '\'' for x in secs_temp)
                q = sqls_config['cbond_deriv_indicators']['Sql']
                q = q.format(secs_temp_str=secs_temp_str, date=pd.to_datetime(temp_date).strftime('%Y%m%d'))
                cb_deriv_indicators_temp = dq.read_sql(q)
                cb_deriv_indicators_temp.drop_duplicates(inplace=True)
                # cb_deriv_indicators_temp['date'] = temp_date
                cb_deriv_indicators = pd.concat([cb_deriv_indicators, cb_deriv_indicators_temp])
        cb_deriv_indicators.rename(columns={'tradingday': 'end_date'}, inplace=True)
        cb_deriv_indicators['cbond_windcode'] = convert_market_code(cb_deriv_indicators['cbond_market'])
        cb_deriv_indicators['cbond_windcode'] = cb_deriv_indicators.apply(lambda x: x['cbond_code'] + x['cbond_windcode'], axis=1)
        # 数据清洗：剔除重复数据中大宗平台交易数据
        duplicated_code = cb_deriv_indicators.loc[
                                cb_deriv_indicators['cbond_code'].duplicated()].sort_values(by='cbond_code')['cbond_code'].to_list()
        cb_deriv_indicators_filt = cb_deriv_indicators[cb_deriv_indicators['cbond_code'].isin(duplicated_code) & (
            cb_deriv_indicators['a_cbond_market'].isin([83, 90]))]
        cb_deriv_indicators_filt = pd.concat([cb_deriv_indicators_filt,
                                              cb_deriv_indicators[~cb_deriv_indicators['cbond_code'].isin(duplicated_code)]])
        cb_deriv_indicators_filt = cb_deriv_indicators_filt.sort_values(by='cbond_code')
        self.deriv_indicators = cb_deriv_indicators_filt

    def monitor_special_clause_risk(self):
        """
        监控特殊条款风险
        :return:
        """
        print("监控特殊条款风险...")
        # 1. 获取转债回售条款
        call_clause = self.call_clause
        remain_amount = self.remain_amount_holdings
        call_condition = pd.merge(call_clause, remain_amount.drop(columns=['cbond_code']), on='cbond_innercode',
                                  how='left')
        # 判断当前余额是否超过强赎条款最低余额
        # 是否处于赎回期
        call_condition['if_in_call_period'] = (call_condition['end_date'] >= call_condition['callstartdate']) & \
                                               (call_condition['end_date'] <= call_condition['callenddate'])
        # 是否触发剩余规模不足
        call_condition['if_call_by_amount'] = ((call_condition['remain_amount'] / 10000)
                                               < call_condition['callunconvertamount']) #& (call_condition['if_in_call_period'])
        self.call_condition = call_condition
        # 转债映射到组合
        # CBbond_holdings = self.CBbond_holdings[
        #     ['D_DATE', 'C_FUNDNAME', 'C_SUBNAME_BSH', 'code']].copy()
        # CBbond_holdings.rename(columns={'C_SUBNAME_BSH': 'CBOND_NAME', 'code': 'CBOND_WINDCODE'}, inplace=True)
        # CBbond_holdings['CBOND_CODE'] = split_market_code(CBbond_holdings['CBOND_WINDCODE'])
        res_special_clause_risk = self.call_condition[[
            'cbond_code', 'remain_amount', 'callunconvertamount', 'end_date', 'if_call_by_amount']].rename(
            columns={'end_date': 'd_date', 'callunconvertamount': '触发阈值(万)', 'remain_amount': '转债余额(万)'})
        res_special_clause_risk['转债余额(万)'] = res_special_clause_risk['转债余额(万)'] / 10000
        # res_special_clause_risk = pd.merge(CBbond_holdings, res_special_clause_risk, on=['D_DATE', 'CBOND_CODE'], how='left')
        res_special_clause_risk = self._merge_with_holdings(res_special_clause_risk)
        res_special_clause_risk['risk_type'] = '特殊条款风险_转债余额不足'
        res_special_clause_risk['risk_type'] = '特殊条款风险_转债余额不足'
        res_special_clause_risk['风险详情'] = res_special_clause_risk.apply(lambda x: f"市场存量余额:{round(x['转债余额(万)'])}万，存量过低",
                                                                        axis=1)
        res_special_clause_risk['风险详情'] = res_special_clause_risk['风险详情'].where(
            res_special_clause_risk['if_call_by_amount'])
        res_special_clause_risk.sort_values(by=['d_date', 'c_fundname', 'cbond_windcode'], inplace=True)
        return res_special_clause_risk.drop(columns=['cbond_code']).drop_duplicates()

    def monitor_repay_risk(self):
        """
        监控专户兑付风险: 转债变现天数 > 专户剩余天数 (剔除公募产品)
        :return:
        """
        print("监控专户兑付风险...")
        monetize_days = self.monetize_days.copy()
        product_outline = self.product_outline.copy()
        product_outline['产品到期日'] = pd.to_datetime(product_outline['产品到期日'])
        # CBbond_holdings = self.CBbond_holdings[
        #     ['D_DATE', 'C_FULLNAME', 'C_FUNDNAME', 'C_SUBNAME_BSH', 'F_MOUNT', 'code']]
        # CBbond_holdings_prodname  = CBbond_holdings['C_FULLNAME'].drop_duplicates()
        # 专户产品名称
        # CBbond_holdings_prodname_zh = CBbond_holdings_prodname.loc[
        #     CBbond_holdings_prodname.isin(product_outline['产品名称'].drop_duplicates().to_list())]
        # pd.merge(CBbond_holdings, product_outline[['产品名称', '产品到期日']], left_on='C_FULLNAME', right_on='产品名称', how='left')
        df = pd.merge(monetize_days, product_outline[['产品名称', '产品到期日']],
                      left_on='c_fullname', right_on='产品名称', how='left')
        # 剔除公募产品
        df = df[df['产品到期日'].notnull()]
        # 交易日算法
        # df['到期日_tdate'] = df['产品到期日'].apply(lambda x: int(date_utils.strftime(x, '%Y%m%d')))
        # df['到期日_tdate'] = date_utils.TradeDays.offset_date(df['到期日_tdate'].values, None, 0)
        # AShareCalendar = date_utils.TradeDays.get_AShareCalendar()
        # df['到期日_num'] = pd.Index(AShareCalendar['trade_dt']).get_indexer(df['到期日_tdate'])
        # df['评估日_num'] = pd.Index(AShareCalendar['trade_dt']).get_indexer(
        #     df['D_DATE'].apply(lambda x: int(date_utils.strftime(x, '%Y%m%d'))))
        # df['time_to_maturity'] = df['到期日_num'] - df['评估日_num']
        # 日历日算法
        df['time_to_maturity'] = df.apply(lambda x: (x['产品到期日'] - x['d_date']).days, axis=1)
        df['if_monetizedays_exceeds_maturity'] = df['monetize_days'] > df['time_to_maturity']
        df.rename(columns={'time_to_maturity': '距离到期时间', 'c_subname_bsh': 'cbond_name',
                            'monetize_days': '变现天数', 'cbond_windcode': 'cbond_windcode',
                            'volume_mean': '市场近10个交易日成交量'}, inplace=True)
        res = df.loc[:,
                     ['d_date', 'c_fundname', 'cbond_name', 'cbond_windcode', '市场近10个交易日成交量', '变现天数', 'f_asset', 'f_assetratio',
                      '产品到期日', '距离到期时间', 'if_monetizedays_exceeds_maturity']].drop_duplicates()
        res.rename(columns={'time_to_maturity': '距离到期时间', 'c_subname_bsh': 'cbond_name',
                            'monetize_days': '变现天数', 'cbond_windcode': 'cbond_windcode'}, inplace=True)
        res.sort_values(by=['d_date', 'c_fundname', 'cbond_windcode'], inplace=True)
        res['risk_type'] = '专户兑付风险'
        res['风险详情'] = np.nan
        res.loc[res['变现天数'].notnull(), '风险详情'] = res.loc[res['变现天数'].notnull()].apply(
            lambda x: f"近10日日均成交量: {round(x['市场近10个交易日成交量'] / 10000 * 100)}万，以{int(self.participate * 100)}%参与度计算，可参与日均成交量为{round(x['市场近10个交易日成交量'] / 10000 * 100 * self.participate)}万，变现需{round(x['变现天数'])}天\n专户剩余到期日：{x['距离到期时间']}天",
            axis=1)
        res['风险详情'] = res['风险详情'].where(res['if_monetizedays_exceeds_maturity'])
        self.repay_risk_data = df
        return res

    def eval_cond_redeem(self):
        """
        判断是否满足条件赎回
        :return:
        """
        print("判断是否满足条件赎回...")
        # 转换条件，赎回开始、结束日和转换开始、结束日基本一致（日历日和交易日区别）
        if not hasattr(self, 'call_condition'):
            self.monitor_special_clause_risk()
        call_condition = self.call_condition.copy()
        CBbond_holdings = self.CBbond_holdings.copy()
        # 获取最早上市日期
        list_date_min = int(self.basic_info['list_date'].min().strftime('%Y%m%d'))

        # 1. 获取行情
        # 最后持仓日期
        end_date = CBbond_holdings['d_date'].apply(lambda x: int(date_utils.strftime(x, '%Y%m%d'))).min()
        sec_list = CBbond_holdings['code'].unique().tolist()
        duration_days = date_utils.TradeDays.get_adjustdate(list_date_min, end_date, ('D', 1))
        cond_redeem = pd.DataFrame()
        loop_num = int(np.ceil(len(sec_list) / 1000))
        for i in range(loop_num):
            secs_temp = sec_list[i * 1000: min((i + 1) * 1000, len(sec_list))]
            secs_temp_str = ','.join('\'' + str(x.split('.')[0]) + '\'' for x in secs_temp)
            q = sqls_config['cbond_exchg_quote']['Sql']
            q = q .format(secs_temp_str=secs_temp_str, list_date_min=list_date_min, end_date=end_date)

            cond_redeem_temp = dq.read_sql(q)
            cond_redeem_temp.drop_duplicates(inplace=True)
            cond_redeem = pd.concat([cond_redeem, cond_redeem_temp])
        # conv_price = pd.pivot_table(cond_redeem, index='TRADINGDAY', columns='')
        cond_redeem = pd.merge(cond_redeem, self.basic_info[['cbond_innercode', 'stk_code']], on='cbond_innercode', how='left')
        cbond_close_mat = pd.pivot_table(cond_redeem, values='cbond_close', index='tradingday', columns='cbond_code')
        cbond_stkprice_mat = pd.pivot_table(cond_redeem, values='stk_price', index='tradingday', columns='cbond_code')
        cbond_convert_price_mat = pd.pivot_table(cond_redeem, values='convert_price', index='tradingday', columns='cbond_code')
        # index 对齐
        cbond_code_list = [str(x.split('.')[0]) for x in sec_list]
        cbond_close_mat = cbond_close_mat.reindex(columns=cbond_code_list,
                                                  index=pd.to_datetime(duration_days['trade_dt'].astype('str')))
        cbond_stkprice_mat = cbond_stkprice_mat.reindex(columns=cbond_code_list,
                                                  index=pd.to_datetime(duration_days['trade_dt'].astype('str')))
        cbond_convert_price_mat = cbond_convert_price_mat.reindex(columns=cbond_code_list,
                                                  index=pd.to_datetime(duration_days['trade_dt'].astype('str')))
        # 2.逐个判断转换条件
        call_condition['reach_con_call_days'] = np.nan
        call_condition['if_call_by_price'] = np.nan

        for i in range(0, len(cbond_code_list)):
            print(f"判断是否满足条件赎回: {i} / {len(cbond_code_list) - 1}")
            cbond_i = cbond_code_list[i]
            call_i = call_condition.loc[call_condition['cbond_code'] == cbond_i]  # 赎回条款
            call_level_i = call_i['calllevel'].values[0]
            call_cond_day_i = call_i['callconditionday'].values[0]
            call_reachday_i = call_i['callreachday'].values[0]
            if np.isnan(call_cond_day_i):
                continue
            # 股票价格不低于转股价格 call_level 倍
            cond_1_i = (cbond_stkprice_mat >= (cbond_convert_price_mat * call_level_i))
            # 在赎回期内
            if_in_callperiod = cond_1_i[cbond_i].to_frame().apply(lambda x:
                                                                  (x.index >= call_i['callstartdate'].values[0]) &
                                                                  (x.index <= call_i['callenddate'].values[0]))
            # 在赎回期且满足赎回条件
            reach_days_i = (cond_1_i[cbond_i] * if_in_callperiod.iloc[:, 0].values).rolling(
                window=int(call_cond_day_i)).sum().iloc[-1]
            if_call_i = reach_days_i >= call_reachday_i
            call_condition.loc[call_condition['cbond_code'] == cbond_i, 'if_call_by_price'] = if_call_i
            call_condition.loc[call_condition['cbond_code'] == cbond_i, 'reach_con_call_days'] = reach_days_i

        # 摘牌日期
        call_condition = pd.merge(call_condition, self.basic_info[['cbond_code', 'list_enddate']], on='cbond_code',
                                  how='left')
        # 在转换期内、退市之前，满足条件赎回
        call_condition['if_call_by_price_adj'] = call_condition['if_call_by_price'] & \
                                                 (call_condition['if_in_call_period'] &
                                                  (~(call_condition['end_date'] > call_condition['list_enddate'])))
        call_condition_all = call_condition.copy()

        self.call_condition_all = call_condition_all
        self.cond_redeem = cond_redeem
        pass

    def _merge_with_holdings(self, bond_risk_data, on_col=['d_date', 'cbond_code'], how='left'):
        """
        将转债数据映射到持仓
        :param bond_risk_data:
        :return:
        """
        CBbond_holdings = self.CBbond_holdings[
            ['d_date', 'c_fundname', 'c_subname_bsh', 'f_asset', 'f_assetratio', 'code']].copy()
        CBbond_holdings.rename(columns={'c_subname_bsh': 'cbond_name', 'code': 'cbond_windcode'}, inplace=True)
        CBbond_holdings['cbond_code'] = split_market_code(CBbond_holdings['cbond_windcode'])
        df = pd.merge(CBbond_holdings, bond_risk_data, on=on_col,
                      how=how)
        df.drop_duplicates(inplace=True)
        return df

    def _cal_risk_pct(self, bond_risk_data: pd.DataFrame, cond_colname: str):
        """
        计算特定风险个券占组合净值比例
        :param: bond_risk_data['D_DATE', 'CBOND_CODE', cond_colname]
        :return:
        """
        # 若有重复数据，报错
        if bond_risk_data.drop_duplicates().duplicated(subset=['d_date', 'cbond_code']).sum() > 0:
            raise ValueError(f"风险券数据有重复项")
        df = self._merge_with_holdings(bond_risk_data[bond_risk_data[cond_colname]])
        df.sort_values(by=['d_date', 'c_fundname', 'cbond_code'], inplace=True)
        df['risk_ratio'] = df[cond_colname].fillna(0.0) * df['f_assetratio'] / 100 # F_ASSETRATIO为占净值比例（%）
        df['risk_mv'] = df[cond_colname].fillna(0.0) * df['f_asset'] # 市值
        # risk_pct = df.groupby(['C_FUNDNAME', 'D_DATE'])['risk_ratio'].sum().to_frame('risk_pct').reset_index().sort_values(by='C_FUNDNAME')
        risk_pct = df.groupby(['c_fundname', 'd_date'])[['risk_ratio', 'risk_mv']].sum().reset_index().sort_values(by='c_fundname')
        return risk_pct, df.drop(columns=['cbond_code'])


    def _monitor_redeem_impale_risk(self):
        """
        2. 强赎条款刺破溢价： 满足条件赎回条件，且转股溢价率高于15%【占组合净值比例大于10%】
        :return:
        """
        self.eval_cond_redeem()
        call_condition_all = self.call_condition_all
        deriv_indicators = self.deriv_indicators # 转股溢价率、转股价值数据
        cond_redeem = self.cond_redeem
        call_cond_columns = ['cbond_code', 'calllevel', 'callconditionday', 'callreachday', 'callunconvertamount',
                             'if_call_by_amount', 'if_in_call_period', 'reach_con_call_days', 'if_call_by_price_adj']
        redeem_risk = pd.merge(deriv_indicators, call_condition_all[call_cond_columns],
                               on='cbond_code', how='left')
        redeem_risk = pd.merge(redeem_risk,
                               cond_redeem[['cbond_code', 'cbond_close', 'tradingday']].rename(
                                columns={'tradingday': 'end_date'}), on=['cbond_code', 'end_date'], how='left')
        redeem_risk['if_redeem_impale_premium'] = redeem_risk['if_call_by_price_adj'] & (
                    redeem_risk['convertpremiumrate'] > 15)
        redeem_risk.rename(columns={'end_date': 'd_date'}, inplace=True)
        res_redeem_impale_risk, res_redeem_impale_risk_detail = self._cal_risk_pct(redeem_risk[['d_date', 'cbond_code', 'if_redeem_impale_premium']],
                                             'if_redeem_impale_premium')
        res_redeem_impale_risk['risk_type'] = '强赎条款刺破溢价风险'
        res_redeem_impale_risk_detail['risk_type'] = '强赎条款刺破溢价风险'
        self.redeem_risk = redeem_risk
        return res_redeem_impale_risk, res_redeem_impale_risk_detail

    def _monitor_premium_shattered_risk(self):
        """
        3. 市场风险下的溢价破灭风险: 转股溢价率大于30， 且平价>120【占组合净值比例大于10%】
        :return:
        """
        redeem_risk = self.redeem_risk
        redeem_risk['if_premium_shattered'] = (redeem_risk['convertpremiumrate'] > 30) & (
                redeem_risk['cbconvertvalue'] > 120)
        res_premium_shattered_risk, res_premium_shattered_risk_detail = self._cal_risk_pct(redeem_risk[['d_date', 'cbond_code', 'if_premium_shattered']],
                                             'if_premium_shattered')
        res_premium_shattered_risk['risk_type'] = '溢价破灭风险'
        res_premium_shattered_risk_detail['risk_type'] = '溢价破灭风险'
        self.redeem_risk = redeem_risk
        return res_premium_shattered_risk, res_premium_shattered_risk_detail

    def _monitor_liquidity_risk(self):
        """
        4. 低流动性风险: 可转债变现天数 > 10天【占组合净值比例大于10%】
        :return:
        """
        liquidity_risk = self.monetize_days.copy()
        liquidity_risk['if_illiquidity'] = liquidity_risk['monetize_days'] > 10
        # # 20220414: 修复bug
        df = liquidity_risk.copy()
        df.sort_values(by=['d_date', 'c_fundname', 'cbond_code'], inplace=True)
        cond_colname = 'if_illiquidity'
        df['risk_ratio'] = df[cond_colname].fillna(0.0) * df['f_assetratio'] / 100 # F_ASSETRATIO为占净值比例（%）
        df['risk_mv'] = df[cond_colname].fillna(0.0) * df['f_asset'] # 市值
        risk_pct = df.groupby(['c_fundname', 'd_date'])[['risk_ratio', 'risk_mv']].sum().reset_index().sort_values(by='c_fundname')

        res_liquidity_risk_detail_columns = ['d_date', 'c_fundname', 'cbond_name', 'f_asset', 'f_assetratio', 'cbond_windcode', 'if_illiquidity', 'risk_ratio', 'risk_mv']
        df = df.rename(columns={'c_subname_bsh': 'cbond_name'}).reindex(columns=res_liquidity_risk_detail_columns)

        res_liquidity_risk = risk_pct.copy()
        res_liquidity_risk_detail = df.copy()

        # res_liquidity_risk, res_liquidity_risk_detail = self._cal_risk_pct(liquidity_risk[['d_date', 'cbond_code', 'if_illiquidity']],
        #                                      'if_illiquidity')
        res_liquidity_risk['risk_type'] = '低流动性风险'
        res_liquidity_risk_detail['risk_type'] = '低流动性风险'
        return res_liquidity_risk, res_liquidity_risk_detail

    def _monitor_speculate_risk(self):
        """
        5. 投机炒作风险：近10个交易日平均换手率>50%【占组合净值比例大于10%】
        :return:
        """
        speculate_risk = self.monetize_days.copy()
        speculate_risk['if_speculate'] = speculate_risk['turnover_mean'] > 50
        res_speculate_risk, res_speculate_risk_detail = self._cal_risk_pct(speculate_risk[['d_date', 'cbond_code', 'if_speculate']],
                                             'if_speculate')
        res_speculate_risk['risk_type'] = '投机炒作风险'
        res_speculate_risk_detail['risk_type'] = '投机炒作风险'
        return res_speculate_risk, res_speculate_risk_detail

    def _monitor_premium_agg(self):
        """
        强赎条款刺破溢价+溢价破灭合并
        :return:
        """
        redeem_risk = self.redeem_risk
        # 两种风险满足其中之一
        redeem_risk['if_premium_agg'] = (redeem_risk['if_redeem_impale_premium'] | redeem_risk['if_premium_shattered'])
        res_premium_risk_agg, res_premium_risk_agg_detail = self._cal_risk_pct(redeem_risk[['d_date', 'cbond_code', 'if_premium_agg']],
                                             'if_premium_agg')
        res_premium_risk_agg['risk_type'] = '溢价风险（合并）'
        res_premium_risk_agg_detail['risk_type'] = '溢价风险（合并）'
        self.redeem_risk = redeem_risk
        return res_premium_risk_agg, res_premium_risk_agg_detail

    def monitor_invest_risk(self):
        """
        监控投资风险
        :return:
        """
        risk_ratio_all = pd.DataFrame()
        risk_ratio_detail_all = pd.DataFrame()
        # 1. 专户兑付风险
        res_repay_risk = self.monitor_repay_risk()

        # 2. 强赎条款刺破溢价： 满足条件赎回条件，且转股溢价率高于15%【占组合净值比例大于10%】
        res_redeem_impale_risk, res_redeem_impale_risk_detail = self._monitor_redeem_impale_risk()
        risk_ratio_all = risk_ratio_all.append(res_redeem_impale_risk)
        risk_ratio_detail_all = risk_ratio_detail_all.append(res_redeem_impale_risk_detail.drop(columns=['if_redeem_impale_premium']))

        # 3. 市场风险下的溢价破灭风险: 转股溢价率大于30， 且平价>120【占组合净值比例大于10%】
        res_premium_shattered_risk, res_premium_shattered_risk_detail = self._monitor_premium_shattered_risk()
        risk_ratio_all = risk_ratio_all.append(res_premium_shattered_risk)
        risk_ratio_detail_all = risk_ratio_detail_all.append(
            res_premium_shattered_risk_detail.drop(columns=['if_premium_shattered']))

        # 3.5 强赎条款刺破溢价或溢价破灭合并（组合风险提示）
        res_premium_risk_agg, res_premium_risk_agg_detail = self._monitor_premium_agg()
        risk_ratio_all = risk_ratio_all.append(res_premium_risk_agg)
        risk_ratio_detail_all = risk_ratio_detail_all.append(res_premium_risk_agg_detail.drop(columns=['if_premium_agg']))

        # 4. 低流动性风险: 可转债变现天数 > 10天【占组合净值比例大于10%】
        res_liquidity_risk, res_liquidity_risk_detail = self._monitor_liquidity_risk()
        risk_ratio_all = risk_ratio_all.append(res_liquidity_risk)
        risk_ratio_detail_all = risk_ratio_detail_all.append(
            res_liquidity_risk_detail.drop(columns=['if_illiquidity']))

        # 5. 投机炒作风险：近10个交易日平均换手率>50%【占组合净值比例大于10%】
        res_speculate_risk, res_speculate_risk_detail = self._monitor_speculate_risk()
        risk_ratio_all = risk_ratio_all.append(res_speculate_risk)
        risk_ratio_detail_all = risk_ratio_detail_all.append(
            res_speculate_risk_detail.drop(columns=['if_speculate']))

        # 是否触发
        risk_ratio_all['是否触发'] = risk_ratio_all['risk_ratio'] > 0.1
        risk_ratio_all['风险详情'] = np.nan
        risk_ratio_all.loc[risk_ratio_all['risk_type'] == '溢价风险（合并）', '风险详情'] = '高溢价率转债持仓总量过高'
        risk_ratio_all.loc[risk_ratio_all['risk_type'] == '低流动性风险', '风险详情'] = '低流动性可转债持仓总量过高'
        risk_ratio_all.loc[risk_ratio_all['risk_type'] == '投机炒作风险', '风险详情'] = '高换手率转债持仓总量过高'
        risk_ratio_all['风险详情'] = risk_ratio_all['风险详情'].where(risk_ratio_all['是否触发'])
        risk_ratio_all = risk_ratio_all[['d_date', 'c_fundname', 'risk_ratio', 'risk_mv', '是否触发', 'risk_type', '风险详情']]
        return res_repay_risk, risk_ratio_all, risk_ratio_detail_all

    def monitor_sec_redeem_risk(self):
        """
        监控个券赎回风险: 距离强赎登记日5日内或变现天数超过专户剩余天数【注意公募产品无变现天数】
        :return:
        """
        AShareCalendar = date_utils.TradeDays.get_AShareCalendar()
        AShareCalendar['trade_datetime'] = AShareCalendar['trade_dt'].apply(
                                                    lambda x: date_utils.strptime(str(int(x)), '%Y%m%d'))
        redeem_ann = dq.read_sql(sqls_config['cbond_redeem_announce']['Sql'])
        redeem_ann.rename(columns={'secuabbr': 'cbond_name', 'secucode': 'cbond_code', 'callregdate': '赎回登记日'},
                          inplace=True)
        redeem_ann = redeem_ann[['cbond_code', '赎回登记日']]
        df = self._merge_with_holdings(redeem_ann, on_col=['cbond_code'])
        df.dropna(subset=['赎回登记日'], inplace=True)
        # 计算距离赎回天数【交易日】
        df['赎回登记日_num'] = pd.Index(AShareCalendar['trade_datetime']).get_indexer(df['赎回登记日'])
        df['ddate_num'] = pd.Index(AShareCalendar['trade_datetime']).get_indexer(df['d_date'])
        df['剩余天数（交易日）'] = (df['赎回登记日_num'] - df['ddate_num'])
        res_sec_redeem_risk = df.drop(columns=['赎回登记日_num', 'ddate_num'])

        # 新增：补充最近10个交易日成交量及变现天数
        # res_sec_redeem_risk = pd.merge(res_sec_redeem_risk, self.volume_mean.rename(columns={'volume_mean': '市场近10个交易日成交量'}),
        #                                on='CBOND_CODE', how='left')
        repay_risk_data = self.repay_risk_data[['d_date', 'c_fundname', 'cbond_name', '市场近10个交易日成交量', '变现天数',
                                                '距离到期时间', 'if_monetizedays_exceeds_maturity']]
        res_sec_redeem_risk = pd.merge(res_sec_redeem_risk, repay_risk_data, on=['d_date', 'c_fundname', 'cbond_name'], how='left')
        if res_sec_redeem_risk.shape[0] > 0:
            # 专户产品（有剩余天数字段的）
            if res_sec_redeem_risk.loc[res_sec_redeem_risk['变现天数'].notnull()].shape[0] > 0:
                res_sec_redeem_risk.loc[res_sec_redeem_risk['变现天数'].notnull(), '风险详情'] = res_sec_redeem_risk.loc[
                        res_sec_redeem_risk['变现天数'].notnull()].apply(lambda
                        x: f"强制赎回日：{date_utils.strftime(x['赎回登记日'], '%Y/%m/%d')}\n剩余天数： {x['剩余天数（交易日）']}\n市场近10个交易日日均成交量: {round(x['市场近10个交易日成交量'] / 10000 * 100)}万,以{int(self.participate * 100)}%参与度计算,可参与日均成交量为{round(x['市场近10个交易日成交量'] / 10000 * 100 * self.participate)}万,变现需{round(x['变现天数'])}天\n专户剩余到期日：{x['距离到期时间']}天",
                        axis=1)

            # 公募产品（变现天数数据缺失的）
            if res_sec_redeem_risk.loc[res_sec_redeem_risk['变现天数'].isnull()].shape[0] > 0:
                res_sec_redeem_risk.loc[res_sec_redeem_risk['变现天数'].isnull(), '风险详情'] = res_sec_redeem_risk.loc[
                        res_sec_redeem_risk['变现天数'].isnull()].apply(lambda
                        x: f"强制赎回日：{date_utils.strftime(x['赎回登记日'], '%Y/%m/%d')}\n剩余天数： {x['剩余天数（交易日）']}",
                        axis=1)
        res_sec_redeem_risk = res_sec_redeem_risk.sort_values(by=['d_date', 'c_fundname', 'cbond_name'])
        self.res_sec_redeem_risk = res_sec_redeem_risk
        return res_sec_redeem_risk

    def _get_rc_alert_cb_port(self):
        """
        获取组合风险提示
        """
        port_risk = pd.DataFrame()
        # 1. 赎回风险
        # CBond_个券强赎风险
        res_sec_redeem_risk = self.res_sec_redeem_risk.copy()
        cond = (res_sec_redeem_risk['剩余天数（交易日）'] <= 5)# | (res_sec_redeem_risk['if_monetizedays_exceeds_maturity'])
        res_sec_redeem_risk['if_sec_reddem_risk'] = cond
        df_risk = res_sec_redeem_risk[cond]
        if df_risk.shape[0] > 0:
            df_risk['risk_type'] = '赎回风险'
            port_risk_i = df_risk.reindex(columns=['d_date', 'risk_type', 'c_fundname', 'cbond_name', 'cbond_windcode',
                                                   'f_asset', 'f_assetratio', '风险详情'])
            port_risk_i = port_risk_i.rename(columns={'风险详情': 'risk_info'})
            port_risk_i['f_assetratio'] = port_risk_i['f_assetratio'] / 100
            port_risk = pd.concat([port_risk, port_risk_i])

        # 2. 兑付风险
        res_repay_risk = self.res_repay_risk.copy()
        df_risk = res_repay_risk.dropna(subset=['风险详情'])
        if df_risk.shape[0] > 0:
            df_risk = df_risk.copy()
            df_risk.loc[:, 'risk_type'] = '兑付风险'
            port_risk_i = df_risk.reindex(columns=['d_date', 'risk_type', 'c_fundname', 'cbond_name', 'cbond_windcode',
                                                   'f_asset', 'f_assetratio', '风险详情'])
            port_risk_i = port_risk_i.rename(columns={'风险详情': 'risk_info'})
            port_risk_i['f_assetratio'] = port_risk_i['f_assetratio'] / 100
            port_risk = pd.concat([port_risk, port_risk_i])

        # 3. 市场存量不足（sheet: CBond_特殊条款风险_转债余额不足）
        res_special_clause_risk = self.res_special_clause_risk.copy()
        df_risk = res_special_clause_risk[res_special_clause_risk['是否触发']]
        if df_risk.shape[0] > 0:
            df_risk['risk_type'] = '市场存量不足'
            port_risk_i = df_risk.reindex(columns=['d_date', 'risk_type', 'c_fundname', 'cbond_name', 'cbond_windcode',
                                                   'f_asset', 'f_assetratio', '风险详情'])
            port_risk_i = port_risk_i.rename(columns={'风险详情': 'risk_info'})
            port_risk_i['f_assetratio'] = port_risk_i['f_assetratio'] / 100
            port_risk = pd.concat([port_risk, port_risk_i])

        # 4. 溢价破灭风险(CBond_风险比例, 刺破溢价 + 溢价破灭之和， 超过10%展示)
        risk_ratio_all = self.risk_ratio_all.copy()
        cond = (risk_ratio_all['risk_type'] == '溢价风险（合并）') & (risk_ratio_all['是否触发'])
        df_risk = risk_ratio_all[cond]
        if df_risk.shape[0] > 0:
            df_risk['risk_type'] = '溢价破灭风险'
            port_risk_i = df_risk.reindex(columns=['d_date', 'risk_type', 'c_fundname', 'cbond_name', 'cbond_windcode',
                                                   'risk_mv', 'risk_ratio', '风险详情']).rename(columns={'risk_mv': 'f_asset', 'risk_ratio': 'f_assetratio'})
            port_risk_i = port_risk_i.rename(columns={'风险详情': 'risk_info'})
            port_risk_i[['cbond_name', 'cbond_windcode']] = '-'
            port_risk = pd.concat([port_risk, port_risk_i])

        # 5.低流动性风险
        risk_ratio_all = self.risk_ratio_all.copy()
        cond = (risk_ratio_all['risk_type'] == '低流动性风险') & (risk_ratio_all['是否触发'])
        df_risk = risk_ratio_all[cond].copy()
        if df_risk.shape[0] > 0:
            df_risk.loc[:, 'risk_type'] = '低流动性风险'
            port_risk_i = df_risk.reindex(columns=['d_date', 'risk_type', 'c_fundname', 'cbond_name', 'cbond_windcode',
                                                   'risk_mv', 'risk_ratio', '风险详情']).rename(columns={'risk_mv': 'f_asset', 'risk_ratio': 'f_assetratio'})
            port_risk_i = port_risk_i.rename(columns={'风险详情': 'risk_info'})
            port_risk_i[['cbond_name', 'cbond_windcode']] = '-'
            port_risk = pd.concat([port_risk, port_risk_i])

        # 6. 炒作风险
        risk_ratio_all = self.risk_ratio_all.copy()
        cond = (risk_ratio_all['risk_type'] == '投机炒作风险') & (risk_ratio_all['是否触发'])
        df_risk = risk_ratio_all[cond]
        if df_risk.shape[0] > 0:
            df_risk['risk_type'] = '炒作风险'
            port_risk_i = df_risk.reindex(columns=['d_date', 'risk_type', 'c_fundname', 'cbond_name', 'cbond_windcode',
                                                   'risk_mv', 'risk_ratio', '风险详情']).rename(columns={'risk_mv': 'f_asset', 'risk_ratio': 'f_assetratio'})
            port_risk_i = port_risk_i.rename(columns={'风险详情': 'risk_info'})
            port_risk_i[['cbond_name', 'cbond_windcode']] = '-'
            port_risk = pd.concat([port_risk, port_risk_i])

        # 持仓占比转换为小数
        # port_risk['f_assetratio'] = port_risk['f_assetratio'] / 100
        # 列转换为大写
        port_risk = port_risk.reindex(columns=['d_date', 'risk_type', 'c_fundname', 'cbond_name', 'cbond_windcode',
       'f_asset', 'f_assetratio', 'risk_info'])
        port_risk.columns = list(map(lambda x: x.upper(), port_risk.columns))
        if port_risk.shape[0] > 0:
            port_risk['D_DATE'] = port_risk['D_DATE'].apply(lambda x: date_utils.strftime(x, '%Y-%m-%d'))
        return port_risk




    # def monitor_price_pct_bias(self):
    #     cm = CbondMarket(self.basedate, self.basedate)
    #     # 全市场异常转债
    #     market_bias_upward, market_bias_downward = cm.cal_price_bias()
    #     cbond_holdings = self.CBbond_holdings.rename(columns={'d_date': 'tradingday', 'code': 'cbond_windcode'})
    #     cbond_holdings['if_hold'] = 1
    #     market_bias_upward = pd.merge(market_bias_upward, cbond_holdings[['tradingday', 'cbond_windcode', 'if_hold']],
    #              on=['tradingday', 'cbond_windcode'], how='left')
    #     market_bias_downward = pd.merge(market_bias_downward,
    #                                     cbond_holdings[['tradingday', 'cbond_windcode', 'if_hold']],
    #                             on=['tradingday', 'cbond_windcode'], how='left')
    #
    #     # 起始日期：前
    #     beg_date = str(date_utils.TradeDays.offset_date(
    #         CBbond_holdings['d_date'].apply(lambda x: int(date_utils.strftime(x, '%Y%m%d'))).max(), -1)[0])
    #
    #     pass


    def CalculateAll(self):
        # 2. 特殊条款风险
        res_special_clause_risk = self.monitor_special_clause_risk()
        # 3. 投资风险
        res_repay_risk, risk_ratio_all, risk_ratio_detail_all = self.monitor_invest_risk()
        # 1. 个券赎回风险
        res_sec_redeem_risk = self.monitor_sec_redeem_risk()
        # 4. 2022/8/16：转债新规下新增监控项（见 \25. 部门研究\量化组\2. 风险管理\风险指标及因子\可转债风险指标）

        # 一些中间数据
        repay_risk_data = self.repay_risk_data.rename(columns={
                                'if_monetizedays_exceeds_maturity': '是否变现天数超过专户剩余天数'}) # 平均换手率、变现天数等
        redeem_risk_data = self._merge_with_holdings(self.redeem_risk.drop(
                            columns=['cbond_innercode', 'cbond_name','cbond_market', 'a_cbond_market', 'cbond_windcode'])) # 转换价值、转股溢价率、是否满足赎回条件等
        self.res_special_clause_risk = res_special_clause_risk.rename(columns={'if_call_by_amount': '是否触发'})
        self.res_repay_risk = res_repay_risk.rename(columns={'if_monetizedays_exceeds_maturity': '是否触发'})
        self.risk_ratio_all = risk_ratio_all
        self.risk_ratio_detail_all = risk_ratio_detail_all
        self.repay_risk_data = repay_risk_data.drop(columns=['产品名称']).sort_values(by=['c_fullname', 'cbond_code'])
        redeem_risk_data.rename(columns={'if_call_by_amount': '是否满足余额不足赎回条件',
                                         'if_in_call_period': '是否处于赎回期',
                                         'reach_con_call_days': '满足条件赎回天数',
                                         'if_call_by_price_adj': '是否满足价格赎回条件',
                                         'if_redeem_impale_premium': '是否触发强赎条款刺破溢价风险',
                                         'if_premium_shattered': '是否触发溢价破灭风险',
                                         'if_premium_agg': '是否触发溢价风险（合计）'}, inplace=True)
        redeem_risk_data = redeem_risk_data.dropna(subset=['callconditionday'])
        redeem_risk_data['赎回条件'] = redeem_risk_data.apply(lambda
                                      x: f"已达到：过去{x['callconditionday']}个交易日，有{x['callreachday']}"
                                         f"个交易日的收盘价不低于当期转股价格的{int(x['calllevel'] * 100)}%",
                                                          axis=1)
        redeem_risk_data['赎回条件'] = redeem_risk_data['赎回条件'].where(redeem_risk_data['是否满足价格赎回条件'])
        self.redeem_risk_data = redeem_risk_data.sort_values(by=['c_fundname', 'cbond_code'])


        if hasattr(self, 'special_clause'):
            self.special_clause_all = self._merge_with_holdings(self.special_clause, on_col=['cbond_code'], how='inner')
            self.special_clause_all = self.special_clause_all.drop(columns=['f_assetratio', 'cbond_code', 'cbond_windcode'])
            pass


    def _get_rc_cb_sec_redeem(self):
        res_sec_redeem_risk = self.res_sec_redeem_risk.copy()
        cond = (res_sec_redeem_risk['剩余天数（交易日）'] <= 5)# | (res_sec_redeem_risk['if_monetizedays_exceeds_maturity'])
        res_sec_redeem_risk['if_sec_redeem_risk'] = cond
        # 处理成落库表格式
        mapping = {'赎回登记日': 'redeem_rec_date',
                   '剩余天数（交易日）': 'remaining_days',
                   '市场近10个交易日成交量': 'trading_vol_10d', '变现天数': 'liq_days',
                   '距离到期时间': 'time_to_maturity', '风险详情': 'risk_info',
                   'if_monetizedays_exceeds_maturity': 'if_exceeds_mat'}
        res_sec_redeem_risk = res_sec_redeem_risk.rename(columns=mapping)
        cols = ['d_date', 'c_fundname', 'cbond_name', 'cbond_windcode', 'f_asset', 'f_assetratio', 'redeem_rec_date',
                  'remaining_days', 'trading_vol_10d', 'liq_days', 'time_to_maturity', 'if_exceeds_mat',
                  'risk_info', 'if_sec_redeem_risk']
        df_risk = res_sec_redeem_risk.reindex(columns=cols)
        df_risk = df_risk.copy()
        df_risk.loc[: , ['d_date', 'redeem_rec_date']] = df_risk[['d_date', 'redeem_rec_date']].applymap(
                                                                        lambda x: date_utils.strftime(x, '%Y-%m-%d'))
        # 持仓占比转换为小数
        # df_risk = df_risk[df_risk['if_sec_redeem_risk']]
        df_risk['f_assetratio'] = df_risk['f_assetratio'] / 100
        df_risk.columns = list(map(lambda x: x.upper(), df_risk.columns))
        # 数量类型调整
        df_risk['IF_EXCEEDS_MAT'] = df_risk['IF_EXCEEDS_MAT'].astype(float)
        # df_risk = df_risk.convert_dtypes(convert_integer=False).copy()
        return df_risk

    def _get_rc_cb_clause(self):
        res_special_clause_risk = self.res_special_clause_risk.copy()
        mapping = {'转债余额(万)': 'remain_amount',
                   '触发阈值(万)': 'trigger_threshold',
                   '是否触发': 'if_trigger',
                   '风险详情': 'risk_info'}
        res_sec_redeem_risk = res_special_clause_risk.rename(columns=mapping)
        cols = ['d_date', 'c_fundname', 'cbond_name', 'cbond_windcode', 'f_asset', 'f_assetratio',
                'remain_amount', 'trigger_threshold', 'if_trigger', 'risk_type', 'risk_info']
        df_risk = res_sec_redeem_risk[cols]
        df_risk[['d_date']] = df_risk[['d_date']].applymap(lambda x: date_utils.strftime(x, '%Y-%m-%d'))
        # 持仓占比转换为小数
        # df_risk = df_risk[df_risk['if_trigger']].copy()
        df_risk['f_assetratio']  = df_risk['f_assetratio'] / 100
        df_risk.columns = list(map(lambda x: x.upper(), df_risk.columns))
        # df_risk = df_risk.convert_dtypes(convert_integer=False).copy()
        return df_risk

    def _get_rc_cb_repay(self):
        res_repay_risk = self.res_repay_risk.copy()
        # res_repay_risk['trading_amount_10d_5pct'] = res_repay_risk['市场近10个交易日成交量'] * 100 * 0.05
        res_repay_risk['trading_amount_10d_5pct'] = np.nan
        mapping = {'市场近10个交易日成交量': 'avg_trading_vol_10d',
                   '变现天数': 'liq_days',
                   '产品到期日': 'maturity',
                   '距离到期时间': 'time_to_maturity',
                   '是否触发': 'if_trigger',
                   '风险详情': 'risk_info'}
        res_repay_risk = res_repay_risk.rename(columns=mapping)
        cols = ['d_date', 'c_fundname', 'cbond_name', 'cbond_windcode', 'f_asset', 'f_assetratio',
                'avg_trading_vol_10d', 'liq_days', 'maturity', 'time_to_maturity',
                'if_trigger', 'risk_type', 'risk_info', 'trading_amount_10d_5pct']
        df_risk = res_repay_risk.loc[:, cols]
        df_risk = df_risk.copy()
        df_risk[['d_date', 'maturity']] = df_risk[['d_date', 'maturity']].applymap(lambda x: date_utils.strftime(x, '%Y-%m-%d'))
        # 持仓占比转换为小数
        # df_risk = df_risk[df_risk['if_trigger']]
        df_risk['f_assetratio'] = df_risk['f_assetratio'] / 100
        df_risk.columns = list(map(lambda x: x.upper(), df_risk.columns))
        # df_risk = df_risk.convert_dtypes(convert_integer=False).copy()
        return df_risk

    def _get_rc_cb_risk_ratio(self):
        risk_ratio_all = self.risk_ratio_all.copy()
        mapping = {'是否触发': 'if_trigger', '风险详情': 'risk_info'}
        risk_ratio_all = risk_ratio_all.rename(columns=mapping)
        cols = ['d_date', 'c_fundname', 'risk_ratio', 'risk_mv', 'if_trigger', 'risk_type', 'risk_info']
        df_risk = risk_ratio_all.loc[:, cols]
        df_risk = df_risk.copy()
        df_risk[['d_date']] = df_risk[['d_date']].applymap(lambda x: date_utils.strftime(x, '%Y-%m-%d'))
        df_risk = df_risk[df_risk['risk_ratio'] > 0]
        df_risk.columns = list(map(lambda x: x.upper(), df_risk.columns))
        # df_risk = df_risk.convert_dtypes(convert_integer=False).copy()
        return df_risk

    def _get_rc_cb_risk_ratio_detail(self):
        """
        只存储 risk_ratio>0 的数据
        """
        risk_ratio_detail_all = self.risk_ratio_detail_all.copy()
        cols = ['d_date', 'c_fundname', 'cbond_name', 'cbond_windcode', 'f_asset', 'f_assetratio',
                'risk_ratio', 'risk_mv', 'risk_type']
        df_risk = risk_ratio_detail_all.loc[:, cols]
        df_risk = df_risk.copy()
        df_risk[['d_date']] = df_risk[['d_date']].applymap(lambda x: date_utils.strftime(x, '%Y-%m-%d'))
        # 持仓占比转换为小数
        df_risk = df_risk[df_risk['risk_ratio'] > 0]
        df_risk['f_assetratio'] = df_risk['f_assetratio'] / 100
        df_risk.columns = list(map(lambda x: x.upper(), df_risk.columns))
        df_risk[['RISK_RATIO', 'RISK_MV']] = df_risk[['RISK_RATIO', 'RISK_MV']].astype(float)
        # df_risk = df_risk.convert_dtypes(convert_integer=False).copy()
        return df_risk

    def _get_rc_cb_deriv_data(self):
        redeem_risk_data = self.redeem_risk_data.copy()
        # 处理成落库表格式
        mapping = {'是否满足余额不足赎回条件': 'if_remainmont_insufficient',
                   '是否处于赎回期': 'if_in_redeem_period',
                   '满足条件赎回天数': 'reach_days',
                   '是否满足价格赎回条件': 'if_reach_price_cond',
                   '是否触发强赎条款刺破溢价风险': 'if_redeem_impale_premium',
                   '是否触发溢价破灭风险': 'if_premium_shattered',
                   '是否触发溢价风险（合计）': 'if_premium_agg',
                   '赎回条件': 'redeem_con_desc'}
        redeem_risk_data = redeem_risk_data.rename(columns=mapping)
        cols = ['d_date', 'c_fundname', 'cbond_name', 'cbond_windcode', 'f_asset', 'f_assetratio',
                'cbconvertvalue', 'convertpremiumrate', 'calllevel', 'callconditionday', 'callreachday',
                'callunconvertamount', 'if_remainmont_insufficient', 'if_in_redeem_period', 'reach_days',
                'if_reach_price_cond', 'cbond_close', 'if_redeem_impale_premium', 'if_premium_shattered',
                'if_premium_agg', 'redeem_con_desc']
        df_risk = redeem_risk_data[cols]
        df_risk = df_risk.copy()
        df_risk[['d_date']] = df_risk[['d_date']].applymap(lambda x: date_utils.strftime(x, '%Y-%m-%d'))
        # 持仓占比转换为小数
        df_risk['f_assetratio'] = df_risk['f_assetratio'] / 100
        df_risk[['if_remainmont_insufficient', 'if_in_redeem_period', 'if_reach_price_cond',
                 'if_redeem_impale_premium', 'if_premium_shattered',
                'if_premium_agg']] = df_risk[['if_remainmont_insufficient', 'if_in_redeem_period', 'if_reach_price_cond',
                 'if_redeem_impale_premium', 'if_premium_shattered',
                'if_premium_agg']].astype(float)
        df_risk.columns = list(map(lambda x: x.upper(), df_risk.columns))
        # df_risk = df_risk.convert_dtypes(convert_integer=False).copy()
        return df_risk

    def _get_rc_cb_liqdays_data(self):
        """
        添加两个字段：if_liq_risk：变现天数超过10天
        """
        monetize_days = self.monetize_days.copy()
        # monetize_days['trading_amount_10d_5pct'] = monetize_days['volume_mean'] * 100 * 0.05
        monetize_days['trading_amount_10d_5pct'] = np.nan
        monetize_days['if_liq_risk'] = monetize_days['monetize_days'] > 10
        # 处理成落库表格式
        mapping = {'c_subname_bsh': 'cbond_name',
                   'turnover_mean': 'turnover_mean_10D',
                   'volume_mean': 'volume_mean_10D',
                   'monetize_days': 'liq_days'}
        monetize_days = monetize_days.rename(columns=mapping)
        cols = ['d_date', 'c_fundname', 'cbond_name', 'cbond_windcode', 'f_asset', 'f_assetratio',
                'f_mount', 'remain_amount', 'turnover_mean_10D', 'volume_mean_10D', 'liq_days',
                'trading_amount_10d_5pct', 'if_liq_risk']
        df_risk = monetize_days[cols]
        df_risk = df_risk.copy()
        df_risk[['d_date']] = df_risk[['d_date']].applymap(lambda x: date_utils.strftime(x, '%Y-%m-%d'))
        # 持仓占比转换为小数
        df_risk['f_assetratio'] = df_risk['f_assetratio'] / 100
        df_risk.columns = list(map(lambda x: x.upper(), df_risk.columns))
        # df_risk = df_risk.convert_dtypes(convert_integer=False).copy()
        return df_risk

    def insert_2_db(self):
        """
        插入到数据库
        """
        # 1. RC_ALERT_CB_PORT
        RC_ALERT_CB_PORT = self._get_rc_alert_cb_port()
        # 2. RC_CB_SEC_REDEEM (添加字段 if_sec_reddem_risk)
        RC_CB_SEC_REDEEM = self._get_rc_cb_sec_redeem()
        # 3. RC_CB_CLAUSE(CBond_特殊条款风险_转债余额不足)
        RC_CB_CLAUSE = self._get_rc_cb_clause()
        # 4. RC_CB_REPAY(CBond_专户兑付风险(添加字段))
        RC_CB_REPAY = self._get_rc_cb_repay()
        # 5.RC_CB_RISK_RATIO(CBond_风险比例)
        RC_CB_RISK_RATIO = self._get_rc_cb_risk_ratio()
        # 6. RC_CB_RISK_RATIO_DETAIL(CBond_风险比例明细)
        RC_CB_RISK_RATIO_DETAIL = self._get_rc_cb_risk_ratio_detail()
        # 7. RC_CB_DERIV_DATA(CBond_中间数据_衍生指标)
        RC_CB_DERIV_DATA = self._get_rc_cb_deriv_data()
        # 8.RC_CB_LIQDAYS_DATA(CBond_中间数据_变现天数,  (添加字段))
        RC_CB_LIQDAYS_DATA = self._get_rc_cb_liqdays_data()
        table_names = ['RC_ALERT_CB_PORT', 'RC_CB_SEC_REDEEM', 'RC_CB_CLAUSE', 'RC_CB_REPAY',
                       'RC_CB_RISK_RATIO', 'RC_CB_RISK_RATIO_DETAIL', 'RC_CB_DERIV_DATA', 'RC_CB_LIQDAYS_DATA']
        datas = [RC_ALERT_CB_PORT, RC_CB_SEC_REDEEM, RC_CB_CLAUSE, RC_CB_REPAY,
                       RC_CB_RISK_RATIO, RC_CB_RISK_RATIO_DETAIL, RC_CB_DERIV_DATA, RC_CB_LIQDAYS_DATA]
        insert2db(table_name=table_names, res_list=datas, basedate=self.basedate)
        pass


    def saveAll(self, save_dir):
        # TODO: 若此项目日后改为用项目跑，即pycharm搭配虚拟环境，则下述输出文件的代码需用xlwings重新输出
        """
        输出时将asset_ratio 单位调整为 0.00】
        F_ASSET 单位调整为万
        :param writer:
        :return:
        """
        res_sec_redeem_risk = self.res_sec_redeem_risk.copy()
        if self.res_sec_redeem_risk.shape[0] > 0:
            res_sec_redeem_risk['持仓占比'] = res_sec_redeem_risk['f_assetratio'] / 100
            res_sec_redeem_risk['持仓市值'] = res_sec_redeem_risk.apply(lambda x: f"{round(x['f_asset'] / 10000)}万", axis=1)

        res_special_clause_risk = self.res_special_clause_risk.copy()
        res_special_clause_risk['持仓占比'] = res_special_clause_risk['f_assetratio'] / 100
        res_special_clause_risk['持仓市值'] = res_special_clause_risk.apply(lambda x: f"{round(x['f_asset'] / 10000)}万", axis=1)

        res_repay_risk = self.res_repay_risk.copy()
        res_repay_risk['持仓占比'] = res_repay_risk['f_assetratio'] / 100
        res_repay_risk['持仓市值'] = res_repay_risk.apply(
            lambda x: f"{round(x['f_asset'] / 10000)}万", axis=1)

        risk_ratio_all = self.risk_ratio_all.copy()
        risk_ratio_all['risk_mv（万）'] = risk_ratio_all.apply(lambda x: f"{round(x['risk_mv'] / 10000)}万", axis=1)

        risk_ratio_detail_all = self.risk_ratio_detail_all.copy()
        risk_ratio_detail_all['持仓占比'] = risk_ratio_detail_all['f_assetratio'] / 100
        risk_ratio_detail_all['持仓市值'] = risk_ratio_detail_all.apply(lambda x: f"{round(x['f_asset'] / 10000)}万", axis=1)
        risk_ratio_detail_all['risk_mv（万）'] = risk_ratio_detail_all.apply(
            lambda x: f"{round(x['risk_mv'] / 10000)}万", axis=1)

        repay_risk_data = self.repay_risk_data.copy()
        repay_risk_data['持仓占比'] = repay_risk_data['f_assetratio'] / 100
        repay_risk_data['持仓市值'] = repay_risk_data.apply(lambda x: f"{round(x['f_asset'] / 10000)}万", axis=1)

        redeem_risk_data = self.redeem_risk_data.copy()
        redeem_risk_data['持仓占比'] = redeem_risk_data['f_assetratio'] / 100
        redeem_risk_data['持仓市值'] = redeem_risk_data.apply(lambda x: f"{round(x['f_asset'] / 10000)}万", axis=1)

        monetize_days = self.monetize_days.copy()
        monetize_days['持仓占比'] = monetize_days['f_assetratio'] / 100


        writer = pd.ExcelWriter(save_dir)
        res_sec_redeem_risk.to_excel(writer, sheet_name='CBond_个券强赎风险')
        res_special_clause_risk.to_excel(writer, sheet_name='CBond_特殊条款风险_转债余额不足')
        # if hasattr(self, 'special_clause_all'):
        #     self.special_clause_all.to_excel(writer, sheet_name='CBond_其他特殊条款【新增转债】')
        res_repay_risk.to_excel(writer, sheet_name='CBond_专户兑付风险')
        risk_ratio_all.to_excel(writer, sheet_name='CBond_风险比例')
        risk_ratio_detail_all.to_excel(writer, sheet_name='CBond_风险比例明细')
        repay_risk_data.to_excel(writer, sheet_name='CBond_中间数据_兑付风险')
        redeem_risk_data.to_excel(writer, sheet_name='CBond_中间数据_衍生指标')
        monetize_days.to_excel(writer, sheet_name='CBond_中间数据_变现天数')
        # writer.save()
        writer.close()


        # 写入excel
        # app = xw.App(visible=False)
        # wb = app.books.add()  # 新建workbook
        # ws = wb.sheets.add('CBond_个券强赎风险')
        # ws.range('A1').value = self.res_sec_redeem_risk
        # ws = wb.sheets.add('CBond_特殊条款风险_转债余额不足')
        # ws.range('A1').value = self.res_special_clause_risk
        # if hasattr(self, 'special_clause_all'):
        #     ws = wb.sheets.add('CBond_其他特殊条款【新增转债】')
        #     ws.range('A1').value = self.special_clause_all
        # ws = wb.sheets.add('CBond_专户兑付风险')
        # ws.range('A1').value = self.res_repay_risk
        # ws = wb.sheets.add('CBond_风险比例')
        # ws.range('A1').value = self.risk_ratio_all
        # ws = wb.sheets.add('CBond_风险比例明细')
        # ws.range('A1').value = self.risk_ratio_detail_all
        # ws = wb.sheets.add('CBond_中间数据_兑付风险')
        # ws.range('A1').value = self.repay_risk_data
        # ws = wb.sheets.add('CBond_中间数据_衍生指标')
        # ws.range('A1').value = self.redeem_risk_data
        # ws = wb.sheets.add('CBond_中间数据_变现天数')
        # ws.range('A1').value = self.monetize_days
        # # excelname = save_path + '%s_RiskIndicators_CBOND.xlsx'%t.replace('-', '')
        # wb.save(save_dir)
        # app.quit()
        # pass
        





