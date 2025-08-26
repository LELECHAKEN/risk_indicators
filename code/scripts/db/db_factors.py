# -*- coding: utf-8 -*-
'''
# @Desc    :
# @Author  : zhouyy
# @Time    : 2021/5/21
'''
import pandas as pd
import numpy as np
from typing import Dict
from sqlalchemy import exc
import traceback
import datetime
from sqlalchemy import and_, column, or_
from . import (sqls_config, column)
from .relation_db import OracleDB
from .util import DoesNotExist
from ..utils import logger

market_mapping = {'JY2Wind': {83: '.SH', 90: '.SZ', 310: '.IOC', 72: '.HK', 18: '.BJ'},
                  'Wind2JY': {'SH': 83, 'SZ': 90, 'IOC': 310, 'HK': 72, 'BJ': 18}}

def convert_market_code(market_code, from_='JY', to_='Wind'):
    """
    交易市场代码切换
    :param market_code:
    :return:
    """
    # JY的SECUMARKET(83, 90) 转换为wind代码后缀 (.SH, .SZ)
    if (from_ == 'JY') & (to_ == 'Wind'):
        return pd.DataFrame(market_code).replace(market_mapping['JY2Wind']).values

    if (from_ == 'Wind') & (to_ == 'JY'):
        return pd.DataFrame(market_code).replace(market_mapping['Wind2JY']).values


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
    if (from_ == 'JY') & (to_ == 'Wind'):
        sec_code['wind_market'] = convert_market_code(sec_code[dict['jy_market']])
        sec_code['sec_code_adj'] = sec_code.apply(
            lambda x: x[dict['sec_code']][1:] if x['wind_market'] == '.HK' else x[dict['sec_code']], axis=1)
        res = sec_code.apply(
            lambda x: np.nan if pd.isnull(x['wind_market']) else str(x['sec_code_adj']) + str(x['wind_market']),
            axis=1).to_list()
        return res



class DbFactor(object):
    def __init__(self, log_info: Dict, factor_cfg: str = sqls_config):
        """
        初始化
        :param log_info:
        :param factor_cfg:
        """
        self.log_info = log_info
        self.cfg = factor_cfg
        # self.db_oper = DataQuery(self.log_info)
        self.db_oper = OracleDB(log_info)

    def close(self):
        '''关闭数据库资源'''
        self.db_oper.close()

    def _rebuild(self):
        '''新建DbOperation实例'''
        self.db_oper = OracleDB(self.log_info)

    def valid_code_length(self, codelist, maxnum=1000):
        if len(codelist) > maxnum:
            return False
        else:
            return True

    def read_sql2df(self, sql):
        try:
            df = self.db_oper.read_sql(sql)
        except Exception as e:
            # 若连接被关闭，则新建一个实例
            logger.info(f"提取数据库失败，error:{e}, 尝试重连...")
            self._rebuild()
            df = self.db_oper.read_sql(sql)
            logger.info("重新连接成功.")
        return df

    def get_factor_from_dw(self, factor_name: str, code_list='all', field='innercode', **kwargs):
        """
        从dw提取因子值
        :param factor_name:
        :param code_list:
        :param field:
        :param kwargs:
        :return:
        """
        sql = self.cfg[factor_name]['Sql']
        params = ""
        for k, v in kwargs.items():
            params += k.lower() + '=' + v + ','

        if code_list == '':
            pass
        else:
            if code_list == 'all':
                codes = ''
            elif type(code_list) is str:
                codes = " and " + field + "= '" + code_list +"'"
            elif type(code_list) is list:
                if self.valid_code_length(code_list):
                    codes = f" and {field} in ({','.join(repr(x) for x in code_list)})"
                else:
                    raise Exception('length of code exceeds maximum length of sql(1000)')
            else:
                raise TypeError(f'Type {type(code_list)} not support for code_list')
            if params == '':
                params = 'field="' + codes + '"'
            else:
                params = params[:-1] + ',field="' + codes + '"'

        sql_param = eval("sql.format(" + params + ")")
        try:
            df = self.db_oper.read_sql(sql_param)
        except Exception as e:
            # 若连接被关闭，则新建一个实例
            self._rebuild()
            df = self.db_oper.read_sql(sql_param)

        if 'Return' in self.cfg[factor_name].keys():
            df.columns = self.cfg[factor_name]['Return']
        return df

    def get_factor(self, factor_name: str, code_list: list, field='f_info_windcode', **kwargs):
        """
        从dw提取因子值, 支持长度超过1000的code_list
        :param factor_name:
        :param code_list:
        :param field:
        :param kwargs:
        :return:
        """
        print(f'=============get_factor_from_db:{factor_name}==============')
        if type(code_list) is list:
            code_list = list(set(code_list))
        if (type(code_list) is str) or ((type(code_list) is list) and self.valid_code_length(code_list)):
            df = self.get_factor_from_dw(factor_name, code_list, field, **kwargs)
        else:
            windcode_list_split = [code_list[i:i + 1000] for i in range(0, len(code_list), 1000)]
            df = pd.DataFrame()
            for windcode_list_i in windcode_list_split:
                df_i = self.get_factor_from_dw(factor_name, list(windcode_list_i), field, **kwargs)
                df = pd.concat([df, df_i])
        self.db_oper.close()
        return df.reset_index().drop('index', axis=1)


    def delete_table(self, table, cond, schema):
        """
        根据条件删除数据表
        :param table:
        :param cond:
        :param schema:
        :return:
        """
        # cond = or_(and_(column("name") == "zhangsan", column("age") == 66), and_(column("name") == "lisi", column("age") == 62))
        condition = cond
        try:
            self.db_oper.delete(table.lower(), condition, schema)
            logger.info(f"{table} 删除成功，条件为：{cond}")
        except (DoesNotExist, exc.NoSuchTableError):
            # logger.info(f"删除失败(数据可能不存在)：{table} ，条件为：{cond}, 错误为：{traceback.format_exc()} ")
            # logger.info(f"删除失败(数据可能不存在)：{table} ，条件为：{cond}")
            pass

    def insert_table(self, table: str, data: pd.DataFrame, schema: str, if_exists='append'):
        try:
            data = data.copy()
            data['insert_time'] = datetime.datetime.now()
            self.db_oper.insert_dataframe(table=table.lower(), data=data, schema=schema, if_exists=if_exists)
            logger.info(f"插入数据成功，table: {table}")
        except Exception as e:
            logger.error(f"数据插入失败，table：{table}, 错误为：{traceback.format_exc()}")

    def insert2db(self, table_name: list, res_list: list, cond, schema='quant'):
        """
        插入数据表, 按照条件删除数据，然后再插入
        """
        for table, data in zip(table_name, res_list):
            self.delete_table(table, cond, schema=schema)
            self.insert_table(table, data, schema=schema)
