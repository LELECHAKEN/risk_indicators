# @Time : 2021/8/9 11:23 
# @Author : for wangp
# @File : db_utils.py 
# @Software: PyCharm
import datetime
import numpy as np
import pandas as pd

from . import OracleDB, sqls_config
from ..settings import config
from ..utils import error_utils


db_quant = OracleDB(config['data_base']['QUANT']['url'])

def convert_columns(data):
    data.columns = [x.upper() if x.islower() else x for x in data.columns]
    return data

class JYDB_Query(object):
    def __init__(self):
        self._connect_database()

    def _connect_database(self):
        # 检查数据库地址是否配置
        if 'JY' not in config['data_base'].keys():
            raise error_utils.DoesNotExist('请在配置文件data_base中填写JY数据库地址')

        # 提取数据
        try:
            self.db = OracleDB(config['data_base']['JY']['url'])
        except Exception:
            raise error_utils.DBError('数据库连接失败！')

    def sec_query(self, sql_name, sec_list, *args):
        '''
        sql查询语句，统一化拆分代码为1000*n的结构，防止数据过大取不出来
        :param sql_name: sql.yaml中对应的sql名称
        :param sec_list: 证券代码，类型为list
        :param flex_arg: 无名的灵活参数，传入sql的第二个参数，string即可。故在传入时不能给名称
        :return: DataFrame
        '''
        if type(sec_list) != list:
            sec_list = [sec_list]
        sec_list = list(set(sec_list))    #去重

        str_status = type(sec_list[0]) ==  str

        sec_data = pd.DataFrame()
        loop_num = int(np.ceil(len(sec_list)/1000))
        for i in range(loop_num):
            secs_temp = sec_list[i*1000: min((i+1)*1000, len(sec_list))]
            if str_status:
                secs_temp_str = ','.join('\'' + str(x.split('.')[0]) + '\'' for x in secs_temp)
            else:
                secs_temp_str = ','.join(str(x) for x in secs_temp)
            full_args = (secs_temp_str, ) + args
            q = sqls_config[sql_name]['Sql']%full_args
            sec_data_temp = self.db.read_sql(q)
            sec_data = pd.concat([sec_data, sec_data_temp])
        sec_data.columns = [x.upper() if x.islower() else x for x in sec_data.columns]

        return sec_data

    def sql_query(self, q):
        data = pd.read_sql(q, self.db)
        return data

    def close_query(self):
        self.db.close()

class WINDDB_Query(object):
    def __init__(self):
        self._connect_database()

    def _connect_database(self):
        # 检查数据库地址是否配置
        if 'WIND' not in config['data_base'].keys():
            raise error_utils.DoesNotExist('请在配置文件data_base中填写WIND数据库地址')

        # 提取数据
        try:
            self.db = OracleDB(config['data_base']['WIND']['url'])
        except Exception:
            raise error_utils.DBError('数据库连接失败！')

    def sec_query(self, sql_raw, sec_list, **kwargs):
        '''
        sql查询语句，统一化拆分代码为1000*n的结构，防止数据过大取不出来, SQL语句中代码列表必须命名为code
        :param sql_raw: sql语句
        :param sec_list: 证券代码，类型为list
        :param kwargs: sql语句中需要传入的其他参数
        :return: DataFrame
        '''
        if type(sec_list) != list:
            sec_list = [sec_list]
        sec_list = list(set(sec_list))    #去重

        sec_data = pd.DataFrame()
        loop_length = 1000
        loop_num = int(np.ceil(len(sec_list)/loop_length))
        for i in range(loop_num):
            code_part = sec_list[loop_length * i: loop_length * (i + 1)] \
                if i < loop_num - 1 else sec_list[loop_length * i: len(sec_list)]
            code_part = tuple(code_part) if len(code_part) > 1 else '(\'' + code_part[0] + '\')'
            data_part = self.db.read_sql(sql_raw.format(code=code_part, **kwargs))
            sec_data = pd.concat([sec_data, data_part])

        return sec_data

    def sql_query(self, q):
        data = pd.read_sql(q, self.db)
        return data

    def close_query(self):
        self.db.close()