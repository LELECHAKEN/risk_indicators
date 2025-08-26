# coding = utf-8
"""
Author: DaisyZhou

date: 2020/2/7 11:41
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys
sys.path.append('../')
from ..settings import config
from ..utils import error_utils
from ..db import OracleDB, sqls_config
from calendar import monthrange
P_DIR_OF_THIS_FILE = str(Path(__file__).resolve().parent)
_trade_date_path = os.path.join(P_DIR_OF_THIS_FILE, 'AShareCalendar.csv')


def validate_date(date: str, date_format="%Y-%m-%d"):
    """validate date is legal"""
    try:
        datetime.strptime(str(date), date_format)
    except ValueError:
        raise ValueError(f"Incorrect date format, should be {date_format}")

def convert_date(date: str, from_: str, to_: str):
    return datetime.strptime(str(date), from_).strftime(to_)

def _valid_n_convert(date, date_format="%Y-%m-%d", to_="%Y%m%d"):
    validate_date(date, date_format)
    return convert_date(date, date_format, to_)

def valid_n_convert(date, to_="%Y%m%d"):
    if isinstance(date, str):
        if all([len(date) > 8, ('-') in date]):
            return _valid_n_convert(date, date_format="%Y-%m-%d", to_=to_)
        elif all([len(date) > 8, ('/') in date]):
            return _valid_n_convert(date, date_format="%Y/%m/%d", to_=to_)
        elif len(date) == 8:
            return _valid_n_convert(date, date_format="%Y%m%d", to_=to_)
    elif isinstance(date, int) & (len(str(date)) == 8):
        validate_date(date, date_format="%Y%m%d")
        return str(int(date))
    else:
        raise ValueError("请输入正确的日期格式： 如：yyyy/mm/dd, yyyy-mm-dd")

def retrieve_n_tradeday(t, n):
    '''
    取给定日期过去第n个交易日日期
    :param t: string/datetime/timestamp, 需查询的给定日期
    :param n: int, 日期偏移量, 仅支持向历史偏移
    :return: string, 过去第n个交易日日期
    '''
    if type(t) != str:
        t = t.strftime('%Y-%m-%d')
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    q = sqls_config['past_tradeday']['Sql']%t
    tradeday = db_risk.read_sql(q).sort_values(by=['c_date'])
    return tradeday.iloc[(-1)*(n+1)][0]

def get_end_of_month(t: str, month_shift: int):
    '''
    计算距离当前日期前（后）n个月的月末日期

    Parameters
    ----------
    t: str, must be "YYYY-MM-DD"， 当前日期
    month_shift: 前n个月用负数表示，后n个月用正数表示

    Returns
    -------
    str, "YYYY-MM-DD"

    '''
    t0_date = datetime.strptime(t, "%Y-%m-%d")
    t1_year = t0_date.year + (t0_date.month + month_shift - 1) // 12
    t1_month = (t0_date.month + month_shift) % 12 if (t0_date.month + month_shift)%12 !=0 else 12
    t1_date = datetime(t1_year, t1_month, monthrange(t1_year, t1_month)[1])
    return t1_date.strftime("%Y-%m-%d")

def get_past_quarter(t: str, n: int):
    """
    获取过去n个季度的日期（如果ct_date为06-30，则包括当前季度）

    Parameters
    ----------
    t: str, must be "YYYY-MM-DD", 当前日期
    n: int, 过去n个季度的日期

    Returns
    -------
    list of str, "YYYY-MM-DD"

    """
    ct_date = datetime.strptime(t, "%Y-%m-%d")
    results = []
    if ct_date.month in [3, 6, 9, 12] and (ct_date.day == monthrange(ct_date.year, ct_date.month)[1]):
        results.append(t)
        for i in range(3, (n-1)*3+1, 3):
            results.append(get_end_of_month(t=t, month_shift=-i))
    else:
        ct_qtr = (ct_date.month - 1) // 3 + 1
        ct_qtr_date = datetime(ct_date.year, ct_qtr*3, monthrange(ct_date.year, ct_qtr*3)[1])
        ct_qtr_date = ct_qtr_date.strftime("%Y-%m-%d")
        for i in range(3, n*3+1, 3):
            results.append(get_end_of_month(t=ct_qtr_date, month_shift=-i))
    return results

def get_past_qtr_date(t: str, n: int):
    return get_past_quarter(t=t, n=n)[-1]

strptime = datetime.strptime
strftime = datetime.strftime

class TradeDays(object):
    """Trade date utilities"""

    def __init__(self):
        pass
    
    @classmethod
    def _download(cls):
        print('更新AShareCalendar文件...')
        try:
            date_shift = (datetime.now() + timedelta(days=20)).strftime('%Y%m%d')
            year_end = datetime(datetime.now().year, 12, 31).strftime('%Y%m%d')
            # 取到当年/20日后的年末
            end_date = date_shift if date_shift > year_end else year_end
            ssql = sqls_config['ashare_calendar']['Sql']
            ssql = ssql.format(beg_date='20000101', end_date=f"{end_date[:4]}1231")
        except Exception as e:
            raise error_utils.DoesNotExist("sqls.yaml中无sql语句: ashare_calendar")

        # 检查数据库地址是否配置
        if 'JY' not in config['data_base'].keys():
            raise error_utils.DoesNotExist('请在配置文件data_base中填写JY数据库地址')

        # 提取数据
        try:
            db = OracleDB(config['data_base']['JY']['url'])
            df = db.read_sql(ssql)
            db.close()
        except Exception:
            raise error_utils.DBError('数据库连接失败！')

        df.columns = list(map(lambda x: x.upper(), df.columns))
        df = df.rename(columns={'TradingDate': 'TRADE_DAYS'})
        df.drop(columns=['ID'], inplace=True)
        df.to_csv(_trade_date_path)
        return df


    @classmethod
    def get_AShareCalendar(cls, trade_date_path=_trade_date_path, res_type='int') -> pd.DataFrame:
        if not os.path.exists(trade_date_path):
            cls._download()
        AShareCalendar = pd.read_csv(trade_date_path, encoding='gbk', index_col=0)
        AShareCalendar.sort_values(by=['TRADINGDATE'], inplace=True)
        trade_dt = pd.DataFrame(AShareCalendar['TRADINGDATE'].values, columns=['trade_dt'])
        # 转成yyyymmdd数值型
        if res_type == 'int':
            trade_dt = pd.to_datetime(trade_dt['trade_dt']).apply(
                                    lambda x: datetime.strftime(x, '%Y%m%d')).astype('int').to_frame('trade_dt')
        elif res_type == 'datetime':
            trade_dt = pd.to_datetime(trade_dt['trade_dt'])

        if (pd.to_datetime(str(trade_dt.iloc[-1, 0])) - datetime.now() <= timedelta(days=20)) | \
                (datetime.now().month in [12, 1]):
            cls._download()
        return trade_dt

    @classmethod
    def resample_tradedate(cls, rule=('W', 'end')) -> pd.DataFrame:
        """
        将交易日转换成周度、月度等
        :param trade_dt: pd.DataFrame or pd.Series
        :param rule:('W','begin')->turple: 'W','M' + 'begin', 'end'
        :return:
        """

        trade_dt = cls.get_AShareCalendar(_trade_date_path)

        trade_date = trade_dt.copy()
        trade_date.columns = ['trade_dt']

        if isinstance(trade_date, pd.Series):
            trade_date = trade_date.to_frame()
        trade_date = trade_date.applymap(lambda x: datetime.strptime(str(x), '%Y%m%d')) #DataFrame


        if not isinstance(rule, tuple):
            raise ValueError("请输入正确的调仓模式 ('M','end')")

        if rule[0] == 'W':
            #获取每周最后一个交易日
            trade_date['period'] = trade_date.iloc[:, 0].apply(lambda x: x.weekday())
            if rule[1] == 'end':
                trade_date['diff'] = trade_date['period'].diff(-1)
                trade_date = trade_date.loc[trade_date['diff'] > 0, 'trade_dt'].to_frame()
            elif rule[1] == 'begin':
                trade_date['diff'] = trade_date['period'].diff(1)
                trade_date = trade_date.loc[trade_date['diff'] < 0, 'trade_dt'].to_frame()
        elif rule[0] == 'M':
            trade_date['period'] = trade_date.iloc[:,0].apply(lambda x: x.month)
            if rule[1] == 'end':
                trade_date['diff'] = trade_date['period'].diff(-1)
                trade_date = trade_date.loc[trade_date['diff'] != 0, 'trade_dt'].to_frame()
            elif rule[1] == 'begin':
                trade_date['diff'] = trade_date['period'].diff(1)
                trade_date = trade_date.loc[trade_date['diff'] != 0, 'trade_dt'].to_frame()
        elif rule[0] == 'Q':
            trade_date['period'] = trade_date.iloc[:,0].apply(lambda x: x.quarter)
            if rule[1] == 'end':
                trade_date['diff'] = trade_date['period'].diff(-1)
                trade_date = trade_date.loc[trade_date['diff'] != 0, 'trade_dt'].to_frame()
            elif rule[1] == 'begin':
                trade_date['diff'] = trade_date['period'].diff(1)
                trade_date = trade_date.loc[trade_date['diff'] != 0, 'trade_dt'].to_frame()
        else:
            raise ValueError("请输入正确resample参数")
        ##转换成yyyymmdd int型
        trade_date = trade_date['trade_dt'].apply(lambda x: int(datetime.strftime(x, '%Y%m%d')))
        trade_date = pd.DataFrame(trade_date.values, columns=['trade_dt'])
        return trade_date

    @classmethod
    def offset_date(cls, date_input: np.array,
                    n: int, mode='D', type='trade_date', if_modify=False, res_type='int') -> np.array:
        """
        对日期进行偏移(逻辑为前序找离当前日期最近的一个交易日，在此基础上进行偏移)
        :param date_input: 需要偏移的日期序列 array或Series, yyyymmdd int型,
        :param trade_dt_all: 所有交易日,dataframe或series
        :param n: 偏移量，0表示取离输入日期最近的一个交易日 或 当前所属（日历）月/季度末
        :param mode: 'D','W','M'
        :param type: trade_date, calendar, 按交易日、日历日
        :param if_modify: 若日期超出索引范围是否进行修正
        :return:
        """


        trade_dt_all = cls.get_AShareCalendar(_trade_date_path)

        if isinstance(date_input, int):
            date_input = np.array([date_input])
        elif isinstance(date_input, str):
            if (len(date_input) == 8):
                date_input = np.array([int(date_input)])
            else:
                raise ValueError("请输入正确的日期格式！")



        trade_dt_all.columns = ['trade_dt']

        date_offseted = []
        # date_input = np.array([20050207,20050710])
        # 格式转换成Series
        if isinstance(trade_dt_all, pd.DataFrame):
            trade_dt_all = trade_dt_all.iloc[:, 0]

        # 日期偏移
        if type == 'trade_date':
            if mode == 'D':
                # 离输入日期最近的一个交易日
                date_last = list(map(lambda x: trade_dt_all.values[trade_dt_all.values <= x][-1], list(date_input)))
                date_last_idx = pd.Index(trade_dt_all).get_indexer(date_last)
                adj_date = pd.DataFrame(trade_dt_all)
            elif (mode == 'W') | (mode == 'M') | (mode == 'Q'):
                # resample成对应周期
                adj_date = cls.resample_tradedate(trade_dt_all, (mode, 'end'))
                try:
                    date_last = list(map(lambda x: adj_date.values[adj_date.values <= x][-1], list(date_input)))
                    date_last_idx = pd.Index(adj_date.iloc[:,0].values).get_indexer(date_last) # adj_date为pd.DataFrame
                except IndexError:
                    print("日期偏移超过索引范围!")
            try:
                date_offseted = adj_date.iloc[date_last_idx + n, 0].values
                if res_type == 'datetime':
                    date_offseted = np.array(list(map(lambda x: datetime.strptime(str(x), '%Y%m%d'), date_offseted)))

            except IndexError as e:
                print("日期偏移超过索引范围!")
                new_idx = date_last_idx + n
                if np.where(new_idx > len(trade_dt_all) - 1)[0] > 0:
                    print('index: {} 超过最大日期!'.format(np.where(new_idx > len(trade_dt_all) - 1)[0][0]))

                if len(np.where(new_idx < 0)[0]) > 0:
                    print('index: {} 小于最小日期!'.format(np.where(new_idx < 0)[0][0]))

                if if_modify:
                    print("将进行日期修正！")
                    new_idx = np.where(new_idx > len(trade_dt_all) - 1, len(trade_dt_all) - 1,
                                       np.where(new_idx < 0, 0, new_idx))
                    # new_idx = np.where(new_idx < 0, 0, new_idx)
                    date_offseted = trade_dt_all.iloc[new_idx, 0]
        elif type == 'calendar':
            date_input = pd.DataFrame(date_input).applymap(lambda x: datetime.strptime(str(int(x)), '%Y%m%d'))
            if mode == 'D':
                date_offseted = date_input.applymap(lambda x: x + n*pd.tseries.offsets.DateOffset())
            elif (mode == 'M'):
                # date_last = date_input.applymap(lambda x: x - pd.tseries.offsets.MonthEnd())
                date_offseted = date_input.applymap(lambda x: x + n*pd.tseries.offsets.MonthEnd())
            elif (mode == 'Q'):
                # n=0表示当前所属季度末（包含当季度最后一天）
                # date_last = date_input.applymap(lambda x: x - pd.tseries.offsets.QuarterEnd())
                date_offseted = date_input.applymap(lambda x: x + n * pd.tseries.offsets.QuarterEnd())

            if res_type == 'int':
                #转换成yyyymmddint型
                date_offseted = date_offseted.applymap(lambda x: int(datetime.strftime(x, '%Y%m%d'))).iloc[:,0].values
            elif res_type == 'datetime':
                date_offseted = date_offseted.iloc[:,0].values

        return date_offseted


    @classmethod
    def get_adjustdate(cls, beg_date: int, end_date: int,
                       adj_mode=('M', 'end'), res_type='int') -> pd.DataFrame:
        """
        根据起始日，截至日，调仓模式确定调仓日
        :param trade_dt: yyyymmdd数值型, pd.DataFrame
        :param beg_date: yyyymmdd数值型
        :param end_date: yyyymmdd数值型
        :param adj_mode:
                         ('M','end'),('M','begin')
                         ('D', int),
                         ('W','end'),('W','begin')
                         ('custom', adj_date_arg)
        :return: adj_date -> Dataframe
        """
        if (beg_date == end_date) & (adj_mode == ('D', 1)):
            if res_type == 'datetime':
                return pd.Series(datetime.strptime(str(end_date), '%Y%m%d'))
            else:
                return pd.Series(end_date)

        trade_dt = cls.get_AShareCalendar(_trade_date_path)

        """调整测试起止时间"""
        try:
            beg_date_new = trade_dt[trade_dt >= beg_date].dropna().iloc[0, 0]
            end_date_new = trade_dt[trade_dt <= end_date].dropna().iloc[-1, 0]

            beg_date_newidx = np.where(trade_dt.iloc[:,0] == beg_date_new)[0][0]
            end_date_newidx = np.where(trade_dt.iloc[:,0] == end_date_new)[0][0]
        except Exception:
            raise ("获取测试起止日期出错, 现有数据起始日:{}，截止日:{}".format(trade_dt.iloc[0,0], trade_dt.iloc[-1,0]))


        """确定调仓日"""
        if (adj_mode[0] == 'M') | (adj_mode[0] == 'W'):
            # trade_dt = trade_dt.iloc[beg_date_newidx: end_date_newidx+1]
            adj_date = cls.resample_tradedate(trade_dt, adj_mode)
            adj_date = adj_date[(adj_date>=beg_date) & (adj_date<=end_date)].dropna()
            adj_date = adj_date.astype(trade_dt.iloc[0].values)
        elif adj_mode[0] == 'D':
            adj_date = trade_dt.iloc[beg_date_newidx:end_date_newidx+1:adj_mode[1], 0]
        elif adj_mode[0] == 'custom':
            adj_date = adj_mode[1]
        adj_date = pd.DataFrame(adj_date.values, columns=['trade_dt'])

        # 调整输出类型
        if res_type == 'datetime':
            adj_date = adj_date['trade_dt'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d')).to_frame('trade_dt')
        return adj_date

    @classmethod
    def _get_trddt_idx(cls, date_input: np.array):
        """
        返回日期在ashare_calendar中的索引
        :param date_input:
        :return:
        """
        AShareCalendar = cls.get_AShareCalendar(_trade_date_path)

        if isinstance(date_input, int):
            date_input = np.array([date_input])
        date_last = cls.offset_date(date_input, 0)
        date_last_idx = pd.Index(AShareCalendar['trade_dt']).get_indexer(date_last)
        return date_last_idx


    @classmethod
    def count_days(cls, begin_date: int, end_date: int, type='trade_date'):
        """
        计算两个日期之间的天数
        :param begin_date:
        :param end_date:
        :param type: ‘trade_date’ 日历日, 'calendar': 交易日
        :return:
        """
        if isinstance(begin_date, int):
            begin_date = np.array([begin_date])
        elif isinstance(begin_date, str):
            if (len(begin_date) == 8):
                begin_date = np.array([int(begin_date)])
            else:
                raise ValueError("请输入正确的日期格式！")

        if isinstance(end_date, int):
            end_date = np.array([end_date])
        elif isinstance(end_date, str):
            if (len(end_date) == 8):
                end_date = np.array([int(end_date)])
            else:
                raise ValueError("请输入正确的日期格式！")


        if type == 'trade_date':
            begin_date_last_idx = cls._get_trddt_idx(begin_date)
            end_date_last_idx = cls._get_trddt_idx(end_date)
            return end_date_last_idx - begin_date_last_idx
        elif type == 'calendar':
            begin_date_ = pd.DataFrame(begin_date).applymap(lambda x: datetime.strptime(str(int(x)), '%Y%m%d'))
            end_date_ = pd.DataFrame(end_date).applymap(lambda x: datetime.strptime(str(int(x)), '%Y%m%d'))
            diff = pd.DataFrame(end_date_.values - begin_date_.values ).applymap(lambda x: x.days).values
            return diff


class DoesNotExist(Exception):
    pass

class DBError(Exception):
    pass

if __name__ == '__main__':
    TradeDays.get_adjustdate(20210101, 20210201, ('D', 1))
