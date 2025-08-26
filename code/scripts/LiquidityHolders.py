#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time   : 2024/1/15 13:56
# !@File   : LiquidityHolders.py
# !@Author : shiyue
import os
import pandas as pd
import numpy as np
from .db import OracleDB, sqls_config
from scripts.db.db_utils import JYDB_Query, WINDDB_Query
from .utils_ri.RiskIndicators import RiskIndicators
from .settings import config, DIR_OF_MAIN_PROG
from .utils.log_utils import logger
from datetime import datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta
from typing import Optional, Union


class LiquidityLiablility(RiskIndicators):
    '''负债端流动性（持有人）'''

    def __init__(self, t, ptf_codes=None):
        self.basedate = t
        self.ptf_codes = ptf_codes
        self._connectingRiskDB()

        self.ptf_info = ['portfolio_code', 'full_name']
        self.share_chg = ['crt', 'subscribe', 'redeem']  # 份额可以分为：存续、申购、赎回

        # 持有人划分维度
        self.class_df = self.db_risk.read_sql(sqls_config['rc_lr_holders']['class_rule'])
        self.class_rule = list(set(self.class_df['class_rule'].to_list())) + ['total']

        self.ret_holder = pd.DataFrame()
        self.ret_cf = pd.DataFrame()
        self.share_df = pd.DataFrame()

    def _format_list(self, x: Union[str, list]):
        return [x] if isinstance(x, str) else x

    def retrieve_basic_data(self, t: Optional = None):
        '''从idc_fundholders_return表中获取基础数据，并对持有期限进行划分档位'''
        t = self.basedate if t is None else t

        q1 = sqls_config['rc_lr_holders']['idc_fundholders_return']
        raw_df = self.db_risk.read_sql(q1.format(t=t)).rename(columns={'shares_ratio': 'ratio'})
        if raw_df.empty:
            logger.error(f'{t} -- idc_fundholders_return 表无数据')
            return None

        ret_holder = raw_df[raw_df['data_type'] == '持有人'].copy()
        ret_cf = raw_df[raw_df['data_type'] == '资金'].copy()

        share_df = self.load_idc_fundholders(t=t)

        self.delete_table(table='fundholders_return_error', t=t, schema='quant', column_name='c_date')
        # 检查基金代码及名称是否正常
        check1 = ret_holder.copy()
        check1['code_length'] = check1['portfolio_code'].apply(lambda x: len(x))
        check1_res = check1[(check1['code_length'] < 9) | (check1['full_name'].isnull())].copy()
        if not check1_res.empty:
            check1_res['full_name'] = check1_res['full_name'].fillna('')
            self.insert_table(table='fundholders_return_error', data=check1_res.drop('code_length', axis=1), t=t)
            ret_holder = ret_holder[(check1['code_length'] == 9) & (~check1['full_name'].isnull())].copy()

        # 划分持有期限的类别
        check_hp = ret_holder[(ret_holder['holding_period'].isnull()) | (ret_holder['value'] == 0)].copy()
        if not check_hp.empty:
            logger.error('持有人数据有误')
            self.insert_table(table='fundholders_return_error', data=check_hp, t=t)
            ret_holder = ret_holder[(~ret_holder['holding_period'].isnull()) & (ret_holder['value'] > 0)].copy()

        bins = [0, 30, 60, 90, 180, 365, 3*365, np.inf]
        labels = ['0-30天', '31-60天', '61-90天', '91-180天', '181-365天', '1-3年', '3年以上']
        ret_holder['hp_span'] = pd.cut(ret_holder['holding_period'], bins, labels=labels, right=True).astype('str').fillna('持有期限为空')

        self.ret_holder = ret_holder
        self.ret_cf = ret_cf
        self.share_df = share_df

    def load_idc_fundholders(self, t: Optional = None):
        '''获取idc_fundholders数表以及单位净值信息（用于计算规模）'''
        t = self.basedate if t is None else t
        t0 = self.get_offset_tradeday(t=t, n=-1)
        q = sqls_config['rc_lr_holders']['idc_fundholders']

        df_t = self.db_risk.read_sql(q.format(t=t))
        df_t0 = self.db_risk.read_sql(q.format(t=t0)).rename(columns={'shares_total': 'all_shares_t0'})
        res = df_t.merge(df_t0[['portfolio_code', 'all_shares_t0']], how='left', on='portfolio_code')
        res['all_shares'] = res['shares_total']
        return res

    def basic_data_check(self, t: Optional = None):
        '''检查IDC_FUNDHOLDERS_RETURN与IDC_HOLDER表间的份额数据是否一致'''
        t = self.basedate if t is None else t
        index_cols = ['portfolio_code', 'full_name', 'holder_type', 'shares_type2']

        # 持有人维度：当日份额
        check_holder = self.ret_holder[self.ret_holder['redemption_date'].isnull()].copy()
        res_holder = check_holder.groupby(index_cols)['shares'].sum().rename('shares_holder').reset_index()

        # 资金维度：当日份额
        check_cf = self.ret_cf[self.ret_cf['redemption_date'].isnull()].copy()
        res_cf = check_cf.groupby(index_cols)['shares'].sum().rename('shares_cf').reset_index()

        # DWS_PROD_HOLDERS_BY_ORG(个人份额、货币基金数据无需核对）
        q = sqls_config['rc_lr_holders']['org_holders']
        res_cpr = self.db_risk.read_sql(q.format(t=t)).rename(columns={'shares': 'shares_cpr'})
        res_cpr = res_cpr[(res_cpr['holder_type'] != '个人') & (~res_cpr['full_name'].str.contains('货币市场基金'))].copy()

        # 1.核对idc_fundholders_return报表内部逻辑：内部核对存续+申购份额
        # 2.核对idc_fundholders_return 与 DWS_PROD_HOLDERS_BY_ORG：必须用资金维度的数据进行核对
        check_df = res_cpr.merge(res_cf, how='outer', on=index_cols).fillna(0)
        check_df = check_df.merge(res_holder, how='left', on=index_cols)

        check_df['shares_inner'] = check_df['shares_holder'] - check_df['shares_cf']
        check_df['shares_outer'] = check_df['shares_cpr'] - check_df['shares_cf']
        check_res = check_df[(abs(check_df['shares_inner']) >= 1) | (abs(check_df['shares_outer']) >= 1)].copy()

        if not check_res.empty:
            logger.error(f'{t} -- idc_fundholders_return 与 idc_holder 间勾稽验证未通过')
            check_res['c_date'] = t
            self.insert2db_single('rc_lr_holders_check', check_res, t=t, t_colname='c_date')

    def _define_shares_status(self, x):
        if isinstance(x.redemption_date, str) and x.redemption_date > '0000-00-00':
            return '赎回'
        if x.subscription_date == x.c_date:
            return '申购'
        return '存续'

    def rc_lr_holders(self, t: Optional = None):
        t = self.basedate if t is None else t

        self.retrieve_basic_data(t=t)  # 获取基础数据
        self.basic_data_check(t=t)  # 份额数据检查

        # 构建框架表
        ptf_info = self.ret_holder[self.ptf_info].drop_duplicates()
        total_df = pd.DataFrame([['total', '总计'], ['total', '总计(机构+产品)']], columns=['class_rule', 'class_name'])
        frame = pd.merge(ptf_info, pd.concat([self.class_df, total_df], ignore_index=True), how='cross')

        result = frame.merge(self.holding_shares(t), how='left', on=list(frame.columns)).fillna(0)  # 份额相关字段
        result = result.merge(self.holding_return(), how='left', on=list(frame.columns))  # 收益相关字段
        result['c_date'] = t

        key_cols = ['c_date'] + self.ptf_info + ['class_rule', 'class_name']
        data_cols = ['crt_value', 'crt_shares', 'crt_ratio', 'hp_sa', 'return_sa', 'return_ann_sa',
                     'hp_wa', 'return_wa', 'return_ann_wa', 'subscribe_value', 'subscribe_shares', 'subscribe_ratio',
                     'redeem_value', 'redeem_shares', 'redeem_ratio']
        result = result.reindex(columns=key_cols+data_cols)

        self.insert2db_single(table='rc_lr_holders', data=result, t=t, ptf_code=self.ptf_codes, t_colname='c_date')

    def holding_shares(self, t):
        cols = ['value', 'shares', 'ratio']
        crt_cols = dict(zip(cols, ['crt_' + i for i in cols]))
        sub_cols = dict(zip(cols, ['subscribe_' + i for i in cols]))
        red_cols = dict(zip(cols, ['redeem_' + i for i in cols]))

        # 存续份额可以直接从持有人维度计算
        crt_holder = self.ret_holder[self.ret_holder['redemption_date'].isnull()].copy()

        # 申购和赎回需要从资金维度开始计算
        index_cols = ['c_date', 'portfolio_code', 'full_name', 'holder_code', 'holder_name']
        holder_info = self.ret_holder[index_cols + ['holder_type', 'shares_type2', 'hp_span']].copy()

        sub_cf = self.ret_cf[self.ret_cf['subscription_date'] == t].copy()
        sub_holder = sub_cf.groupby(index_cols)[cols].sum().reset_index()
        sub_holder = sub_holder.merge(holder_info, how='left', on=index_cols)

        red_cf = self.ret_cf[self.ret_cf['redemption_date'] == t].copy()
        red_holder = red_cf.groupby(index_cols)[cols].sum().reset_index()
        red_holder = red_holder.merge(holder_info, how='left', on=index_cols)

        results = pd.DataFrame()
        for rule in self.class_rule:
            index = self.ptf_info + [rule] if rule != 'total' else self.ptf_info
            crt_res = crt_holder.groupby(index)[cols].sum().rename(columns=crt_cols).reset_index()
            sub_res = sub_holder.groupby(index)[cols].sum().rename(columns=sub_cols).reset_index()
            red_res = red_holder.groupby(index)[cols].sum().rename(columns=red_cols).reset_index()
            re_rule = {rule: 'class_name'} if rule != 'total' else {}
            res = crt_res.merge(sub_res, how='outer', on=index).merge(red_res, how='outer', on=index).rename(columns=re_rule)
            if 'class_name' not in res.columns:
                res['class_name'] = '总计(机构+产品)'
            res['class_rule'] = rule
            results = pd.concat([results, res.fillna(0)], ignore_index=True)

        # 根据idc_fundholders计算个人及组合层的数据
        indv_df = self.transform_share_df('holder_type', 'individual', '个人', results.columns)  # 个人
        total_df = self.transform_share_df('total', 'total', '总计', results.columns)  # 组合总计
        results = pd.concat([results, indv_df, total_df], ignore_index=True)

        return results

    def transform_share_df(self, class_rule: str, class_old: str, class_new: str, keep_cols: list):
        '''把idc_fundholders里的持有人份额数据转化为rc_lr_holders表的形式'''
        old_cols = ['%s_%s' % (i, class_old) for i in ['shares', 'subscription', 'redemption']]
        rename_cols = dict(zip(old_cols, ['%s_shares' % i for i in self.share_chg]))
        res = self.share_df.rename(columns=rename_cols)
        for c in self.share_chg:
            res['%s_value' % c] = res['%s_shares' % c] * res['nav']
            if c == 'redeem':
                res['%s_ratio' % c] = res['%s_shares' % c] / res['all_shares_t0']
            else:
                res['%s_ratio' % c] = res['%s_shares' % c] / res['all_shares']
        res[['class_rule', 'class_name']] = [class_rule, class_new]
        return res.reindex(columns=keep_cols)

    def holding_return(self):
        '''持有人当天的收益率包含当天赎回的持有人收益率'''

        cols = ['holding_period', 'hp_return', 'hp_return_ann']
        sa_cols = dict(zip(cols, ['hp_sa', 'return_sa', 'return_ann_sa']))

        func_wa = lambda x: pd.Series(np.average(x[cols], weights=x.value, axis=0),
                                      index=['hp_wa', 'return_wa', 'return_ann_wa'])

        results = pd.DataFrame()
        for rule in self.class_rule:
            index = self.ptf_info + [rule] if rule != 'total' else self.ptf_info
            sa_res = self.ret_holder.groupby(index)[cols].mean().rename(columns=sa_cols).reset_index()
            wa_res = self.ret_holder.groupby(index).apply(func_wa).reset_index()
            re_rule = {rule: 'class_name'} if rule != 'total' else {}
            res = pd.merge(sa_res, wa_res, how='outer', on=index).rename(columns=re_rule)
            if 'class_name' not in res.columns:
                res['class_name'] = '总计(机构+产品)'
            res['class_rule'] = rule
            results = pd.concat([results, res.fillna(0)], ignore_index=True)

        return results


class LiquidityAsset(RiskIndicators):
    '''资产端流动性（汇总统计流动型模型结果）'''

    def __init__(self, t, ptf_codes=None):
        self.basedate = t
        self.ptf_codes = ptf_codes
        self.liq_days = [1, 2, 3, 5]   # 需计算的可变现天数
        self.liq_level_d = {'资产总计': ['asset_type'], '一级分类': ['asset_type', 'liq_type_l1'],
                            '二级分类': ['asset_type', 'liq_type_l1', 'liq_type_l2']}  # 流动性分类的统计方式
        self.key_cols = ['portfolio_code', 'type_level', 'asset_type', 'liq_type_l1', 'liq_type_l2']
        self._connectingRiskDB()
        self._fundNameMapping()

    def deal_ptf_codes(self, data):
        if self.ptf_codes is not None:
            data = data[data['portfolio_code'].isin(self.ptf_codes)].copy()
        return data

    def retrieve_pos_detail(self, t=""):
        '''组合全量的持仓情况，含流动性维度下资产类型的分类标签'''
        # 从公共数表直接取数，涉及资产类型：股票、转债、债券、ABS、基金、衍生品、银行存款
        t = self.get_latest_tradeday(self.basedate if t == "" else t)
        data = self.db_risk.read_sql(sqls_config['rc_lr_asset']['position'].format(t=t))
        repo_reverse = self._pos_repo_reverse(t=t)  # 逆回购分类需要通过到期日进行计算

        result = pd.concat([data, repo_reverse], ignore_index=True).reindex(columns=data.columns)
        result = self.deal_ptf_codes(result)

        return result

    def _pos_repo_reverse(self, t=""):
        '''清洗逆回购的流动性类型'''
        t = self.basedate if t == "" else t
        data = self.db_risk.read_sql(sqls_config['rc_lr_asset']['repo_reverse'].format(t=t))

        # 设置逆回购流动性的一二级分类，剩余期限按照自然日进行计算
        data[['asset_type', 'liq_type_l1']] = ['逆回购', '逆回购']
        data['liq_type_l2'] = np.where(data['dlvy_date'] <= self.get_offset_tradeday(t, 1), '剩余1日',
                                       np.where(data['dlvy_date'] == self.get_offset_tradeday(t, 2), '剩余2日',
                                                '剩余3日及以上'))
        return data.rename(columns={'repo_vol': 'amount', 'repo_nprc': 'asset'})

    def get_offset_month(self, t="", n=1):
        '''
        月度日期偏移函数
        :param t: str or datetime，当前日期
        :param n: int, 偏移量，可正可负
        :return: str
        '''
        t = self.basedate if t == "" else t
        adj_day = 1 if n < 0 else -1
        date_t = datetime.strptime(t, "%Y-%m-%d") if isinstance(t, str) else t
        past_m = (int(date_t.year) * 12 + int(date_t.month) + n) % 12
        past_m = 12 if past_m == 0 else past_m
        past_y = int((int(date_t.year) * 12 + int(date_t.month) + n) / 12)
        past_d = monthrange(past_y, past_m)[1] if date_t.day > monthrange(past_y, past_m)[1] else date_t.day
        offset_t = datetime(past_y, past_m, past_d) + relativedelta(days=adj_day)
        return offset_t.strftime("%Y-%m-%d")

    def rc_lr_asset_realization(self, t=""):
        t = self.basedate if t == "" else t

        # 合并资产持仓量及变现能力: 1)流动性模型覆盖的证券持仓; 2)逆回购； 3)高流动性资产(现金、基金、期货)
        pos_detail = self.retrieve_pos_detail(t=t)
        pos_all = self.liq_position(t=t, pos_detail=pos_detail)
        liq_all = pd.concat([self.liq_securities(t=t), self.liq_repo_reverse(t=t), self.liq_high_level(t=t)])
        data = pd.merge(pos_all, liq_all, how='left', on=self.key_cols)
        data = self.deal_ptf_codes(data)

        # 计算组合层的变现能力
        liq_asset = data[data['type_level'] == '资产总计'].copy()
        cal_cols = ['liq_amount', 'liq_asset', 'pos_amount', 'pos_asset', 'sec_number']
        liq_ptf = liq_asset.groupby(['portfolio_code', 'liq_days'])[cal_cols].sum().reset_index()
        liq_ptf['type_level'] = '组合总计'
        results = pd.concat([liq_ptf, data], ignore_index=True)

        # 补充持仓占比及组合名称
        basic_info = self.db_risk.read_sql(sqls_config['rc_lr_asset']['basic_info'].format(t=t))
        results = results.merge(basic_info, how='left', on='portfolio_code')
        results['liq_ratio'] = results['liq_asset'] / results['net_asset']
        results['pos_ratio'] = results['pos_asset'] / results['net_asset']
        results['c_date'] = t

        col_names = ['c_date', 'portfolio_code', 'c_fundname', 'type_level', 'asset_type', 'liq_type_l1', 'liq_type_l2',
                     'liq_days', 'liq_amount', 'liq_asset', 'liq_ratio', 'pos_amount', 'pos_asset', 'pos_ratio', 'sec_number']
        results = results.reindex(columns=col_names)

        # 剔除永利八期
        results = results[~results['portfolio_code'].isin(['SJ2100.SMA', '011367.MOM'])].copy()
        self.insert2db_single('rc_lr_asset_realization', results, t=t, t_colname='c_date', ptf_code=self.ptf_codes)

        return results

    def liq_position(self, t="", pos_detail=None):
        '''组合在流动性模型的分类体系下的持仓情况及证券数量'''
        t = self.basedate if t == "" else t
        pos = self.retrieve_pos_detail(t=t) if pos_detail is None else pos_detail

        # 持仓债券数量分类统计
        sec_cols = ['portfolio_code', 'sec_code', 'asset_type', 'liq_type_l1', 'liq_type_l2']
        sec_types = ['股票', '债券', 'ABS', '可转债', '基金', '衍生品']
        data_secs = pos.loc[pos['asset_type'].isin(sec_types), sec_cols].drop_duplicates()
        sec_res = pd.DataFrame()
        for level in self.liq_level_d.keys():
            temp = data_secs.groupby(['portfolio_code'] + self.liq_level_d[level])['sec_code'].count().reset_index()
            temp['type_level'] = level
            sec_res = pd.concat([sec_res, temp], ignore_index=True)

        # 持仓资产规模及占比分类统计
        pos_res = pd.DataFrame()
        for level in self.liq_level_d.keys():
            temp = pos.groupby(['portfolio_code'] + self.liq_level_d[level])[['amount', 'asset']].sum().reset_index()
            temp['type_level'] = level
            pos_res = pd.concat([pos_res, temp], ignore_index=True)

        cols_dict = {'amount': 'pos_amount', 'asset': 'pos_asset', 'sec_code': 'sec_number'}
        result = pd.merge(pos_res, sec_res, how='left', on=self.key_cols).rename(columns=cols_dict)
        return result

    def liq_repo_reverse(self, t=""):
        t = self.basedate if t == "" else t
        data = self._pos_repo_reverse(t=t)

        # 计算可变现天数，可变现天数按照交易日进行计算
        results = pd.DataFrame()
        for level in self.liq_level_d.keys():
            for liq_day in self.liq_days:
                data_liq = data[data['dlvy_date'] <= self.get_offset_tradeday(t, liq_day)].copy()
                res = data_liq.groupby(['portfolio_code']+self.liq_level_d[level])[['amount', 'asset']].sum().reset_index()
                res[['type_level', 'liq_days']] = [level, liq_day]
                results = pd.concat([results, res], ignore_index=True)

        results = results.rename(columns={'amount': 'liq_amount', 'asset': 'liq_asset'})
        return results

    def liq_high_level(self, t="", pos_detail=None):
        '''高流动性资产（全部都可以在1日变现），包含：银行存款，基金, 期货'''
        t = self.basedate if t == "" else t
        pos_detail = self.retrieve_pos_detail(t=t) if pos_detail is None else pos_detail
        data = pos_detail[(pos_detail['asset_type'].isin(['基金', '银行存款']) |
                           (pos_detail['liq_type_l2'].str.contains('期货')))].copy()

        # 计算可变现天数，持仓基金均在一日内可以变现
        results = pd.DataFrame()
        for level in self.liq_level_d.keys():
            liq_types = self.liq_level_d[level]
            if not set(liq_types).issubset(set(list(data.columns))):
                continue
            for liq_day in self.liq_days:
                res = data.groupby(['portfolio_code'] + liq_types)[['amount', 'asset']].sum().reset_index()
                res[['type_level', 'liq_days']] = [level, liq_day]
                results = pd.concat([results, res], ignore_index=True)

        results = results.rename(columns={'amount': 'liq_amount', 'asset': 'liq_asset'})
        return results

    def liq_securities(self, t=""):
        '''dpe_lr_holding表中资产的可变现天数，包括债券、转债、股票、ABS'''
        t = self.basedate if t == "" else t
        sec_holding = self.db_risk.read_sql(sqls_config['rc_lr_asset']['securities_holding'].format(t=t))
        # 处理portfolio_code为空的情况
        sec_holding['c_fundcode'] = sec_holding['c_fullname'].map(self.fullname_to_code)
        sec_holding['portfolio_code'] = sec_holding['portfolio_code'].fillna(sec_holding['c_fundcode'])

        # 处理港交所股票代码前面多一个0的问题
        func = lambda x: x.code[1:] if '.HK' in x.code and x.LiqType == '股票' and len(x) > 7 else x.code
        sec_holding['code'] = sec_holding.apply(func, axis=1)
        sec_type = self.db_risk.read_sql(sqls_config['rc_lr_asset']['securities_type'].format(t=t))
        data = sec_holding.merge(sec_type, how='left', on='code')

        # 分类汇总
        results = pd.DataFrame()
        for level in self.liq_level_d.keys():
            res_list = []
            for col in ['liq_amount', 'liq_asset']:
                calc_cols = [col + '_' + str(i) for i in self.liq_days]
                liqdays_dict = dict(zip(calc_cols, self.liq_days))
                temp = data.rename(columns=liqdays_dict)
                res = temp.groupby(['portfolio_code'] + self.liq_level_d[level])[self.liq_days].sum().stack().to_frame(name=col)
                res_list.append(res)
            res_level = res_list[0].merge(res_list[1], how='outer', left_index=True, right_index=True).reset_index()
            res_level['type_level'] = level
            res_level = res_level.rename(columns={list(res_level.columns)[-4]: 'liq_days'})
            results = pd.concat([results, res_level])

        return results

    def dpe_lr_holding_expand(self, t=""):
        latest_td = self.get_latest_tradeday(self.basedate if t == "" else t)
        trans_cols = ['portfolio_code', 'tdvolume_1m', 'tdvalue_1m', 'tdprice_1m', 'tdvolume_1m_i', 'tdvalue_1m_i']
        lqscore_cols = ['lqscore_cnbd', 'lqpct_cnbd']

        # 检查持仓数据
        sql_raw = sqls_config['bond_transaction']['dpe_lr_holding']
        holdings = self.db_risk.read_sql(sql_raw.format(t=latest_td)).drop(['insert_time'] + trans_cols + lqscore_cols, axis=1)
        if len(holdings) == 0:
            logger.warning('dpe_lr_holding表无当日数据，请检查。')
            return None

        transaction = self.bond_transaction(holdings=holdings, new_cols=trans_cols, t=latest_td)
        lq_score = self.liquidity_score_cnbd(holdings=holdings, new_cols=lqscore_cols, t=latest_td)
        result = transaction.merge(lq_score, how='left', on='code')

        # 插入数据库
        self.delete_table('dpe_lr_holding', latest_td, 'quant', column_name='d_date')
        self.insert_table('dpe_lr_holding', result, 'quant', t=latest_td)

    def bond_transaction(self, holdings: pd.DataFrame, new_cols: list, t: str):
        '''个券及主体过去1个月的成交量'''
        wind_dq = WINDDB_Query()

        # 找到持仓券主体发行的全部证券
        sec_list = holdings[holdings['LiqType'] != '股票']['code'].to_list()
        issuer_secs = wind_dq.sec_query(sqls_config['bond_transaction']['bond_issuer'], sec_list=sec_list)
        all_secs = issuer_secs['bond_code'].to_list()   # 持仓券主体发行的全部证券
        bond_info = wind_dq.sec_query(sqls_config['bond_transaction']['bond_info'], sec_list=all_secs)  # 获取债券相关信息

        # 取数：近一月债券成交量、成交金额（全价）
        bg_date = self.get_offset_month(t, -1)
        sql_raw = sqls_config['bond_transaction']['period_transaction']
        trade_1m = wind_dq.sec_query(sql_raw, sec_list=all_secs, t0=bg_date.replace("-", ""), t1=t.replace("-", ""))
        trade_1m = trade_1m.merge(issuer_secs, how='left', on='bond_code').merge(bond_info, how='left', on='bond_code')
        # 原始数据是成交手数，交易所1手是10张，银行间1手是100张
        trade_1m['volume'] = trade_1m.apply(lambda x: x.volume * (10 if x.bond_mkt in ['SSE', 'SZSE'] else 100), axis=1)

        # 纯债&转债：个券过去1个月的成交量、成交价格、成交均价
        cols_b = {'volume': 'tdvolume_1m', 'value': 'tdvalue_1m'}
        bond_res = trade_1m.groupby('bond_code')[['volume', 'value']].sum().reset_index().rename(columns=cols_b)
        bond_res['tdprice_1m'] = bond_res.apply(lambda x: x.tdvalue_1m/x.tdvolume_1m if x.tdvolume_1m > 0 else np.nan, axis=1)
        results = holdings.merge(bond_res, how='left', left_on='code', right_on='bond_code')

        # 针对纯债计算：相同主体过去1个月的成交量、成交价格
        trade_1m_i = trade_1m[~trade_1m['windl1type'].isin(['可转债', '可交换债', '可分离转债存债', '资产支持证券'])].copy()
        cols_i = {'volume': 'tdvolume_1m_i', 'value': 'tdvalue_1m_i'}
        issuer_res = trade_1m_i.groupby('issuer_code')[['volume', 'value']].sum().reset_index().rename(columns=cols_i)

        bond_to_issuer = issuer_secs.set_index('bond_code')['issuer_code'].to_dict()
        bond_types = ['信用债', '同业存单', '利率债']
        results['issuer_code'] = results.apply(lambda x: bond_to_issuer[x.code] if x.LiqType in bond_types else None, axis=1)
        results = results.merge(issuer_res, how='left', on='issuer_code').drop('issuer_code', axis=1)

        # 匹配产品代码
        ptf_code = self.db_risk.read_sql(sqls_config['bond_transaction']['ptf_code'])
        results['portfolio_code'] = results['c_fundname'].map(ptf_code.set_index('c_fundname')['portfolio_code'].to_dict())

        results = results.reindex(columns=list(holdings.columns) + new_cols)
        return results

    def liquidity_score_cnbd(self, holdings: pd.DataFrame, new_cols: list, t: str):
        '''中债流动性指标'''
        jydq = JYDB_Query()
        sec_list = list(set(holdings[holdings['LiqType'] != '股票']['code'].to_list()))
        sec_digit = [i[0:i.find(".")] for i in sec_list]

        lq_score = jydq.sec_query('liquidity_score_cnbd', sec_digit, t)
        lq_score.columns = [i.lower() for i in lq_score]

        lq_score['mkt'] = lq_score['secumarket'].map({89: '.IB', 83: '.SH', 90: '.SZ', '72': '.HK'})
        lq_score['code'] = lq_score['secucode'] + lq_score['mkt']
        lq_score = lq_score[lq_score['code'].isin(sec_list)].copy()

        return lq_score.reindex(columns=['code'] + new_cols)

    def rc_lr_liquidity_level(self, t=""):
        '''统计组合的流动性水平分布'''
        t = self.basedate if t == "" else t
        pos_detail = self.retrieve_pos_detail(t=t)
        liq_level = self.db_risk.read_sql(sqls_config['rc_lr_asset']['liq_level_label'])
        data = pos_detail.merge(liq_level, how='left', on=['asset_type', 'liq_type_l1', 'liq_type_l2'])

        # 计算资产层的流动性水平分类
        result = pd.DataFrame()
        for level in self.liq_level_d.keys():
            temp = data.groupby(['portfolio_code'] + self.liq_level_d[level] + ['liq_level'])[['amount', 'asset']].sum().reset_index()
            temp['type_level'] = level
            result = pd.concat([result, temp], ignore_index=True)

        # 计算组合层的变现能力
        res_ptf = data.groupby(['portfolio_code', 'liq_level'])[['amount', 'asset']].sum().reset_index()
        res_ptf['type_level'] = '组合总计'
        results = pd.concat([result, res_ptf], ignore_index=True)

        # 补充持仓占比及组合名称
        basic_info = self.db_risk.read_sql(sqls_config['rc_lr_asset']['basic_info'].format(t=t))
        results = results.merge(basic_info, how='left', on='portfolio_code')
        results['liq_ratio'] = results['asset'] / results['net_asset']
        results['c_date'] = t

        # 设置列名
        cols_dict = {'amount': 'liq_amount', 'asset': 'liq_asset'}
        col_names = ['c_date', 'portfolio_code', 'c_fundname', 'type_level', 'asset_type', 'liq_type_l1', 'liq_type_l2',
                     'liq_level', 'liq_amount', 'liq_asset', 'liq_ratio']
        results = results.rename(columns=cols_dict).reindex(columns=col_names)

        # 剔除永利八期
        results = results[~results['portfolio_code'].isin(['SJ2100.SMA', '011367.MOM'])].copy()
        results = self.deal_ptf_codes(results)

        self.insert2db_single('rc_lr_liquidity_level', results, t=t, t_colname='c_date', ptf_code=self.ptf_codes)

        return results


