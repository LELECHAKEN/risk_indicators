# @Time : 2022/2/10 11:03 
# @Author : for wangp
# @File : risk_alert_additional.py 
# @Software: PyCharm
import os
import datetime
import pandas as pd
import numpy as np
from WindPy import w
from sqlalchemy import and_, column

from .settings import config
from .utils.log_utils import logger
from .db import OracleDB, column, sqls_config
from .utils_ri.RiskIndicators import RiskIndicators
from .trade_dt.date_utils import TradeDays, convert_date
from .db.db_factors import (DbFactor, convert_market_code, convert_sec_code)


class PortfolioWarning_add(RiskIndicators):
    def __init__(self, t, ptf_codes=None):
        '''
        初始化
        :param t: 计算基期
        '''
        self.basedate = t
        self._format_ptf_codes(ptf_codes)
        self.path_bindex_com = config['shared_drive_data']['bond_index_component']['path']
        self.db_risk = OracleDB(config['data_base']['QUANT']['url'])
        self.db_wind = OracleDB(config['data_base']['WIND']['url'])
        # self.bond_index_info = self.db_risk.read_sql(sqls_config["bond_index_fund"]["sql"])

        self._connectingRiskDB()
        self._define_index_prods()
        self._load_portfolioinfo()
        self.check_nav_data()
        # self._load_threshold_db()
        self.rm_base = pd.merge(self.retrieve_rm_threshold(), self.retrieve_rm_indicators(),
                                how='left', on='portfolio_code').dropna(subset=['net_asset']).fillna(np.nan)
        last_year_last_tdate = TradeDays.offset_date(
                            int(f"{int(self.basedate[:4]) - 1}-12-31".replace('-', '')), 0, 'D')[0]
        self.this_year_first_cdate = f"{int(self.basedate[:4])}-01-01"
        self.this_year_first_tdate = datetime.datetime.strftime\
            (TradeDays.offset_date(int(last_year_last_tdate), 1, 'D', res_type='datetime')[0], '%Y-%m-%d')

    def _define_index_prods(self):
        '''定义债券指数产品及对应的指数代码、更新日期等信息'''
        self.prod_dict = {'009171.OF': {'bch_code': 'CBA08303.CS', 'update_date': self.basedate, 'lower_bound': 1, 'upper_bound': 5},
                          '006925.OF': {'bch_code': 'CBA07403.CS', 'update_date': self.basedate, 'lower_bound': 1, 'upper_bound': 3},
                          '011983.OF': {'bch_code': 'CBA09301.CS', 'update_date': self.basedate, 'lower_bound': 3, 'upper_bound': 5}}
        self.prod_df = pd.DataFrame.from_dict(self.prod_dict, orient='index').reset_index().rename(columns={'index': 'portfolio_code'})

    def update_prods(self, portfolio_code, bch_code, update_date, lower_bound, upper_bound):
        prod_dict = {portfolio_code: {'bch_code': bch_code, 'update_date': update_date, 'lower_bound': lower_bound, 'upper_bound': upper_bound}}
        prod_df = pd.DataFrame.from_dict(self.prod_dict, orient='index').reset_index().rename(columns={'index': 'portfolio_code'})
        self.insert_table('dpe_bond_index', prod_df, 'quant', 'append')

    def _load_portfolioinfo(self):
        '''
        加载组合代码等基础信息
        '''
        q = sqls_config['ptf_basicinfo']['Sql']%(self.basedate)
        self.portfolioinfo = self.db_risk.read_sql(q)

    def _load_threshold_db(self):
        '''
        加载并向数据库更新投委会决议制定的各阈值
        '''
        threshold = pd.read_excel(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\PortfolioType\指标监控-投委会.xlsx', engine='openpyxl')
        data = threshold.loc[:, ['产品全称', '久期阈值', '剩余期限阈值', '回撤阈值', '滚动30天胜率阈值']].copy()
        data.columns = ['c_fullname', 'threshold_duration', 'threshold_ptm', 'threshold_dd_oneyear', 'threshold_winratio']
        self.threshold = pd.merge(data, self.portfolioinfo, left_on='c_fullname', right_on='full_name', how='left').drop(columns=['full_name'])
        self.insert_table('manual_threshold_add', self.threshold, if_exists='replace')

    def load_index_component(self, portfolio_code):
        '''加载各债券指数成分券清单'''
        file_name = '成分列表(中债)%s(%s).xlsx'%(self.prod_dict[portfolio_code]['update_date'], self.prod_dict[portfolio_code]['bch_code'])
        file_path = os.path.join(self.path_bindex_com, file_name)
        index_comp = pd.read_excel(file_path, engine='openpyxl')
        return index_comp

    def calc_bond_index_dura_single(self, portfolio_code):
        '''
        计算给定组合对应的债券指数久期，口径剔除待偿期在上下限以外的成分券后重新加权\n
        :param portfolio_code: string, 组合代码
        :return: float, 组合对应的指数久期
        '''
        index_comp = self.load_index_component(portfolio_code)
        upper_bound = self.prod_dict[portfolio_code]['upper_bound']
        lower_bound = self.prod_dict[portfolio_code]['lower_bound']

        filter_cnt = index_comp.loc[(index_comp['待偿期(年)'] <= lower_bound) | (index_comp['待偿期(年)'] >= upper_bound), '指数市值权重']
        new_comp = index_comp.loc[(index_comp['待偿期(年)'] >= lower_bound) & (index_comp['待偿期(年)'] <= upper_bound), :].copy()
        index_dura = (new_comp['估价修正久期'] * new_comp['指数市值权重']).sum()/new_comp['指数市值权重'].sum()
        logger.info('正在计算%s对应的指数久期，还原久期为%.2f，已剔除总%d只债券，共%.2f%%的权重。'%(portfolio_code, index_dura, filter_cnt.shape[0], filter_cnt.sum()))
        return index_dura

    def calc_bond_index_dura_all(self):
        '''计算并更新所有债券指数基金组合所对应的各个指数久期'''
        for portfolio_code in self.prod_dict.keys():
            index_dura = self.calc_bond_index_dura_single(portfolio_code)
            self.prod_dict[portfolio_code]['index_dura'] = index_dura
            self.prod_dict[portfolio_code]['calc_date'] = self.basedate

            # 更新数据库中债券指数久期的阈值
            condition = column("portfolio_code") == portfolio_code
            self.db_risk.update("manual_threshold_add", condition=condition, data={"threshold_duration": index_dura, 'insert_time': datetime.datetime.now()}, schema="risk")
            logger.info('%s对应债券指数久期已更新，阈值为%.2f'%(portfolio_code, index_dura))

        new_df = pd.DataFrame.from_dict(self.prod_dict, orient='index').reset_index().rename(columns={'index': 'portfolio_code'})
        self.insert2db_single('dpe_bond_index_dura', new_df, t=self.basedate, t_colname='CALC_DATE')

    def _bm_index_duration(self, bch_codes: list):
        '''从wind获取债券指数的修正久期'''
        w.start()
        temp = w.wsd(','.join(bch_codes), "modifiedduration", self.basedate, self.basedate)
        bch_dur = pd.DataFrame(temp.Data, index=['dur_bm'], columns=temp.Codes).T.reset_index().rename(
            columns={'index': 'bm_index_code'})
        return bch_dur

    def retrieve_rm_threshold(self, dd_period='近一年'):
        sql_thold = sqls_config['rm_threshold']['sql']
        df_thold = self.db_risk.read_sql(sql_thold.format(t=self.basedate))
        dd_cols = ['dd_thold', 'dd_no_morepos', 'dd_key_warn', 'dd_alert_l2', 'dd_alert_l1', 'dd_target', 'dd_tg_alert']
        df_thold[dd_cols] = (-1) * df_thold[dd_cols]

        # 加载指数的阈值表
        sql_index = sqls_config['rm_threshold_indexfund']['sql']
        index_thold = self.db_risk.read_sql(sql_index.format(t=self.basedate)).fillna({'dur_dev_thold': 0})

        # 指数久期与回撤
        index_codes = index_thold['bm_index_code'].to_list()
        index_dur = self._bm_index_duration(index_codes)
        sql_indexind = sqls_config['rm_indicators_bmindex']['sql']
        index_dd = self.db_risk.read_sql(sql_indexind.format(t=self.basedate, period=dd_period, code=','.join(repr(x) for x in index_codes)))

        # 根据指数的久期及回撤确定阈值
        index_thold = index_thold.merge(index_dur, how='left', on='bm_index_code').merge(index_dd, how='left', on='bm_index_code')

        # 将1-3、1-5、3-5的回撤基准修改为同业竞品平均
        sql_cpr = sqls_config['rm_threshold_cpr']['sql'].format(t=self.basedate, dd_period=dd_period)
        cpr_dd = self.db_risk.read_sql(sql_cpr).set_index('ranking_list')['cpr_dd'].to_dict()
        for adj_type in ['1-3年利率债指数',  '1-5年利率债指数', '3-5年利率债指数']:
            idx = index_thold[index_thold['type_l3'] == adj_type].index
            index_thold.loc[idx, 'dd_bm'] = cpr_dd[adj_type]

        # 计算指数产品的回撤阈值
        index_thold['dd_thold'] = index_thold['dd_bm'] - index_thold['dd_thold']
        index_thold['dur_thold'] = index_thold.apply(lambda x: x.dur_bm * (1+x.dur_dev_thold) if x.dur_dev_thold > 0 else x.dur_thold, axis=1)
        index_thold['dd_alert_l1'] = index_thold['dd_bm'] - index_thold['dd_alert_l1']
        index_thold['dd_key_warn'] = index_thold['dd_bm'] - index_thold['dd_key_warn']

        # 对碳中和的回撤阈值进行特殊处理
        idx_th = index_thold.set_index('portfolio_code').to_dict('index')
        info1 = idx_th['013654.OF']  # 指数维度的回撤阈值
        idx1 = df_thold[df_thold['portfolio_code'] == '013654.OF'].index
        df_thold.loc[idx1, 'bm_index_code'] = info1['bm_index_code']
        df_thold.loc[idx1, 'dd_bm'] = info1['dd_bm']
        df_thold.loc[idx1, 'dur_bm'] = info1['dur_bm']
        df_thold.loc[idx1, 'dd_thold'] = min(df_thold.loc[idx1, 'dd_thold'].values[0], info1['dd_thold'])
        df_thold.loc[idx1, 'dd_no_morepos'] = min(df_thold.loc[idx1, 'dd_no_morepos'].values[0], 0.9 * info1['dd_thold'])
        index_thold = index_thold[index_thold['portfolio_code'] != '013654.OF'].copy()

        # 对双利的回撤阈值进行特殊处理
        info2 = idx_th['002521.OF']  # 指数维度的回撤阈值
        idx2 = df_thold[df_thold['portfolio_code'] == '002521.OF'].index
        df_thold.loc[idx2, 'bm_index_code'] = info2['bm_index_code']
        df_thold.loc[idx2, 'dd_bm'] = info2['dd_bm']
        df_thold.loc[idx2, 'dd_thold'] = info2['dd_thold']
        df_thold.loc[idx2, 'dd_key_warn'] = info2['dd_thold']
        index_thold = index_thold[index_thold['portfolio_code'] != '002521.OF'].copy()

        df_thold = pd.concat([df_thold, index_thold], ignore_index=True)

        return df_thold

    def retrieve_rm_indicators(self, dd_period='近一年'):
        sql_raw = sqls_config['rm_indicators']['sql']
        res_1yr = self.db_risk.read_sql(sql_raw.format(t=self.basedate, period=dd_period, period_bck='成立以来'))

        # 采用经理增聘以来：双利
        adj_code2 = ['002521.OF']
        adj_res2 = self.db_risk.read_sql(sql_raw.format(t=self.basedate, period='陶毅接手以来', period_bck='陶毅接手以来'))
        adj_res2 = adj_res2[adj_res2['portfolio_code'].isin(adj_code2)].copy()

        # def competitor_crtdd(type, bg_date):
        #     q = sqls_config['rm_indicators_competitor']['sql']
        #     nav_df = self.db_risk.read_sql(q.format(type=type, bg_date=bg_date))
        #     return nav_df['nav'].iloc[-1] / nav_df['nav'].max() -1
        # adj_res2['dd_mkt'] = adj_res2.apply(lambda x: competitor_crtdd(x.ranking_list, x.period_bgdate), axis=1)

        # 采用经理增聘以来：鑫享
        adj_code3 = ['008723.OF']
        adj_res3 = self.db_risk.read_sql(sql_raw.format(t=self.basedate, period='杨野接手以来', period_bck='杨野接手以来'))
        adj_res3 = adj_res3[adj_res3['portfolio_code'].isin(adj_code3)].copy()

        # 采用经理增聘以来：鑫享
        adj_code4 = ['017220.OF']
        adj_res4 = self.db_risk.read_sql(sql_raw.format(t=self.basedate, period='袁旭接手以来', period_bck='袁旭接手以来'))
        adj_res4 = adj_res4[adj_res4['portfolio_code'].isin(adj_code4)].copy()

        adj_codes = adj_code2 + adj_code3 + adj_code4
        adj_res = pd.concat([adj_res2, adj_res3, adj_res4])

        keep_res = res_1yr[~res_1yr['portfolio_code'].isin(adj_codes)].copy()
        result = pd.concat([keep_res, adj_res], ignore_index=True).drop(columns=['period_bgdate', 'ranking_list'])

        # 合并股票ETF持仓
        self._fund_position()
        result = result.merge(self.stk_etf, how='left', on='portfolio_code')
        result['stk_p_fund_pos'] = result[['stk_pos', 'stk_fund_pos']].sum(axis=1)

        # 信用债资产久期
        credit_dur = self.db_risk.read_sql(sqls_config['rm_indicators_credit_dur']['sql'].format(t=self.basedate))
        result = result.merge(credit_dur, how='left', on='portfolio_code')

        return result

    def _fund_position(self):
        q = sqls_config['rm_indicators_fundpos']['sql']
        data = self.db_risk.read_sql(q.format(t=self.basedate))

        # check是否存在未匹配到的ETF基金
        check_data = data[data['cgstype_l1'].isnull()].copy()

        # 股票ETF
        stk_etf_l2 = ['1.6 股票ETF基金', '1.12 港股通股票ETF基金']
        stk_etf_l3 = ['6.1.5 QDII跨境股票ETF基金']
        temp = data[(data['cgstype_l2'].isin(stk_etf_l2)) | (data['cgstype_l3'].isin(stk_etf_l3))].copy()
        stk_etf = temp.groupby('portfolio_code')['fmv_nav_ratio'].sum().reset_index()
        self.stk_etf = stk_etf.rename(columns={'fmv_nav_ratio': 'stk_fund_pos'})

    def duration_alert(self):
        '''
        久期监控，阈值见投委会相关决议(其中债指基金按照指数久期的0.5-1.5倍来监控)
        :return:
        '''
        q = sqls_config['duration_alert']['Sql']%self.basedate
        port_dura = self.db_risk.read_sql(q).rename(columns={'avgDuration_port': 'avgduration_port'})
        # port_dura = port_dura.loc[port_dura['threshold_duration'].notna(), :].copy()
        # 纯债基金
        self.dura_alert_1 = port_dura.loc[~port_dura['c_fundcode'].isin(self.prod_dict.keys()) &
                                          (port_dura['avgduration_port'] > port_dura['threshold_duration']), :].drop(columns=['portfolio_code'])
        # 债券指数基金
        self.dura_alert_2 = port_dura.loc[port_dura['c_fundcode'].isin(self.prod_dict.keys()) &
                                          ((port_dura['avgduration_port'] > 1.5 * port_dura['threshold_duration']) |
                                          (port_dura['avgduration_port'] < 0.5 * port_dura['threshold_duration'])), :].drop(columns=['portfolio_code'])
        self.dura_alert = self.dura_alert_1.append(self.dura_alert_2, sort=False)

        self.insert2db_single('rc_alert_duration', self.dura_alert, t=self.basedate)

    def ptm_alert(self):
        '''
        货基剩余期限监控，阈值见投委会相关决议
        '''
        q = sqls_config['ptm_alert']['Sql']%self.basedate
        port_ptm = self.db_risk.read_sql(q).rename(columns={'avgPtm': 'avgptm'})
        # port_ptm = port_ptm.loc[port_ptm['threshold_duration'].notna(), :].copy()
        self.ptm_alert = port_ptm.loc[port_ptm['avgptm'] > port_ptm['threshold_ptm'], :].drop(columns=['portfolio_code'])
        self.ptm_port = port_ptm.copy()

        self.insert2db_single('rc_alert_ptm', self.ptm_alert, t=self.basedate)

    def drawdown_alert(self):
        '''
        回撤监控，阈值见投委会相关决议
        :return:
        '''
        q = sqls_config['drawdown_alert']['Sql']%self.basedate
        port_dd = self.db_risk.read_sql(q)
        # port_dura = port_dura.loc[port_dura['threshold_duration'].notna(), :].copy()
        self.dd_data = port_dd.dropna(subset=['threshold_dd_oneyear'])
        # self.dd_alert = port_dd.loc[port_dd['maxdrawdown_oneyear'] * (-1) > port_dd['threshold_dd_oneyear'], :].drop(columns=['portfolio_code'])


        dd_1yr_df = self.db_risk.read_sql(sqls_config["rc_alert_dd_oneyear"]["sql"].format(t=self.basedate))
        self.dd_alert = dd_1yr_df.copy()
        self.insert2db_single("rc_alert_dd_oneyear", dd_1yr_df, t=self.basedate)

    def winratio_alert(self):
        '''
        胜率监控(过去一年)，阈值见投委会相关决议
        :return:
        '''
        startdate = self.basedate.replace(self.basedate[:4], str(int(self.basedate[:4]) - 1))
        enddate = self.basedate
        q = sqls_config['winratio_alert']['Sql']%(startdate, enddate)
        self.winratio_alert = self.db_risk.read_sql(q)
        self.insert2db_single('rc_alert_winratio', self.winratio_alert, t=self.basedate)

    def alert_rating(self):
        # 组合持仓券
        hld_df = self.db_risk.read_sql(sqls_config['rm_b_plus_holding']['sql'].format(t=self.basedate))
        idx_cols = ['portfolio_code', 'c_fundname']
        b_plus = hld_df.groupby(idx_cols)['f_asset'].sum().reset_index().rename(columns={'f_asset': 'b_plus_asset'})
        b = hld_df[hld_df['inner_rating_b'] != 'B+'].groupby(idx_cols)['f_asset'].sum().reset_index().rename(columns={'f_asset': 'b_asset'})
        b_hld = pd.merge(b_plus, b, how='outer', on=idx_cols).fillna(0)

        # 组合规模
        na_df = self.rm_base[['c_date', 'portfolio_code', 'net_asset']].copy()
        b_hld = b_hld.merge(na_df, how='left', on='portfolio_code')
        b_hld['b_plus_ratio'] = b_hld['b_plus_asset'] / 1e8 / b_hld['net_asset']
        b_hld['b_ratio'] = b_hld['b_asset'] / 1e8 / b_hld['net_asset']

        # 持仓阈值
        def _threshold(x):
            if x.net_asset <= 4:
                return np.nan, np.nan
            if x.net_asset <= 20:
                return np.nan, 0.5
            if x.net_asset <= 30:
                return 0.3, 0.15
            return 0.15, 0.1

        b_hld[['b_plus_thold', 'b_thold']] = b_hld.apply(_threshold, axis=1, result_type='expand')

        # 判断是否触警
        def _alert_res(x):
            if x.b_plus_ratio > x.b_plus_thold:
                if x.b_ratio > x.b_thold:
                    return "B+及以下&B及以下持仓突破阈值"
                return "B+及以下持仓突破阈值"
            if x.b_ratio > x.b_thold:
                return "B及以下持仓突破阈值"
            return "未触警"

        b_hld['alert_level'] = b_hld.apply(_alert_res, axis=1)
        res_cols = ['c_date', 'portfolio_code', 'c_fundname', 'net_asset', 'alert_level',
                    'b_plus_asset', 'b_asset', 'b_plus_ratio', 'b_ratio', 'b_plus_thold', 'b_thold']
        rm_rating = b_hld.reindex(columns=res_cols)
        self.insert2db_single('ads_rm_result_rating', rm_rating, t=self.basedate, t_colname='c_date')

        self.rm_rating = rm_rating[rm_rating['alert_level'] != '未触警'].copy()

    def alert_dur(self):
        result = self.rm_base.copy()
        # 部分混合产品的久期阈值取决于权益仓位的高低
        result['dur_thold'] = result.apply(self._dur_thold_adj, axis=1)
        result['alert_level'] = result.apply(self._dur_alert_rule, axis=1, result_type='expand')
        result['dur_deviation'] = result.apply(lambda x: x.dur_fund/x.dur_bm - 1 if x.dur_bm > 0 else np.nan, axis=1)

        keep_cols = ['c_date', 'portfolio_code', 'c_fundname', 'type_l0', 'type_l1', 'type_l2', 'manager', 'net_asset',
                     'alert_level', 'dur_thold', 'dur_fund', 'dur_bm', 'dur_deviation',
                     'credit_dur_thold', 'credit_dur_fund']
        self.rm_dur = result.reindex(columns=keep_cols)
        self.rm_dur_rpt = self.rm_dur[self.rm_dur['alert_level'] != '未触警'].copy()

        self.insert2db_single('rm_result_dur', self.rm_dur, t=self.basedate, t_colname='c_date')

    def alert_dd(self):
        result = self.rm_base.copy()
        # 投决会回撤风险预算
        result[['alert_level', 'dd_alert_level', 'dur_alert_level']] = result.apply(self._dd_alert_rule, axis=1, result_type='expand')
        # 风险目标预警
        result[['alert_level', 'dd_alert_level']] = result.apply(self._target_rule, axis=1, result_type='expand')

        # 特殊处理双利
        result[['alert_level', 'dd_alert_level']] = result.apply(self._special_rule, axis=1, result_type='expand')
        keep_cols = ['c_date', 'portfolio_code', 'c_fundname', 'type_l0', 'type_l1', 'type_l2', 'manager', 'net_asset',
                     'alert_level', 'dd_thold', 'dd_alert_level', 'dur_alert_level', 'dd_fund', 'dd_dur_fund', 'dd_mkt',
                     'dd_bm', 'dur_fund', 'cb_pos', 'stk_pos', 'eq_pos', 'dev_alert_l1', 'track_dev']
        self.rm_dd = result.reindex(columns=keep_cols)
        self.rm_dd_rpt = self.rm_dd[self.rm_dd['alert_level'] != '未触警'].copy()

        # for col in ['dd_fund', 'dd_dur_fund', 'dd_mkt', 'dd_bm', 'stk_pos', 'eq_pos']:
        for col in ['dd_fund', 'dd_dur_fund', 'dd_mkt', 'dd_bm']:
            self.rm_dd[col] = self.rm_dd[col].map(lambda x: np.nan if x == 0 else x)

        self.insert2db_single('rm_result_dd', self.rm_dd, t=self.basedate, t_colname='c_date')

    def alert_position(self):
        result = self.rm_base[(~self.rm_base['cb_upper'].isnull()) | (~self.rm_base['stk_upper'].isnull()) |
                              (~self.rm_base['eq_upper'].isnull()) | (~self.rm_base['stk_fund_upper'].isnull())].copy()
        result = result.assign(eq_lower=0)
        for asset in ['stk', 'cb', 'eq']:
            result[f'{asset}_limit'] = result.apply(lambda x: self._pos_limit(x[f'{asset}_upper'], x[f'{asset}_lower']),
                                                    axis=1)
        result['eq_pos'] = np.where(result['portfolio_code'].isin(['021345.OF', '014678.OF']),
                                    result[['eq_pos', 'stk_fund_pos']].sum(axis=1), result['eq_pos'])
        result['alert_level'] = result.apply(self._pos_alert_rule, axis=1)

        # 与基金合同重合的条目，并邮件通知负责人；如预警则按照净价重新计算一遍，
        alert_prods = result[result['alert_level'] != '未触警'].copy()
        if not alert_prods.empty:
            alert_reason = self.db_risk.read_sql(sqls_config['rm_threshold_pos']['sql'])
            result = result.merge(alert_reason, how='left', left_on=['portfolio_code', 'c_fundname', 'alert_level'],
                                  right_on=['portfolio_code', 'c_fundname', 'asset_type'])
            f = lambda x: '投决会决议' if x.alert_level != '未触警' and not isinstance(x.reason, str) else x.reason,
            result['reason'] = result.apply(f, axis=1)

        keep_cols = ['c_date', 'portfolio_code', 'c_fundname', 'type_l0', 'type_l1', 'type_l2', 'manager', 'net_asset',
                     'alert_level', 'cb_upper', 'cb_lower', 'cb_pos', 'stk_upper', 'stk_lower', 'stk_pos',
                     'eq_upper', 'eq_pos', 'stk_fund_upper', 'stk_fund_pos', 'stk_p_fund_upper', 'stk_p_fund_pos',
                     'reason', 'stk_limit', 'cb_limit', 'eq_limit']
        self.rm_pos = result.reindex(columns=keep_cols)
        self.rm_pos_rpt = self.rm_pos[self.rm_pos['alert_level'] != '未触警'].copy()

        self.insert2db_single('rm_result_pos', self.rm_pos, t=self.basedate, t_colname='c_date')

    def _dd_alert_rule(self, data: pd.DataFrame):

        def _purebond_early_alert(data: pd.DataFrame):
            if data.dd_fund < data.dd_alert_l2 and data.dur_fund > data.dur_alert_l2:
                return '早期预警-2档', data.dd_alert_l2, data.dur_alert_l2
            if data.dd_fund < data.dd_alert_l1 and data.dur_fund > data.dur_alert_l1:
                return '早期预警-1档', data.dd_alert_l1, data.dur_alert_l1
            return '未触警', np.nan, np.nan

        def _cbond_early_alert(data: pd.DataFrame):
            if data.dd_fund < data.dd_alert_l2:
                return '早期预警', data.dd_alert_l2, np.nan
            if data.dd_fund < data.dd_alert_l1 and (data.dur_fund > data.dur_alert_l1 or data.cb_pos > data.cb_alert_l1):
                return '早期预警', data.dd_alert_l1, data.dur_alert_l1
            return '未触警', np.nan, np.nan

        def _fiplus_early_alert(data: pd.DataFrame):
            if data.dd_fund < data.dd_alert_l2:
                return '早期预警-2档', data.dd_alert_l2, np.nan
            if data.dd_fund < data.dd_alert_l1:
                return '早期预警-1档', data.dd_alert_l1, np.nan
            return '未触警', np.nan, np.nan

        def _index_early_alert(data: pd.DataFrame):
            if data.dd_fund < data.dd_alert_l1 and data.track_dev > data.dev_alert_l1:
                return '超额回撤及跟踪误差早期预警', data.dd_alert_l1, np.nan
            if data.dd_fund < data.dd_alert_l1:
                return '超额回撤早期预警', data.dd_alert_l1, np.nan
            if data.track_dev > data.dev_alert_l1:
                return '跟踪误差早期预警', data.dd_alert_l1, np.nan
            return '未触警', np.nan, np.nan

        if data.dd_fund < data.dd_thold < 0:
            return '突破阈值', data.dd_thold, np.nan
        if data.dd_fund < data.dd_no_morepos < 0:
            return '禁止加仓', data.dd_no_morepos, np.nan
        if data.dd_fund < data.dd_key_warn < 0:
            return '重点预警', data.dd_key_warn, np.nan
        if data.type_l0 == '公募' and data.type_l1 == '混合型' and data.type_l2 == '一级债基':
            return _cbond_early_alert(data)
        if data.type_l0 == '公募' and data.type_l1 == '混合型':
            return _fiplus_early_alert(data)
        if data.type_l0 == '公募' and data.type_l1 in ('债券型', '多策略型') and data.type_l2 != '标准型':
            return _purebond_early_alert(data)
        if data.type_l2 == '债券指数':
            return _index_early_alert(data)
        return '未触警', np.nan, np.nan

    def _target_rule(self, data: pd.DataFrame):
        tg_level, tg_dd = "未触警", np.nan
        if data.dd_fund < data.dd_target:
            tg_level, tg_dd = "风险目标预警", data.dd_target
        if data.dd_fund < data.dd_tg_alert:
            tg_level, tg_dd = "风险目标早期预警", data.dd_tg_alert

        if data.alert_level == '未触警':
            return tg_level, tg_dd
        if tg_level == '未触警':
            return data.alert_level, data.dd_alert_level

        return data.alert_level + "&" + tg_level, min(tg_dd, data.dd_alert_level)

    def _special_rule(self, data: pd.DataFrame):
        if data.portfolio_code in ['002521.OF']:
            if data.dd_bm <= -0.08 and data.dd_fund <= data.dd_key_warn:
                return '重点预警', data.dd_key_warn
        return data.alert_level, data.dd_alert_level

    def _pos_limit(self, upper, lower):
        limit = '-'
        if upper > 0 and (lower >= 0 or lower is np.nan or pd.isnull(lower)):
            str_lower = int(lower*100) if lower > 0 else 0
            limit = f'{str_lower}% - {int(upper*100)}%'
        if upper is np.nan and lower > 0:
            limit = f'≥{int(lower*100)}%'
        if upper == 0:
            limit = '不可投'
        return limit

    def _pos_alert_rule(self, data: pd.DataFrame):
        def _stk_alert(data: pd.DataFrame):
            if data.stk_pos > data.stk_upper > 0:
                return '股票突破上限'
            if data.stk_p_fund_pos > data.stk_p_fund_upper > 0:
                return '股票(含ETF)突破上限'
            if 0 < data.stk_pos < data.stk_lower:
                return '股票低于下限'
            return ''

        def _cb_alert(data: pd.DataFrame):
            if data.cb_pos > data.cb_upper > 0:
                return '转债突破上限'
            if 0 < data.cb_pos < data.cb_lower:
                return '转债低于下限'
            return ''

        def _fund_alert(data: pd.DataFrame):
            if data.stk_fund_pos > data.stk_fund_upper > 0:
                return '股票ETF突破上限'
            return ''

        def _eq_alert(data: pd.DataFrame):
            if data.eq_pos > data.eq_upper > 0:
                return '含权资产突破上限'
            return ''

        res = [_stk_alert(data), _cb_alert(data), _fund_alert(data), _eq_alert(data)]
        res_adj = [i for i in res if i != '']
        res_adj = '&'.join(res_adj) if len(res_adj) > 0 else '未触警'
        return res_adj

    def _dur_thold_adj(self, x):
        adj_ptf = ['010923.OF', '021241.OF']
        if x.portfolio_code not in adj_ptf:
            return x.dur_thold
        eq_pos = x.stk_p_fund_pos + 0.5 * x.cb_pos
        if eq_pos > 0.15:
            return 20
        if 0.05 < eq_pos <= 0.15:
            return 15
        if eq_pos <= 0.05:
            return 10
        return 0

    def _dur_alert_rule(self, data: pd.DataFrame):
        cond_dur = data.dur_fund > data.dur_thold > 0
        cond_credit_dur = data.credit_dur_fund > data.credit_dur_thold > 0
        if cond_dur and cond_credit_dur:
            return '组合久期与信用资产久期均破阈值'
        if cond_dur:
            return '组合久期破阈值'
        if cond_credit_dur:
            return '信用资产久期破阈值'
        return '未触警'

    def _load_equity_dd_budget(self):
        """
        加载权益公募回撤预算
        """
        sql_thold = sqls_config['rm_threshold_eq']['Sql']
        equity_dd_budget = self.db_risk.read_sql(sql_thold)
        self.equity_dd_budget = equity_dd_budget

    def get_fund_daily_ret(self, portfolio_codes: list, beg_date: str, end_date: str):
        """
        计算基金日度收益
        @param portfolio_codes:
        @param beg_date:
        @param end_date:
        @return: index: pd.datetime
        """
        db_risk = DbFactor(config['data_base']['QUANT']['url'])
        fund_ret = db_risk.get_factor('ret_adj', code_list=portfolio_codes,
                                   beg_date=f'\"{beg_date}\"',
                                   end_date=f'\"{end_date}\"', field='a.portfolio_code')
        fund_ret = fund_ret.sort_values(by=['portfolio_code', 'd_date']).rename(columns={'return': 'ret'})
        fund_ret.index = pd.to_datetime(fund_ret['d_date'])
        return fund_ret

    @ classmethod
    def get_wind_industry_index_ret(cls, benchmark_code: list, beg_date, end_date):
        """
        计算基准指数收益率
        @param benchmark_codes:
        @return:
        """
        db_wind = DbFactor(config['data_base']['WIND']['url'])
        index_ret = db_wind.get_factor('wind_industry_index', code_list=benchmark_code, beg_date=f'\"{beg_date.replace("-", "")}\"',
                                     end_date=f'\"{end_date.replace("-", "")}\"', field='s_info_windcode')
        index_ret['trade_dt'] = index_ret['trade_dt'].apply(lambda x: convert_date(x,
                                            from_='%Y%m%d', to_='%Y-%m-%d'))
        index_ret = index_ret.drop_duplicates(subset=['trade_dt', 'index_code'], keep='last')
        return index_ret
    
    @ classmethod
    def _cal_comp_bench_ret(cls, trace_bench_codes: pd.DataFrame, beg_date, end_date, split_by=','):
        """
        计算基准指数区间收益，对于成立时间晚于beg_date的做调整
        @param trace_bench_info: ['benchmark_code', 'benchmark_wgt'] 复合指数：成分指数之间用,隔开
        @param beg_date:
        @param end_date:
        @return:   
        """
        # 指数成分解析
        codes_struct = {}
        codes_all = []
        for i in range(0, trace_bench_codes.shape[0]):
            code_i = trace_bench_codes.benchmark_code.iloc[i]
            wgt_i = trace_bench_codes.benchmark_wgt.iloc[i]
            codes_struct[code_i] = dict(filter(lambda x: x[1] != '', dict(zip(code_i.split(split_by),
                                                                              wgt_i.split(split_by))).items()))
            codes_all = codes_all + list(codes_struct[code_i].keys())
        df_codes_struct = pd.DataFrame(codes_struct).stack().to_frame('benchmark_wgt').astype('float')
        df_codes_struct = df_codes_struct.reset_index().rename(columns={'level_0': 'benchmark_code', 'level_1': 'index_cmp_name'})
        df_codes_struct = pd.merge(df_codes_struct,
                          trace_bench_codes[['benchmark_code', 'setup_date']].rename(columns={'benchmark_code': 'index_cmp_name'}),
                                   on='index_cmp_name', how='left')
        df_codes_struct['start_dt_offset'] = df_codes_struct['setup_date'].apply(
                                        lambda x: convert_date(
                                            TradeDays.offset_date(int(x.replace('-', '')), -1, 'D')[0],
                                            from_='%Y%m%d', to_='%Y-%m-%d'))

        # 成分指数日收益率， 处理成A股交易日
        beg_date_offset = TradeDays.offset_date(int(beg_date.replace('-', '')), -1, 'D')[0]
        beg_date_offset = convert_date(str(beg_date_offset), from_='%Y%m%d', to_='%Y-%m-%d')
        trade_dts = TradeDays.get_adjustdate(int(beg_date_offset.replace('-', '')), int(end_date.replace('-', '')), ('D', 1))
        trade_dts['trade_dt'] = trade_dts['trade_dt'].apply(lambda x: convert_date(str(x),
                                                                                   from_='%Y%m%d', to_='%Y-%m-%d'))
        index_ret = cls.get_wind_industry_index_ret(codes_all, beg_date_offset, end_date)
        index_daterange = index_ret.groupby('index_code')['trade_dt'].agg(['min', 'max'])
        if any(index_daterange['min'] > beg_date):
            logger.warning(f"存在指数行情起始日期晚于beg_date! 请注意")
        index_ret_mat = index_ret.pivot(index='trade_dt', columns='index_code', values='ret')
        index_cumret_mat = (1 + index_ret_mat.fillna(0)).cumprod() - 1
        index_cumret_mat = index_cumret_mat.reindex(index=trade_dts['trade_dt'])
        index_ret_adj = (1 + index_cumret_mat).pct_change().stack().to_frame('ret').reset_index()

        # # 合成复合指数
        comp_index_ret = pd.merge(df_codes_struct.rename(columns={'benchmark_code': 'index_code'}),
                                  index_ret_adj[['trade_dt', 'index_code', 'ret']], left_on='index_code',
                                    right_on='index_code', how='right')
        comp_index_ret = comp_index_ret.sort_values(by=['index_code', 'trade_dt', 'ret'])
        # 剔除产品成立以前的数据
        comp_index_ret = comp_index_ret.loc[comp_index_ret['trade_dt'] >= comp_index_ret['setup_date']]
        comp_index_ret['comp_cumret'] = comp_index_ret.groupby(['index_cmp_name', 'index_code'])['ret'].transform(
                                                lambda x: (1 + x).cumprod() - 1)
        comp_index_ret['wgted_ret'] = comp_index_ret['ret'] * comp_index_ret['benchmark_wgt']
        # 先对日收益率加权，然后加总为合成指数日收益率
        index_cumret = comp_index_ret.groupby(['index_cmp_name', 'trade_dt'])['wgted_ret'].sum().to_frame('ret')
        # comp_index_ret['wgted_comp_cumret'] = comp_index_ret['comp_cumret'] * comp_index_ret['benchmark_wgt']
        # index_cumret = comp_index_ret.groupby(['index_cmp_name', 'trade_dt'])['wgted_comp_cumret'].sum().to_frame('cumret')
        # index_cumret['ret'] = index_cumret.groupby(level=0)['cumret'].transform(lambda x: (1 + x).pct_change())
        index_cumret = index_cumret.reset_index()
        index_cumret = pd.merge(index_cumret, trace_bench_codes[['benchmark_code', 'setup_date']].rename(columns={'benchmark_code': 'index_cmp_name'}),
                                on='index_cmp_name', how='left')
        index_cumret = index_cumret.rename(columns={'index_cmp_name': 'index_name'})
        return index_cumret.reindex(columns=['trade_dt', 'index_name', 'ret'])

    def _alert_dd_equity_relative(self, budget):
        """
        计算超额收益回撤指标
        """
        # 区间产品收益率
        fund_ret = budget.apply(lambda x: self.get_fund_daily_ret(x.portfolio_code, x.period_bgdate, self.basedate)[:], axis=1)
        fund_ret_daily = pd.DataFrame()
        for i in range(0, fund_ret.shape[0]):
            fund_ret_daily = pd.concat([fund_ret_daily, fund_ret[i]])

        # 区间基准收益率
        bench_ret_daily = self._cal_comp_bench_ret(budget[['benchmark_code', 'benchmark_wgt', 'setup_date']],
                                                   budget['period_bgdate'].min(), self.basedate)

        # 基金收益和基准收益对齐
        port_ret_daily = pd.merge(fund_ret_daily, budget[['portfolio_code', 'benchmark_code']], on='portfolio_code', how='left')
        port_ret_daily = pd.merge(port_ret_daily,
                                bench_ret_daily.rename(columns={'trade_dt': 'd_date', 'index_name': 'benchmark_code'}),
                                  on=['d_date', 'benchmark_code'], how='left', suffixes=['_port_daily', '_bench_daily'])
        port_ret_daily = port_ret_daily.sort_values(by=['portfolio_code', 'd_date'])
        port_cumret = port_ret_daily.groupby(['portfolio_code'])[['ret_port_daily', 'ret_bench_daily']].transform(
            lambda x: (1 + x).cumprod() - 1).rename(
            columns={'ret_port_daily': 'port_cumret', 'ret_bench_daily': 'bench_cumret'})
        port_ret_daily = pd.concat([port_ret_daily, port_cumret], axis=1)
        # 超额收益
        port_ret_daily['exc_cumret'] = port_ret_daily['port_cumret'] - port_ret_daily['bench_cumret']
        # 计算超额收益的最大回撤
        # port_ret_daily['cummax'] = port_ret_daily.groupby('portfolio_code')['exc_cumret'].transform('cummax')
        # port_ret_daily['dd'] = (1 + port_ret_daily['exc_cumret']) / (1 + port_ret_daily['cummax']) - 1
        # port_ret_daily['mdd'] = port_ret_daily.groupby('portfolio_code')['dd'].transform('cummin')
        retain_cols = ['d_date', 'portfolio_code', 'c_fundname', 'benchmark_code', 'port_cumret', 'bench_cumret',
                       'exc_cumret']
        exc_ret = port_ret_daily.reindex(columns=retain_cols).rename(columns={'d_date': 'trade_dt'})
        exc_ret = pd.merge(exc_ret,
                           budget[['portfolio_code', 'alert_period', 'period_bgdate', 'ranking_list']],
                           on='portfolio_code',
                           how='left')

        return exc_ret

    def _check_dd_budget(self, exc_ret, budget):
        """
        check预警
        """
        df = pd.merge(budget.reindex(columns=['portfolio_code', 'dmsion', 'dd_alert', 'rank_alert']), exc_ret,
                      on='portfolio_code', how='right')

        if budget['rank_alert'].notnull().sum() > 0:
            # 有排名预算的: 提取排名情况
            db_risk = DbFactor(config['data_base']['QUANT']['url'])
            fund_rank = db_risk.get_factor('internal_port_perf', code_list=budget['portfolio_code'].to_list(),
                                       beg_date=f'\"{exc_ret["trade_dt"].min()}\"',
                                       end_date=f'\"{self.basedate}\"', field='a.portfolio_code')
            # 收益和排名合并
            df = pd.merge(df, fund_rank[['trade_dt', 'portfolio_code', 'return_period', 'period_bgdate',
                'ranking_list', 'return_rankpct']].rename(columns={'return_period': 'alert_period'}),
                on=['trade_dt', 'portfolio_code', 'alert_period', 'period_bgdate', 'ranking_list'],
                how='left')

        # 检查子类触警情况: 超额收益回撤、排名预算
        df['if_alert_exc_dd'] = df.apply(lambda x: (x.exc_cumret < - x.dd_alert) * 1 if x.dmsion == '超额收益' else 0,
                                            axis=1)
        df['if_alert_rank'] = df.apply(
            lambda x: (x.return_rankpct > x.rank_alert) * 1 if ~np.isnan(x.rank_alert) else 0, axis=1)

        # 组合触警情况: 同时触警
        df['alert_level'] = df.apply(lambda x: '触警' if x.if_alert_exc_dd * x.if_alert_rank else '未触警', axis=1)

        df = pd.merge(df,
                     budget[['portfolio_code', 'risk_budget']], on='portfolio_code', how='left')
        df['risk_budget'] = df['risk_budget'].where(df['alert_level'] != '未触警')
        return df.rename(columns={'risk_budget': 'risk_budget_info'})

    def alert_dd_equity(self):
        """
        权益公募回撤预警: 卓越臻选微盘策略（停用）
        """
        # 加载回撤预算
        self._load_equity_dd_budget()
        self.equity_dd_budget['period_bgdate'] = self.equity_dd_budget['alert_period'].replace(
            {'今年以来': self.this_year_first_cdate})
        budget_relative = self.equity_dd_budget[self.equity_dd_budget['dmsion'] == '超额收益']

        # 计算指标
        if budget_relative.shape[0] > 0:
            exc_ret = self._alert_dd_equity_relative(budget_relative)
            alert_detail = self._check_dd_budget(exc_ret, budget_relative)

        self.alert_dd_equity_rpt = alert_detail[alert_detail['alert_level'] != '未触警']

        # 插入数据库
        alert_detail = alert_detail[alert_detail['trade_dt'] == self.basedate]
        cond = and_(column("trade_dt") >= alert_detail['trade_dt'].min(),
                    column("trade_dt") <= alert_detail['trade_dt'].max(),
                    column("portfolio_code").in_(list(set(alert_detail['portfolio_code'])))
                    )
        table_names = ['RM_RESULT_DD_EQUITY']
        datas = [alert_detail.drop_duplicates()]
        logger.info(f"开始落库，表名{table_names}...")
        db_factor = DbFactor(config['data_base']['QUANT']['url'])
        db_factor.insert2db(table_name=table_names, res_list=datas, cond=cond)
        db_factor.close()

    def calc_all_alert(self):
        '''
        汇总所有监控
        :return:
        '''
        # self.alert_dd_equity()
        self.duration_alert()
        self.ptm_alert()
        self.drawdown_alert()
        self.winratio_alert()
        self.alert_dur()
        self.alert_dd()
        self.alert_position()
        self.alert_rating()
        logger.info('%s所有投委会决议相关监控已全部执行。'%self.basedate)

    def saveAll(self, save_path):
        '''
        将所有预警底稿存入Excel
        :param save_path: string format, 预警结果文件的存储路径
        :return:
        '''
        writer = pd.ExcelWriter(os.path.join(save_path, '%s_RiskAlert_add.xlsx'%self.basedate.replace('-', '')))
        self.ptm_alert.to_excel(writer, sheet_name='rc_alert_ptm', index=False)
        self.winratio_alert.to_excel(writer, sheet_name='rc_alert_winratio', index=False)
        # self.alert_dd_equity_rpt.to_excel(writer, sheet_name='rc_alert_dd_equity', index=False)
        self.ptm_port.to_excel(writer, sheet_name='剩余期限基础数据', index=False)
        self.rm_dur_rpt.to_excel(writer, sheet_name="久期预警结果", index=False)
        self.rm_dd_rpt.to_excel(writer, sheet_name="回撤预警结果", index=False)
        self.rm_pos_rpt.to_excel(writer, sheet_name="仓位预警结果", index=False)
        self.rm_rating.to_excel(writer, sheet_name="评级预警结果", index=False)
        writer.save()