'''
Description: 
Author: Wangp
Date: 2021-01-25 16:35:49
LastEditTime: 2021-06-17 18:06:51
LastEditors: Wangp
'''
import os

import pandas as pd
import datetime
from WindPy import w

from .trade_dt.date_utils import retrieve_n_tradeday, get_past_qtr_date
from .utils_ri.retrieveAttribution import *
from .utils_ri import RiskIndicators
from .db import OracleDB, sqls_config
from .db.db_utils import convert_columns
from .settings import config, DIR_OF_MAIN_PROG
from .utils.log_utils import logger
from .utils.send_email_standard import format_body, send_email

w.start()

class RiskIndicator_mg(RiskIndicators):
    def __init__(self, t, save_path, ptf_codes=None):
        self.basedate = t
        self.save_path = save_path
        self._format_ptf_codes(ptf_codes)
        self.db_risk = OracleDB(config['data_base']['QUANT']['url'])

        yy = str(pd.to_datetime(self.basedate).year)
        mm = str(pd.to_datetime(self.basedate).month)
        self.path_repo = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\Collaterals\%s\\'%self.basedate.replace('-', '')
        self.path_inrt = r'\\shaoafile01\RiskManagement\4. 静态风险监控\4.6 分级授权限额管理\2.信用债\%s年%s月\%s\\'%(yy, mm, self.basedate.replace('-', ''))

        self._loadFile()
        self._loadCoding()
        self._loadHoldings(self.save_path)
        self._loadInnerRating()
        self._loadBenchMarkTarget()
        self._loadNavData()
        self._loadTnPtf()

    def _dealNavData(self):
        '''取单位净值数据：公募取今年以来，专户取当前考核期'''
        if '_bchTarget' not in dir(self):
            self._loadBenchMarkTarget()
        nav = self.nav_f.copy()
        bch_m = self.val.loc[self.val['L_FUNDTYPE'] == '1', ['C_FUNDNAME']].drop_duplicates()   # 公募取今年以来
        bch_m['考核期开始日'] = pd.to_datetime(self.basedate[:4] + '-01-01')
        bch_m['考核期结束日'] = pd.to_datetime(self.basedate)
        bchTarget = self._bchTarget.append(bch_m, sort=False)                                   # 专户取考核期

        data = pd.merge(nav, bchTarget, on='C_FUNDNAME', how='left')
        data = data[(data['D_DATE'] >= data['考核期开始日']) & (data['D_DATE'] <= data['考核期结束日'])].copy()

        return data

    def _loadCoding(self):
        self.data_prod = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\产品一览表.xlsx', engine='openpyxl')
        self._fundName = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\CodingTable.xlsx', sheet_name='产品基础信息', engine='openpyxl').rename(columns={'产品名称': 'C_FULLNAME', '估值系统简称': 'C_FUNDNAME', 'O32产品名称': 'C_FUNDNAME_o32'})
        self.data_prod = pd.merge(self.data_prod, self._fundName[['C_FULLNAME', 'C_FUNDNAME', 'C_FUNDNAME_o32']], left_on=['产品名称'], right_on=['C_FULLNAME'], how='left')
        self._bchPeriod = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\存量专户产品业绩比较基准.xlsx', engine='openpyxl')

    def _loadInnerRating(self):
        self._innRating_map = {'B+': '交易级', 
                               'B': '交易级', 
                               'B-': '交易级', 
                               'CCC+': '投机级', 
                               'CCC': '投机级'}

        q = sqls_config['bond_inner_rating']['Sql']
        self._innerRating = self.db_risk.read_sql(q)[['公司名称', '评级级别', '评级展望']].copy()
        self._innerRating['内评分类'] = self._innerRating['评级级别'].map(self._innRating_map).fillna('*')

    def _loadBenchMarkTarget(self):
        self._bchTarget = pd.read_excel(r'\\shaoafile01\RiskManagement\1. 基础数据\存量专户产品业绩比较基准.xlsx', engine='openpyxl').rename(columns={'估值系统简称': 'C_FUNDNAME'})
        self._bchTarget['存续期限'] = [x.days / 365 for x in self._bchTarget['产品到期日'] - self._bchTarget['产品设立日期']]
        self._bchTarget = self._bchTarget[['C_FUNDNAME', '绝对收益', '存续期限', '考核期比较基准', '考核期开始日', '考核期结束日', '资产委托人']].dropna(subset=['考核期比较基准'])

    def _loadTnPtf(self):
        tn_ptf = self.db_risk.read_sql(sqls_config['tn_portfolio_name']['Sql'])
        self.tn_ptf_d = {}
        for i in range(tn_ptf['n_days'].min(), tn_ptf['n_days'].max()+1):
            self.tn_ptf_d[i] = tn_ptf[tn_ptf['n_days'] == i]['c_fundname'].to_list()

    def _retrieve_data_tn(self, sql: str):
        '''所有产品均取准确日期的持仓信息'''
        data = self.db_risk.read_sql(sql % self.basedate)
        for n, ptf_codes in self.tn_ptf_d.items():
            t_n = self.retrieve_n_tradeday(self.basedate, n)
            temp = self.db_risk.read_sql(sql % t_n)
            new_df = temp[temp['c_fundname'].isin(ptf_codes)].copy()
            keep_df = data[~data['c_fundname'].isin(ptf_codes)].copy()
            data = pd.concat([keep_df, new_df], ignore_index=True)

        if 'd_date' in data.columns:
            data['d_date'] = self.basedate
        return data

    def previous_tradeday(self, t):
        '''取当前日期的前一个交易日'''
        sql_raw = sqls_config['previous_trade_date']['Sql']
        prev_td = self.db_risk.read_sql(sql_raw.format(t=t))['c_date'].iloc[0]
        return prev_td

    def latest_tradeday(self, t):
        '''取当前日期的前一个交易日'''
        sql_raw = sqls_config['lastest_trade_date']['Sql']
        prev_td = self.db_risk.read_sql(sql_raw.format(t=t))['c_date'].iloc[0]
        return prev_td

    def retrieve_n_tradeday(self, t, n):
        '''
        取给定日期过去第n个交易日日期
        :param t: string/datetime/timestamp, 需查询的给定日期
        :param n: int, 日期偏移量, 仅支持向历史偏移
        :return: string, 过去第n个交易日日期
        '''
        if type(t) != str:
            t = t.strftime('%Y-%m-%d')
        q = sqls_config['past_tradeday']['Sql'] % t
        tradeday = self.db_risk.read_sql(q).sort_values(by=['c_date'])
        return tradeday.iloc[(-1) * (n + 1)][0]

    def is_end_of_quarter(self, t):
        return True if t[-5:] in ["03-31", "06-30", "09-30", "12-31"] else False

    def getDuraChg(self):
        '''计算组合久期变化，分别为1周、1个月、3个月'''
        t = self.basedate
        t_1 = w.tdaysoffset(-1, t, "Period=M").Data[0][0].strftime('%Y-%m-%d')
        t_3 = w.tdaysoffset(-3, t, "Period=M").Data[0][0].strftime('%Y-%m-%d')
        t_1w = w.tdaysoffset(-1, t, "Period=W").Data[0][0].strftime('%Y-%m-%d')

        q = sqls_config['rc_mr_holding_t']['Sql']
        data_t = convert_columns(self.db_risk.read_sql(q%t))[['C_FUNDNAME', 'D_DATE', 'avgDuration_port']].rename(columns={'avgDuration_port': 'avgDuration_port_t'})
        data_t1 = convert_columns(self.db_risk.read_sql(q%t_1))[['C_FUNDNAME', 'D_DATE', 'avgDuration_port']].rename(columns={'avgDuration_port': 'avgDuration_port_1M'})
        data_t3 = convert_columns(self.db_risk.read_sql(q%t_3))[['C_FUNDNAME', 'D_DATE', 'avgDuration_port']].rename(columns={'avgDuration_port': 'avgDuration_port_3M'})
        data_t1w = convert_columns(self.db_risk.read_sql(q%t_1w))[['C_FUNDNAME', 'D_DATE', 'avgDuration_port']].rename(columns={'avgDuration_port': 'avgDuration_port_1W'})

        res = pd.merge(data_t, data_t1.drop(columns=['D_DATE']), on=['C_FUNDNAME'], how='left')
        res = pd.merge(res, data_t3.drop(columns=['D_DATE']), on=['C_FUNDNAME'], how='left')
        res = pd.merge(res, data_t1w.drop(columns=['D_DATE']), on=['C_FUNDNAME'], how='left')
        res['1M_chg'] = res['avgDuration_port_t'] - res['avgDuration_port_1M']
        res['3M_chg'] = res['avgDuration_port_t'] - res['avgDuration_port_3M']
        res['1W_chg'] = res['avgDuration_port_t'] - res['avgDuration_port_1W']

        q = sqls_config['rc_derivt']['Sql']%t
        ctd_dura = convert_columns(self.db_risk.read_sql(q))
        res = pd.merge(res, ctd_dura[['C_FUNDNAME', 'D_DATE', 'CTD_EQUI_DURA']], on=['C_FUNDNAME', 'D_DATE'], how='left')
        res['port_dura_ctd'] = res[['avgDuration_port_t', 'CTD_EQUI_DURA']].sum(axis=1)
        res = res.drop(columns=['CTD_EQUI_DURA'])
        res['D_DATE'] = pd.to_datetime(res['D_DATE'])
        return res

    def getInnerRatingDist(self):
        '''各组合投资级、交易级（低评级）内评分布(剔除CRMW保护的债券部分)'''
        data = self._retrieve_data_tn(sqls_config['dpe_portfoliobond']['Sql'])
        data = data[~data['windl2type'].isin(['可转债', '可交换债', '可分离转债存债', '政策银行债'])].copy()
        data = pd.merge(data, self._innerRating, left_on='issuer', right_on='公司名称', how='left')

        # 从投资级债券中剔除被CRMW保护的部分
        crm_df = self.db_risk.read_sql(sqls_config['crwm_protect']['Sql'] % self.basedate)
        data = pd.merge(data, crm_df, how='left', left_on=['c_fullname', 'code'],
                        right_on=['c_fullname', 'crm_subjectcode']).fillna({'crm_mount': 0})
        data['f_assetratio'] = data['f_assetratio'] * (1 - data['crm_mount'] / data['f_mount'])

        res = pd.pivot_table(data, values='f_assetratio', columns=['内评分类'], index=['c_fundname', 'd_date'], aggfunc='sum').reindex(columns=['交易级', '投机级']).reset_index()

        return res.rename(columns={'c_fundname': 'C_FUNDNAME', 'd_date': 'D_DATE'})

    def getMMFIdx(self):
        '''货基专项指标，如七日收益率、流动受限占比等指标'''

        # 估值表下方指标
        col_d = {'c_fundname': 'C_FUNDNAME', 'dev_degree': '影子偏离度', 'anal_yld_7d': '七日年化',
                 'ptfl_avg_matu_day': '平均到期日', 'risk_fr_5d_matu_ast_nav_ratio': '5日金融工具占比'}
        df_ptf = self.db_risk.read_sql(sqls_config['mg_mmf_ptf_level']['Sql'].format(t=self.basedate))
        df_share = self.db_risk.read_sql(sqls_config['mg_mmf_share_level']['Sql'].format(t=self.basedate))
        main_share = pd.Series(config['mmf_share_map'], name='share_type').reset_index().rename(columns={'index': 'portfolio_code'})
        main_share = main_share.merge(df_share, how='left', on=['portfolio_code', 'share_type'])
        res1 = df_ptf.merge(main_share, how='left', on=['c_date', 'portfolio_code'])
        res1 = res1.reindex(columns=['prod_code'] + list(col_d.keys())).rename(columns=col_d)
        res1['影子偏离度'] = res1['影子偏离度'].apply(lambda x: str(round(x*100, 4)) + '%')
        res1['七日年化'] = res1['七日年化'].apply(lambda x: str(round(x * 100, 4)) + '%')

        # 流通受限资产&低评级占比
        data = self.bond_holdings[(self.bond_holdings['L_FUNDTYPE'] == 1) & (self.bond_holdings['L_FUNDKIND2'] == 5)].copy()
        res2 = data[data['WINDL1TYPE'] == '资产支持证券'].groupby(['C_FUNDNAME', 'D_DATE'])['F_ASSETRATIO'].sum().rename('流通受限占比').reset_index()
        res3 = data[~data['LATESTISSURERCREDITRATING2'].isin(['AAA']) & (data['INDUSTRY_SW'] != '行业_利率债')].groupby(['C_FUNDNAME', 'D_DATE'])['F_ASSETRATIO'].sum().rename('主评低于AAA占比').reset_index()

        # 持有人
        sql_holder = sqls_config["mg_mmf_investors"]["Sql"]
        res4 = self.db_risk.read_sql(sql_holder.format(t=self.basedate))

        # 业绩状态
        prod_codes = main_share['prod_code'].to_list()
        sql_ret = sqls_config["mg_mmf_pf"]["Sql"]
        ret_df1 = self.db_risk.read_sql(sql_ret.format(t=self.basedate, period="今年以来", code=tuple(prod_codes)))
        ret_df2 = self.db_risk.read_sql(sql_ret.format(t=self.basedate, period="日度", code=tuple(prod_codes)))
        res5 = ret_df1.merge(ret_df2, how="left", on=['prod_code', 'ranking_list'])

        keep_cols = list(col_d.values()) + ['D_DATE', '流通受限占比', '主评低于AAA占比']
        keep_cols += ['c_fundname', 'ranking_list', 'RET_今年以来', 'MID_RET_今年以来', 'RET_RANK_今年以来', 'RET_日度',
                      'MID_RET_日度', 'RET_RANK_日度', '个人投资者', '机构投资者', '前10大投资者']
        res = pd.merge(res1, res2, on=['C_FUNDNAME'], how='left').fillna({'D_DATE': datetime.datetime.strptime(self.basedate, '%Y-%m-%d')})
        res = pd.merge(res, res3[['C_FUNDNAME', '主评低于AAA占比']], on=['C_FUNDNAME'], how='left')
        res = pd.merge(res, res4, left_on='C_FUNDNAME', right_on="c_fundname", how='left')
        res = pd.merge(res, res5, on='prod_code', how='left').reindex(columns=keep_cols)

        return res

    def filter_overdue_secs(self):
        '''计算各专户组合的期限错配情况'''
        data = self.bond_holdings[(self.bond_holdings['L_FUNDTYPE'] == 3) & ~self.bond_holdings['WINDL2TYPE'].isin(['可转债', '可交换债', '可分离转债存债'])].copy()
        data['下一回售日'] = pd.to_datetime(data['REPURCHASEDATE_wind'])
        data.loc[data['下一回售日'] <= pd.to_datetime(self.basedate), '下一回售日'] = np.nan
        data['债券到期日'] = data['下一回售日'].fillna(value=data['到期日期'])

        data1 = pd.merge(data, self.data_prod[['C_FUNDNAME', '产品到期日']], on='C_FUNDNAME', how='left')
        data1['错配期限'] = [x.days for x in (data1['债券到期日'] - data1['产品到期日'])]
        data1['错配期限_w'] = data1['错配期限'] * data1['F_ASSETRATIO'] / (100*365)
        data1['dura_w'] = data1['F_ASSETRATIO'] * data1['MODIDURA_CNBD'].fillna(0) / 100
        data1['overdue_gp'] = pd.cut(data1['错配期限']/365, bins=[0, 0.5, 1, np.inf], labels=['6M', '6Mto1Y', '1Yabove'])

        self.holdings_overdue = data1.copy()

    def getOverdueSecs(self):
        '''计算各组合的加权错配期限'''
        if 'holdings_overdue' not in dir(self):
            self.filter_overdue_secs()
        data1 = self.holdings_overdue.copy()
        res1 = data1[data1['债券到期日'] > data1['产品到期日']].groupby(['C_FUNDNAME', 'D_DATE'])['F_ASSETRATIO'].sum().rename('期限错配占比').reset_index()
        res2 = data1[data1['债券到期日'] > data1['产品到期日']].groupby(['C_FUNDNAME', 'D_DATE'])['错配期限_w'].sum().rename('平均错配时间').reset_index()
        res = pd.merge(res1, res2, on=['C_FUNDNAME', 'D_DATE'], how='left')
        return res

    def _chgAssetPrice(self, x):
        y = round(x / 1e8, 2)
        if y >= 1:
            y = str(int(y)) + '亿'
        else:
            y = str(int(y * 1e4)) + '万'        
        return y

    def _mergeLowLiq(self, x):
        res = '，'.join(y for y in [x['c_subname_bsh'], x['issuemethod'], x['industry_sw'], self._chgAssetPrice(x['f_asset']), x['rate_latestmir_cnbd'], '估值' + str(round(x['yield_cnbd'],1)) + '%', '剩余期限' + str(round(x['ptmyear'],2)) + '年'])
        return res

    def getLowLiqAsset(self):
        '''期限长于专户到期的低流动性资产：定向工具、私募债、低评级债等'''
        data = self._retrieve_data_tn(sql=sqls_config['dpe_portfoliobond']['Sql'])
        data['LowLiq'] = ((data['windl1type'].isin(['定向工具']) | (data['issuemethod'] == '私募') | data['bond_ratingclass'].isin(['低评级'])) & ~data['windl2type'].isin(['可转债', '可交换债', '可分离转债存债'])).astype(int)
        data['下一回售日'] = pd.to_datetime(data['REPURCHASEDATE_wind'])
        data.loc[data['下一回售日'] <= pd.to_datetime(self.basedate), '下一回售日'] = np.nan
        data['债券到期日'] = data['下一回售日'].fillna(value=data['maturitydate'])
        data = pd.merge(data, self.data_prod[['C_FUNDNAME', '产品到期日']], left_on='c_fundname', right_on='C_FUNDNAME', how='left')

        data1 = data[(data['LowLiq'] == 1) & (data['债券到期日'] > data['产品到期日'])].copy()
        if data1.shape[0] == 0:
            return pd.DataFrame(columns=['C_FUNDNAME', 'D_DATE', '低流动性占比', '低流动性明细'])

        data1['issuemethod'] = data1['issuemethod'].fillna('-')
        data1['industry_sw'] = data1['industry_sw'].fillna('无行业分类')
        res0 = data1.groupby(['c_fundname', 'd_date'])['f_assetratio'].sum().rename('低流动性占比').reset_index()
        res1 = data1.groupby(['c_fundname', 'd_date']).apply(lambda x: ';\n'.join(self._mergeLowLiq(item) for idx, item in x.iterrows())).rename('低流动性明细').reset_index()
        res = pd.merge(res0, res1, on=['c_fundname', 'd_date'], how='left')

        return res.rename(columns={'c_fundname': 'C_FUNDNAME', 'd_date': 'D_DATE'})

    def _calcHWM(self, x):
        # 传入净值序列
        y = x.diff()
        if x[(y >= 0) & (y.shift(-1) < 0)].shape[0] == 0:            # 即净值一路上行
            return np.nan, np.nan
        else:
            idx_hwm = x[(y >= 0) & (y.shift(-1) < 0)].idxmax()
            hwm = x.loc[idx_hwm]
            hwm_t = idx_hwm.iloc[-1].strftime('%Y-%m-%d')

            return hwm, hwm_t

    # 回撤，若为0则说明回撤HWM已恢复or净值一路上行
    # DrawDown = S(t) / HWM(:t) - 1
    def DrawDown_t(self, netvalue_df):
        hwm, hwm_t = self._calcHWM(netvalue_df['NAV'])
        if hwm is np.nan:
            return np.nan
        else:
            return hwm_t

    def ReturnRate(self, netvalue_df):
        '''区间收益率'''
        # returnRate = netvalue_df['NAV'].iloc[-1] / netvalue_df['NAV'].iloc[0] - 1
        returnRate = (netvalue_df['ret']+1).product() - 1
        
        return returnRate

    def getReturnRate_p(self):
        '''计算专户当前考核期的区间收益率'''
        data_nav = pd.merge(self.nav_f, self._bchTarget, on=['C_FUNDNAME'], how='right')
        data_nav = data_nav[(data_nav['D_DATE'] >= data_nav['考核期开始日']) & (data_nav['D_DATE'] <= data_nav['考核期结束日'])
                            & (data_nav['D_DATE'] <= pd.to_datetime(self.basedate))].copy()

        # 从data_nav中删除t+n估值产品的预估数
        del_dates = [self.basedate]
        for n, ptf_names in self.tn_ptf_d.items():
            if n > 1:
                del_dates.append(self.previous_tradeday(del_dates[-1]))
            data_nav = data_nav[~(data_nav['C_FUNDNAME'].isin(ptf_names) & data_nav['D_DATE'].isin(del_dates))].copy()

        grouped = data_nav.groupby(['C_FUNDNAME', '考核期比较基准', '资产委托人'])
        ret_cum = grouped.apply(self.ReturnRate).rename('ReturnRate_cum').reset_index()
        day_cum = grouped.apply(lambda x: (x['D_DATE'].max() - x['D_DATE'].min()).days + 1).rename('days_cum').reset_index()
        # calc_date = grouped.apply(lambda x: x['D_DATE'].max()).rename('calc_date').reset_index()

        res = pd.merge(ret_cum, day_cum, on=['C_FUNDNAME', '考核期比较基准', '资产委托人'], how='left')
        res = pd.merge(res, self._bchTarget[['C_FUNDNAME', '考核期开始日', '考核期结束日']], on=['C_FUNDNAME'])
        # res = pd.merge(res, calc_date, on=['C_FUNDNAME', '考核期比较基准', '资产委托人'], how='left')
        basedate = datetime.datetime.strptime(self.basedate, '%Y-%m-%d')
        res['剩余期限'] = res.apply(lambda x: (x.考核期结束日 - basedate).days, axis=1)

        return res

    def getIndexFundsDev(self, t, days=242):
        '''指数基金正偏离天数占比'''
        # 获取指数基金偏离数据（分级基金取A份额）
        bg_date = self.retrieve_n_tradeday(t, n=days)
        q_dev = sqls_config['index_fund_dev']['sql']
        df_dev = self.db_risk.read_sql(q_dev.format(t=t, t0=bg_date))

        # 统计正偏离天数占比
        func = lambda x: x[x['a_dev'] > 0].count()/x['a_dev'].count() if x['a_dev'].count() > 0 else np.nan
        res = df_dev.groupby('c_fundname').apply(func)['a_dev'].reset_index().rename(columns={'a_dev': '正偏离天数占比'})

        return res

    def getSoldOutSecs(self):
        '''信评出具“择机卖出”的债券清单'''
        sql_sold = sqls_config['sold_out_secs']['Sql']    
        self.soldOutsecs = self.db_risk.read_sql(sql_sold)
        self.soldOutsecs = self.soldOutsecs.loc[:, :'行业二级名称']

    def check_bond_added(self):
        '''检查当日新增/移除预警的债券清单'''
        w.start()
        t_1 = self.retrieve_n_tradeday(self.basedate, 1)
        q = sqls_config['rc_alert_cr']['Sql']%t_1
        cr_t1 = convert_columns(self.db_risk.read_sql(q))[['C_FUNDNAME', 'D_DATE', 'CODE', 'SEC_NAME', 'ASSET_PAR']].copy()
        res = pd.merge(self.cr_alert[['C_FUNDNAME', 'CODE', 'ASSET_PAR']], cr_t1, on=['C_FUNDNAME', 'CODE'], how='outer')
        res['new_or_not'] = res['ASSET_PAR_y'].isna().map({True: 1,  False: 0})
        res['remove_or_not'] = res['ASSET_PAR_x'].isna().map({True: 1,  False: 0})
        res = res.rename(columns={'ASSET_PAR_y': 'last_par', 'D_DATE': 'last_date', 'SEC_NAME': 'last_sec'}).drop(
            columns=['ASSET_PAR_x'])
        return res

    def credit_warning(self):
        '''信用风险预警'''
        q = sqls_config['rc_alert_cr']['Sql']%self.basedate
        self.cr_alert = convert_columns(self.db_risk.read_sql(q))
        self.cr_alert.loc[self.cr_alert['TYPE'] == '一类', 'TYPE_L1'] = '多策略户'
        res_check = self.check_bond_added()
        self.cr_alert = pd.merge(self.cr_alert, res_check, on=['C_FUNDNAME', 'CODE'], how='outer')
        self.cr_alert['par_chg'] = self.cr_alert['ASSET_PAR'] - self.cr_alert['last_par']
        self.cr_alert['F_ASSETRATIO'] = self.cr_alert['F_ASSETRATIO']/100
        self.cr_alert = self.cr_alert.sort_values(by=['TYPE', 'YIELD_CNBD', 'new_or_not'], ascending=False)
        self.cr_alert = self.cr_alert.reindex(columns=['C_FUNDNAME', 'FUNDTYPE', 'TYPE_L1', 'MANAGER', 'SEC_NAME', 'DISPOSAL', 'ASSET_PAR', 'F_ASSETRATIO', 'innerRating_issuer', 'RATE_LATESTMIR_CNBD', 'YIELD_CNBD', 'last_par', 'par_chg', 'new_or_not', 'remove_or_not', 'last_sec'])

    def getAttribution(self):
        '''大类资产贡献，公募取今年以来、专户取当前考核期'''
        basedate = self.holdings['D_DATE'].max()

        # 处理专户考核期
        dateDF1 = self._bchPeriod[['产品名称', '考核期开始日', '考核期结束日']].dropna(subset=['考核期开始日'])
        dateDF1['考核期结束日'] = pd.to_datetime(dateDF1['考核期结束日'])
        dateDF1['截止日期'] = basedate
        dateDF1['截止日期'] = dateDF1[['截止日期', '考核期结束日']].min(axis=1)
        # dateDF1['产品类型'] = '专户'
        dateDF1['考核期开始日'] = [x.strftime('%Y-%m-%d') for x in dateDF1['考核期开始日']]
        dateDF1['截止日期'] = [x.strftime('%Y-%m-%d') for x in dateDF1['截止日期']]
        dateDF1 = dateDF1.drop(columns=['考核期结束日'])
        prods1 = dateDF1['产品名称'].drop_duplicates().tolist()

        # 处理公募区间
        dateDF2 = self.holdings[['C_FULLNAME']].drop_duplicates().rename(columns={'C_FULLNAME': '产品名称'})
        dateDF2['startDate'] = basedate.strftime('%Y-%m-%d')[:4] + '-01-01'     # 从年初开始
        dateDF2['endDate'] = basedate.strftime('%Y-%m-%d')
        prods2 = dateDF2['产品名称'].drop_duplicates().tolist()

        data1 = convert_columns(AssetAllocation_all(self.db_risk, prods1, dateDF1, '3'))                # 3为专户，1为公募
        data2 = convert_columns(AssetAllocation_all(self.db_risk, prods2, dateDF2, '1'))
        data = data1.append(data2, sort=False)
        self._attri_data = data.copy()

        cols = ['R_FUND', 'ATTRI_STOCK', 'ATTRI_BOND', 'ATTRI_CONVERTBOND', 'ATTRI_ABS', 'ATTRI_FUNDINVEST', 'ATTRI_DERIVATIVE', 'ATTRI_CASH', 'ATTRI_REPOBUY', 'ATTRI_OTHER']
        res = data.groupby(['C_FULLNAME'])[cols].apply(AttributionCumulation).reset_index()
        res['D_DATE'] = basedate

        name_map = self.val[['C_FULLNAME', 'C_FUNDNAME']].drop_duplicates()
        res = pd.merge(res, name_map, on='C_FULLNAME', how='left')

        return res

    def getFundInvestment(self):
        q = sqls_config['fund_investment']['Sql']
        data = self._retrieve_data_tn(q).fillna({'fundinvest': 0})
        data.columns = [i.upper() for i in data.columns]
        return data

    def getReturnRank(self):
        df_setup = self.db_risk.read_sql(sqls_config["mg_longterm_setup"]["Sql"].format(t=self.basedate))

        # 特殊调整1：混合产品主基金经理
        adjust_ptf = {"010923.OF": "卢丽阳"}
        df_setup["mgr_aux"] = df_setup["portfolio_code"].map(adjust_ptf).fillna(df_setup["manager"])
        df_mgr = self.db_risk.read_sql(sqls_config["mg_longterm_mgr"]["Sql"].format(t=self.basedate))
        res = df_setup.merge(df_mgr, how="left", on=["portfolio_code", "mgr_aux"])

        # 特殊调整2：策略变更的情况
        df_infochg = self.db_risk.read_sql(sqls_config["mg_longterm_infochg"]["Sql"].format(t=self.basedate))
        dict_infochg_ = df_infochg.set_index("portfolio_code")["bg_date"].to_dict()
        res["mgr_date"] = pd.to_datetime(res["portfolio_code"].map(dict_infochg_).fillna(res["bg_date"]))

        # 指定产品所在的名单
        df_rankinglist = self.db_risk.read_sql(sqls_config["mg_longterm_rankinglist"]["Sql"].format(t=self.basedate))
        res = res.merge(df_rankinglist, how="left", on="portfolio_code")

        # 长期业绩
        res["setup_date_aux"] = res["setup_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        res["mgr_date_aux"] = res["mgr_date"].apply(lambda x: x.strftime("%Y-%m-%d") if x is not pd.NaT else None)

        col_ret = ["return", "return_ann", "return_rank", "return_rankpct", "ret_mid", "ret_ann_mid"]
        df_pf = self.db_risk.read_sql(sqls_config["mg_longterm_performance"]["Sql"].format(t=self.basedate))

        # 特殊区间的业绩表现
        bgdate_dict = {"成立以来": "setup_date_aux", "接手以来": "mgr_date_aux"}
        for period in ["成立以来", "接手以来"]:
            col_dict = dict(zip(col_ret, [i + "_" + period for i in col_ret]))
            period_df = df_pf.drop_duplicates(subset=["period_bgdate", "portfolio_code", "ranking_list"]).copy()
            res = res.merge(period_df, how="left", left_on=["portfolio_code", "ranking_list", bgdate_dict[period]],
                            right_on=["portfolio_code", "ranking_list", "period_bgdate"])
            res = res.drop(["return_period", "period_bgdate"], axis=1).rename(columns=col_dict)

        # 固定区间的业绩表现
        for period in ["今年以来", "近一年", "近二年", "近三年", "近五年", "近三月"]:
            col_dict = dict(zip(col_ret, [i + "_" + period for i in col_ret]))
            period_df = df_pf[df_pf["return_period"] == period].copy()
            res = res.merge(period_df, how="left", on=["portfolio_code", "ranking_list"])

            drop_cols = ["return_ann", "ret_ann_mid"] if period in ["今年以来", "近三月"] else ["return", "ret_mid"]
            res = res.drop(["return_period", "period_bgdate"] + drop_cols, axis=1).rename(columns=col_dict)

        # 规模变动
        last_m = self.previous_tradeday(self.basedate[:7] + "-01")
        offset_n = 2 if self.is_end_of_quarter(self.basedate) else 1
        last_q = self.latest_tradeday(get_past_qtr_date(self.basedate, offset_n))
        last_y = self.previous_tradeday(self.basedate[:4] + "-01-01")
        past_size = self.db_risk.read_sql(sqls_config["mg_past_size"]["Sql"].format(last_m=last_m, last_q=last_q, last_y=last_y))
        res = res.merge(past_size, how="left", on="portfolio_code")

        # 过去一年的回撤情况
        sql_dd = sqls_config["mg_pastyear_dd"]["Sql"]
        dd_1y = self.db_risk.read_sql(sql_dd.format(t=self.basedate, period="近一年"))
        sql_dd_mid = sqls_config["mg_pastyear_dd_median"]["Sql"]
        dd_1y_mid = self.db_risk.read_sql(sql_dd_mid.format(t=self.basedate, period="近一年"))

        dd_res = pd.merge(dd_1y, dd_1y_mid, how='left', on=["ranking_list"])
        res = res.merge(dd_res, how="left", on=["portfolio_code", "ranking_list"])

        # 过去三个月的波动
        vol_3m = self.db_risk.read_sql(sqls_config["mg_volatiliy"]["Sql"].format(t=self.basedate))
        res = res.merge(vol_3m, how="left", on=["portfolio_code", "ranking_list"])

        # 建仓结束以来
        ret_cols = ["return", "ret_mid", "return_rankpct", "return_rank"]
        col_dict = dict(zip(ret_cols, [i + "_建仓结束" for i in ret_cols]))
        pos_end = df_pf[df_pf["return_period"] == "建仓结束以来"].reindex(columns=["portfolio_code", "ranking_list"]+ret_cols).rename(columns=col_dict)
        res = res.merge(pos_end, how="left", on=["portfolio_code", "ranking_list"])

        # 考核开始以来
        col_dict = dict(zip(ret_cols, [i + "_考核开始以来" for i in ret_cols]))
        pos_end = df_pf[df_pf["return_period"] == "考核开始以来"].reindex(
            columns=["portfolio_code", "ranking_list"] + ret_cols).rename(columns=col_dict)
        res = res.merge(pos_end, how="left", on=["portfolio_code", "ranking_list"])

        res = res.drop(["mgr_aux", "bg_date", "setup_date_aux", "mgr_date_aux"], axis=1)
        # res.to_excel(os.path.join(DIR_OF_MAIN_PROG, 'data', '业绩状态.xlsx'))
        return res

    def getDailySize(self):
        '''规模变动日报'''
        sql_size = sqls_config["size_daily"]["Sql"]
        last_m = self.previous_tradeday(self.basedate[:7] + "-01")
        offset_n = 2 if self.is_end_of_quarter(self.basedate) else 1
        last_q = self.latest_tradeday(get_past_qtr_date(self.basedate, offset_n))
        last_y = self.previous_tradeday(self.basedate[:4] + "-01-01")
        res = self.db_risk.read_sql(sql_size.format(last_m=last_m, last_q=last_q, last_y=last_y, t=self.basedate))

        # 非T日估值产品
        for n, ptf_names in self.tn_ptf_d.items():
            td = self.get_offset_tradeday(self.basedate, -1 * n)
            q = sql_size.format(last_m=min(last_m, td), last_q=min(last_q, td), last_y=min(last_y, td), t=td)
            data = self.db_risk.read_sql(q)
            res = pd.concat([res[~res['c_fundname'].isin(ptf_names)], data[data['c_fundname'].isin(ptf_names)].copy()])

        return res

    def getHolderType(self):
        '''根据IDC_HOLDER进行取数，数据会晚一天'''
        data = self.db_risk.read_sql(sqls_config["fund_holders_mg"]["Sql"].format(t=self.basedate))

        miss_data = data[data["holder_type_l2"].isnull()].copy()
        save_data = miss_data.reindex(columns=["c_date", "holder_type", "shares_type"]).drop_duplicates().reset_index(drop=True)
        self.holders_check(save_data)

        col_d = {'机构': 'institution', '产品': 'product', '个人': 'individual'}
        fund_info = data[["c_date", "portfolio_code", "c_fundname"]].drop_duplicates()
        df1 = pd.pivot_table(data, values='hold_ratio', index='portfolio_code', columns='holder_type_l1', aggfunc="sum")
        df1 = df1.reindex(columns=list(col_d.keys())).rename(columns=col_d)
        df2 = pd.pivot_table(data, values='hold_ratio', index='portfolio_code', columns='holder_type_l2', aggfunc="sum")
        res = fund_info.merge(df1, how='left', on='portfolio_code').merge(df2, how='left', on='portfolio_code').fillna(0)

        # 前5大投资者
        top_holder = self.db_risk.read_sql(sqls_config["top_holders_mg"]["Sql"].format(t=self.basedate))
        keep_cols = ["portfolio_code", "investor", "ratio"]
        for i in range(1, 6):
            col_dict = dict(zip(keep_cols[1:], ["top" + str(i) + "_" + col for col in keep_cols[1:]]))
            temp = top_holder[top_holder["rank"] == i].reindex(columns=keep_cols).rename(columns=col_dict).copy()
            res = res.merge(temp, how="left", on="portfolio_code")
        data_bank = self.db_risk.read_sql(sqls_config["top_holders_bank_mg"]["Sql"].format(t=self.basedate))
        res = res.merge(data_bank.query("holder_type_l2 =='银行理财产品'").groupby(['portfolio_code'])['hold_ratio'].max().reset_index().rename(columns = {'hold_ratio':'银行理财产品_最大持有人'}),how ='left',on = 'portfolio_code')
        return res

    def holders_check(self, data):
        receiver = 'shiy02@maxwealthfund.com'
        c_receiver = 'wangp@maxwealthfund.com'
        title = '%s持有人类型检查' % self.basedate
        if data.empty:
            save_path = ''
            title += "-无新增"
            info = '持有人类型检查结果如下—无新增持有人类型'
        else:
            save_path = os.path.join(DIR_OF_MAIN_PROG, 'data', '持有人类型检查', '类型检查_{t}.xlsx'.format(t=self.basedate.replace("-", "")))
            data.to_excel(save_path)
            info = '持有人类型检查结果如下'

        body = format_body([data], [info])
        send_email(receiver, c_receiver, title, body, att=save_path)

    def getCompanyRank(self):
        sql_raw = sqls_config["fund_company_mg"]["Sql"]
        data = self.db_risk.read_sql(sql_raw.format(t=self.basedate))
        return data

    def calcAllIdx(self):
        self.res_dura = self.getDuraChg()
        self.res_innR = self.getInnerRatingDist()
        self.res_mmf = self.getMMFIdx()
        self.res_over = self.getOverdueSecs()
        self.res_lowliq = self.getLowLiqAsset()
        self.res_retCum = self.getReturnRate_p()
        self.IndexFund = self.getIndexFundsDev(self.basedate)
        self.asset_allc = self.getAttribution()
        self.res_fundinvest = self.getFundInvestment()
        self.return_rank = self.getReturnRank()
        self.holder_type = self.getHolderType()
        self.cpy_rank = self.getCompanyRank()
        self.size_daily = self.getDailySize()
        
        self.getSoldOutSecs()
        self.credit_warning()
    
    def saveAllIdx(self, save_path):
        writer = pd.ExcelWriter(os.path.join(save_path, '%s_RiskIndicators_mg.xlsx'%self.basedate.replace('-', '')))
        self.res_dura.to_excel(writer, sheet_name='久期变化', index=False)
        self.res_mmf.to_excel(writer, sheet_name='货基指标', index=False)
        self.res_innR.to_excel(writer, sheet_name='内部评级分布', index=False)
        self.IndexFund.to_excel(writer, sheet_name='指数跟踪正偏离', index=False)
        self.res_over.to_excel(writer, sheet_name='专户_错配', index=False)
        self.res_lowliq.to_excel(writer, sheet_name='专户_低流动性', index=False)
        self.res_retCum.to_excel(writer, sheet_name='专户_收益率缺口', index=False)
        self.asset_allc.to_excel(writer, sheet_name='大类资产归因', index=False)
        self.res_fundinvest.to_excel(writer, sheet_name='基金投资仓位', index=False)
        self._innerRating.to_excel(writer, sheet_name='内部主体评级', index=False)
        self.soldOutsecs.to_excel(writer, sheet_name='择机卖出台账', index=False)
        self.cr_alert.to_excel(writer, sheet_name='信用风险预警', index=False)
        self.return_rank.to_excel(writer, sheet_name='业绩状态', index=False)
        self.holder_type.to_excel(writer, sheet_name='持有人信息', index=False)
        self.cpy_rank.to_excel(writer, sheet_name='company', index=False)
        self.size_daily.to_excel(writer, sheet_name='规模日报底稿', index=False)
        writer.save()
        
        print('MG Indicators done.')

    def insert2db(self):
        dict_sheet = self.load_sheet_columns('RiskIndicators_mg')
        sheet_names = ['久期变化', '货基指标', '内部评级分布', '指数跟踪正偏离', '专户_错配', '专户_低流动性', '专户_收益率缺口']
        table_mg = ['RC_MGMT_DURACHG', 'RC_MGMT_MONETARYFUND', 'RC_MGMT_INNERRATING_DIST', 'RC_MGMT_INDEXFUND',
                    'RC_MGMT_OVERMTRT', 'RC_MGMT_LOWLIQSECS', 'RC_MGMT_RETURNGAP']
        data_mg = [self.res_dura, self.res_mmf, self.res_innR, self.IndexFund, self.res_over, self.res_lowliq, self.res_retCum]
        pairs_mg = [(x, y, z) for x, y, z in zip(sheet_names, table_mg, data_mg)]

        for sheet, table, data in zip(sheet_names, table_mg, data_mg):
            print('==' * 3, table, '==' * 3)
            if 'D_DATE' not in data.columns:
                data['D_DATE'] = pd.to_datetime(self.basedate)
            mmf_keepcols = ['C_FUNDNAME', '影子偏离度', '七日年化', '平均到期日', '5日金融工具占比', 'D_DATE', '流通受限占比', '主评低于AAA占比']
            data = data.reindex(columns=mmf_keepcols) if sheet == '货基指标' else data
            data.columns = dict_sheet[sheet]['Sheet_columns']
            data = data.dropna(subset=['D_DATE'], how='any')
            data = data.replace(np.inf, np.nan).replace((-1) * np.inf, np.nan)
            self.insert2db_single(table, data, t=self.basedate)
    #
    # def load_sheet_columns(self):
    #     '''
    #     加载管理层风险指标的数据字典表，以方便插入数据库
    #     :return:
    #     '''
    #     dict_cols = pd.read_excel(os.path.join(DIR_OF_MAIN_PROG, 'data', 'SheetColumns_RiskIndicators_mg.xlsx'),
    #                               sheet_name=None, engine='openpyxl')
    #     return dict_cols



class RiskIndicator_cmt(RiskIndicator_mg):
    def __init__(self, t, save_path, ptf_codes=None):
        RiskIndicator_mg.__init__(self, t, save_path, ptf_codes)

        # 现金管理团队负责的全量产品
        ptf_info = self.db_risk.read_sql(sqls_config['cmt_ptf_info']['sql'])
        self.ptf_info = ptf_info.copy()
        self.ptf_codes = ptf_info['portfolio_code'].to_list()
        self.general_dates = self.general_dates()

    def general_dates(self):
        last_m = self.previous_tradeday(self.basedate[:7] + "-01")
        offset_n = 2 if self.is_end_of_quarter(self.basedate) else 1
        last_q = self.latest_tradeday(get_past_qtr_date(self.basedate, offset_n))
        last_y = self.previous_tradeday(self.basedate[:4] + "-01-01")

        return {'较月初': last_m, '较季初': last_q, '较年初': last_y}

    def get_size(self):
        '''产品规模相关信息'''
        q_summary = sqls_config['cmt_size_summary']['sql']
        df_summary = self.db_risk.read_sql(q_summary.format(t=self.basedate))
        # 明细数据是t+2
        prev_td = self.get_offset_tradeday(self.basedate, -1)
        q_detail = sqls_config['cmt_size_detail']['sql']
        df_detail = self.db_risk.read_sql(q_detail.format(t=prev_td))

        for chg_name, t0 in self.general_dates.items():
            df1 = self.db_risk.read_sql(q_summary.format(t=t0)).rename(columns={'asset': f'asset_{chg_name}'})
            df_summary = df_summary.merge(df1, how='left', on=['portfolio_code', 'data_type'])
            df_summary[chg_name] = df_summary['asset'] - df_summary[f'asset_{chg_name}']
            df_summary.drop(columns=f'asset_{chg_name}', inplace=True)

            t0_adj = min(t0, prev_td)
            df2 = self.db_risk.read_sql(q_detail.format(t=t0_adj)).rename(columns={'asset': f'asset_{chg_name}'})
            df_detail = df_detail.merge(df2, how='left', on=['portfolio_code', 'org_name'])
            df_detail[chg_name] = df_detail['asset'] - df_detail[f'asset_{chg_name}']
            df_detail.drop(columns=f'asset_{chg_name}', inplace=True)

        res1 = pd.merge(self.ptf_info, df_summary, how='left', on='portfolio_code').drop(columns='ranking_list')
        res2 = pd.merge(self.ptf_info, df_detail, how='left', on='portfolio_code').drop(columns='ranking_list')
        logger.info('CMT: Size Data Done.')
        return res1, res2

    def get_risk_indicators(self):
        # 1) 常规风险指标：久期、杠杆等
        q = sqls_config['cmt_risk_indicators']['sql']
        df1 = self.db_risk.read_sql(q.format(t=self.basedate))

        # 2) 内评分布
        df2 = self.getInnerRatingDist().drop(columns='D_DATE').rename(columns={'C_FUNDNAME': 'c_fundname'})
        df2['B+及以下'] = df2[['交易级', '投机级']].sum(axis=1)
        res = self.ptf_info.merge(df1, how='left', on='c_fundname').merge(df2, how='left', on='c_fundname')

        # 3) 产品对标名单的中位规模
        q_mkt = sqls_config['cmt_size_mkt']['sql']
        df3 = self.db_risk.read_sql(q_mkt.format(t=self.basedate))
        res = pd.merge(res, df3, how='left', on='ranking_list')

        # 4) 基准指数回撤
        q_bm = sqls_config['cmt_bm_index']['sql']
        df4 = self.db_risk.read_sql(q_bm.format(t=self.basedate))
        res = pd.merge(res, df4, how='left', on='portfolio_code')

        # 5) 年化跟踪误差
        q_track_dev = sqls_config['cmt_track_dev']['sql']
        df5 = self.db_risk.read_sql(q_track_dev.format(t=self.basedate))
        res = pd.merge(res, df5, how='left', on='portfolio_code')

        logger.info('CMT: Risk Indicators Done.')
        return res

    def calculate_and_save(self, save_path):
        writer = pd.ExcelWriter(os.path.join(save_path, '%s_RiskIndicators_cmt.xlsx' % self.basedate.replace('-', '')))
        size_summary, size_detail = self.get_size()
        risk_ind = self.get_risk_indicators()
        ret_rank = self.getReturnRank()
        mmf_ind = self.getMMFIdx()
        size_summary.to_excel(writer, sheet_name='规模汇总', index=False)
        size_detail.to_excel(writer, sheet_name='规模明细', index=False)
        risk_ind.to_excel(writer, sheet_name='指标明细', index=False)
        ret_rank.to_excel(writer, sheet_name='业绩状态', index=False)
        mmf_ind.to_excel(writer, sheet_name='货基指标', index=False)
        writer.save()
        logger.info('Cash Management Team Indicators Done.')


if __name__ == '__main__':
    data_path = r'E:\RiskQuant\风险指标\Valuation\\'
    save_path = r'E:\RiskQuant\风险指标\DailyIndicators\\'
    t = '2021-04-22'

    file_val = 'valuation' + t.replace('-', '') + '.json'
    file_nav = '单位净值-基金A&专户.xlsx'
    data_path_out = save_path + '%s\\'%t.replace('-', '')

    mg = RiskIndicator_mg(data_path, file_nav, file_val, t, data_path_out)
    mg.calcAllIdx(save_path)
    mg.saveAllIdx(save_path)