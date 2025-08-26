
import os
import pandas as pd
import numpy as np
from WindPy import w
from datetime import datetime

from scripts.utils.log_utils import logger
from scripts.settings import config, DIR_OF_MAIN_PROG
from scripts.db import OracleDB, sqls_config
from scripts.db.db_utils import convert_columns
from scripts import demo_code

from scripts.risk_warning import PortfolioWarning


def saveData(data_path, Indicators):
    '''
    保存文件\n
    :param data_path: string, 保存路径
    :param Indicators: class, python类，当日指标（如市场风险指标等）
    :return: None
    '''
    writer = pd.ExcelWriter(os.path.join(data_path, 'Holdings.xlsx'))
    Indicators.bond_holdings.to_excel(writer, sheet_name='bond_holdings', index=False)
    Indicators.stock_holdings.to_excel(writer, sheet_name='stock_holdings', index=False)
    Indicators.holdings.to_excel(writer, sheet_name='holdings', index=False)
    writer.save()

    writer = pd.ExcelWriter(os.path.join(data_path, 'data_db.xlsx'))
    Indicators.data_jy.to_excel(writer, sheet_name='data_jy', index=False)
    Indicators.data_wind.to_excel(writer, sheet_name='data_wind', index=False)
    writer.save()

    Indicators.data_wind_equity.to_excel(os.path.join(data_path, 'data_wind_equity.xlsx'), index=False)

    writer = pd.ExcelWriter(os.path.join(data_path, 'data_benchmark.xlsx'))
    Indicators._benchmark.to_excel(writer, sheet_name='benchmark', index=False)
    Indicators._benchmark_gk.to_excel(writer, sheet_name='benchmark_gk', index=False)
    Indicators._benchmark_ind.to_excel(writer, sheet_name='benchmark_ind', index=False)
    Indicators._benchmark_rating.to_excel(writer, sheet_name='benchmark_rating', index=False)
    writer.save()

    Indicators._yield_map.to_excel(os.path.join(data_path, 'yield_map.xlsx'), index=False)
    Indicators.data_fund.to_excel(os.path.join(data_path, 'data_fund.xlsx'), index=False)

    Indicators._lev_all.to_excel(os.path.join(data_path, '_lev_all.xlsx'), index=False)
    Indicators._repo_all.to_excel(os.path.join(data_path ,'_repo_all.xlsx'), index=False)


def retrieve_time_series(save_path, startdate, enddate, file_name='RiskIndicators-v3.xlsx'):
    '''
    生成风险指标的时间序列数据，从数据库取数并保存至本地\n
    :param save_path: string, 保存的文件夹路径
    :param startdate: string, yyyy-mm-dd格式, 起始日期(含)
    :param enddate: string, yyyy-mm-dd格式, 结束日期(含)
    :param file_name: string, 保存的文件名及格式
    :return: None
    '''
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    q = sqls_config['rc_series']['Sql']

    writer = pd.ExcelWriter(os.path.join(save_path, file_name), engine='openpyxl')
    sheets = ['Concentration', 'Basic', 'Market_return', 'Market_return_bch', 'Market_Holding', 'Credit', 'Liquidity', 'Liquidity_core']
    tables = ['RC_CONCENTRATION', 'RC_STYLE', 'RC_MR_RETURN', 'RC_MR_RETURN_BCH', 'RC_MR_HOLDING', 'RC_CREDIT', 'RC_LIQUIDITY', 'RC_LR_CORE']
    dict_cols = pd.read_excel(os.path.join(save_path, '20220805_RiskIndicators.xlsx'), sheet_name=None, engine='openpyxl')
    for (sheet, table) in zip(sheets, tables):
        idx_temp = convert_columns(db_risk.read_sql(q%(table, startdate, enddate)).rename(columns={'PAIN_INDEX': 'pain_index', 'HIT_RATE': 'hit_rate'}))
        idx_temp['D_DATE'] = pd.to_datetime(idx_temp['D_DATE'])
        idx_temp = idx_temp.sort_values(by=['D_DATE', 'C_FUNDNAME'])
        if 'insert_time'.upper() in idx_temp:
            idx_temp = idx_temp.drop(columns=['insert_time'.upper()])
        idx_temp.columns = dict_cols[sheet].columns
        idx_temp.to_excel(writer, sheet_name=sheet, index=False)
    writer.save()


def retrieve_idc_repo(t):
    '''
    回购明细表(idc_repo)\n
    :param t: string, yyyy-mm-dd格式, 取数日期
    :return: None
    '''
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    q = sqls_config['idc_repo_t']['Sql']%t
    data_repo = db_risk.read_sql(q)
    data_repo = data_repo.loc[(data_repo['spot'] == '银行间') & (data_repo['direction'] == '融资回购') & (data_repo['actual_maturity_date'] > t), :].copy()
    path_collateral = config['shared_drive_data']['collateral_file']['path']
    data_repo.to_excel(os.path.join(path_collateral, t.replace('-', ''), 'idc_repo.xlsx'), index=False)


def retrieveBchIdxLastDay(t):
    '''
    比较基准的风险收益指标数据\n
    :param t: string, yyyy-mm-dd格式, 取数日期
    :return: DataFrame
    '''
    db_risk = OracleDB(config['data_base']['QUANT']['url'])
    q = sqls_config['rc_mr_return_bch_t']['Sql']
    t1 = w.tdaysoffset(-1, t, "").Data[0][0].strftime('%Y-%m-%d')
    idx_t1 = convert_columns(db_risk.read_sql(q%t1)).rename(columns={'PAIN_INDEX': 'pain_index', 'HIT_RATE': 'hit_rate'}).drop(columns=['INSERT_TIME'])
    idx_t1['D_DATE'] = pd.to_datetime(t)
    return idx_t1


def check_path(folder_path):
    '''
    检查文件夹是否存在, 若不存在则创建该文件夹\n
    :param folder_path: string, 文件夹路径
    :return: None
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

from dateutil.relativedelta import relativedelta
for num in range(1):
    t = demo_code.retrieve_n_tradeday(datetime.today(), num)
    save_path_rc = config['shared_drive_data']['risk_indicator_daily']['path']
    save_path = os.path.join(DIR_OF_MAIN_PROG, 'data') + '\\'
    data_path_out = save_path + '%s\\' % t.replace('-', '')
    demo_code.check_path(data_path_out)
    logger.info('开始指标计算...')


    rw = PortfolioWarning(t, data_path_out)
    self = rw
    self.connect_jydb()
    datestr = pd.to_datetime(self.basedate).strftime("%Y%m%d")

    

    sql_bd = sqls_config['beg_date']['Sql'].format(t=self.basedate)
    beg_date = self.db_risk.read_sql(sql_bd).values[0][0]

    sql_dd = sqls_config['mf_ret_dd']['Sql'].format(t=self.basedate)
    pfl_dd = self.db_risk.read_sql(sql_dd)
    pfl_dd.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in pfl_dd.columns]
    pfl_dd = pfl_dd.drop_duplicates(subset='PORTFOLIO_CODE')

    sql_th = sqls_config['rm_threshold']['sql'].format(t=self.basedate)
    rm_threshold = self.db_risk.read_sql(sql_th)
    rm_threshold.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in rm_threshold.columns]
    rm_threshold = rm_threshold[['PORTFOLIO_CODE', 'C_FUNDNAME','DD_THOLD', 'DD_ALERT_L1','DD_ALERT_L2', 'DD_KEY_WARN','DD_NO_MOREPOS']] 

    sql_mid_dd = sqls_config['mf_ranking_list_dd_mid_xy']['Sql'].format(t=self.basedate)
    mf_ranking_list_dd_mid = self.db_risk.read_sql(sql_mid_dd)
    mf_ranking_list_dd_mid.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in mf_ranking_list_dd_mid.columns]
    mf_ranking_list_dd_mid = mf_ranking_list_dd_mid.rename(columns={'C_FUNDCODE':'PORTFOLIO_CODE'})

    sql_cb_ret = sqls_config['mf_cb_ret']['Sql'].format(t=self.basedate)
    mf_cb_ret = self.db_risk.read_sql(sql_cb_ret)
    mf_cb_ret.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in mf_cb_ret.columns]

    sql_cb_ret_past1m = sqls_config['mf_cb_ret_past1m']['Sql'].format(t=self.basedate,last_t = (pd.to_datetime(self.basedate)-relativedelta(months=1)).strftime('%Y-%m-%d'))
    mf_cb_ret_past1m = self.db_risk.read_sql(sql_cb_ret_past1m)
    mf_cb_ret_past1m.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in mf_cb_ret_past1m.columns]

    sql_ths = sqls_config['rm_threshold_cbond']['Sql']
    rm_threshold_cbond = self.db_risk.read_sql(sql_ths)
    rm_threshold_cbond.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in rm_threshold_cbond.columns]

    sql_cpd = sqls_config['cb_pos_detail']['Sql'].format(t=self.basedate)
    cb_pos_detail = self.db_risk.read_sql(sql_cpd)
    cb_pos_detail.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in cb_pos_detail.columns]
    cb_pos_detail = cb_pos_detail.rename(columns = {'MV_NAV_RATIO':'F_ASSETRATIO'})

    sql_eq_style = sqls_config['cb_eq_style']['Sql'].format(t=self.basedate)
    cb_eq_style = self.db_risk.read_sql(sql_eq_style)
    cb_eq_style.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in cb_eq_style.columns]

    sql_cb_ld = sqls_config['cb_begin_dt']['Sql'].format(t=self.basedate,last_t = (pd.to_datetime(self.basedate)-relativedelta(months=1)).strftime('%Y-%m-%d'))
    cb_begin_dt = self.db_risk.read_sql(sql_cb_ld)
    cb_begin_dt.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in cb_begin_dt.columns]

    sql_cb_idx = sqls_config['cb_idx']['Sql'].format(t=beg_date)
    cb_idx = self.db_jy.read_sql(sql_cb_idx)
    cb_idx.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in cb_idx.columns]

    sql_cb_ret_past1m = sqls_config['mf_cb_ret_past1m']['Sql'].format(t=self.basedate,last_t = beg_date)
    mf_cb_ret_past1y = self.db_risk.read_sql(sql_cb_ret_past1m)
    mf_cb_ret_past1y.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in mf_cb_ret_past1y.columns]

    sql_mf_cb_dd = sqls_config['mf_cb_dd']['Sql'].format(last_t=beg_date,t = self.basedate)
    mf_cb_dd = self.db_risk.read_sql(sql_mf_cb_dd)
    mf_cb_dd.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in mf_cb_dd.columns]

    mf_cb_dd = mf_cb_dd.query("C_DATE == @self.basedate").merge(mf_cb_dd.query("C_DATE != @self.basedate")[['PORTFOLIO_CODE','C_DATE','NV']].rename(columns={'C_DATE':"DD_BEG_DT",'NV':"NV_PAST"}),how ='left',on = 'PORTFOLIO_CODE')
    mf_cb_dd['DD_BEG_DT'] = mf_cb_dd['DD_BEG_DT'].fillna(mf_cb_dd['C_DATE'])
    mf_cb_dd['NV_PAST'] = mf_cb_dd['NV_PAST'].fillna(mf_cb_dd['NV'])
    mf_cb_dd['DD'] = mf_cb_dd['NV']/mf_cb_dd['NV_PAST']-1

    sql_mf_rank_daily = sqls_config['mf_rank_daily']['Sql'].format(last_t=beg_date,t = self.basedate)
    mf_rank_daily = self.db_risk.read_sql(sql_mf_rank_daily)
    mf_rank_daily.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in mf_rank_daily.columns]
    mf_rank_daily_cum = mf_rank_daily.pivot(index ='C_DATE',columns='RANKING_LIST',values='DAILY_RETURN').fillna(0)+1
    mf_rank_daily_cum = mf_rank_daily_cum.cumprod(axis =0)

    sql_bi = sqls_config['mf_basic_info_master']['Sql'].format(t=self.basedate)
    mf_basic_info = self.db_risk.read_sql(sql_bi)
    mf_basic_info.columns = [config['column_map'][x.upper()] if x.upper() in config['column_map'].keys()
                    else x.upper() for x in mf_basic_info.columns]

    data = pfl_dd.merge(rm_threshold.drop('C_FUNDNAME',axis= 1),on ='PORTFOLIO_CODE' ,how= 'left')
    data = data.merge(mf_ranking_list_dd_mid[['PORTFOLIO_CODE','RANKING_LIST','CRTDD_MID']],on ='PORTFOLIO_CODE',how ='left')
    data['EX_DD'] = data['CRT_DD'].sub(data['CRTDD_MID'],fill_value = 0)
    data['EX_DD'] = data['EX_DD'].mask(data['EX_DD']>0,0)
    data = data.drop(['DD_ALERT_L1','DD_ALERT_L2','DD_KEY_WARN','DD_NO_MOREPOS'],axis =1)

    data_single = rm_threshold_cbond.merge(cb_pos_detail[['PORTFOLIO_CODE','code','SEC_ABBR','F_ASSETRATIO']],on = 'PORTFOLIO_CODE',how ='left')
    data_single['CB_SING_UPPER']*=100
    data_single = pd.merge(data_single.rename(columns={'code':'CBOND_WINDCODE'}),mf_cb_ret.rename(columns={'C_FUNDCODE':'PORTFOLIO_CODE'})[['PORTFOLIO_CODE','CBOND_WINDCODE','TWRR','D_RET','WEIGHT_PROD','W_D_RET_ALL_PROD']],how ='left',on =['PORTFOLIO_CODE','CBOND_WINDCODE'])
    data_single = data_single.merge(cb_eq_style[['CBOND_WINDCODE','EQUITY_STYLE']],how = 'left',on ='CBOND_WINDCODE' )
    data_single['daily_dd_th'] = data_single.apply(lambda x:-0.03 if x['EQUITY_STYLE'] == '债性' else (-0.05 if x['EQUITY_STYLE'] == '平衡' else -0.07),axis =1)
    data_single['monthly_exdd_th'] = data_single.apply(lambda x:-0.06 if x['EQUITY_STYLE'] == '债性' else (-0.1 if x['EQUITY_STYLE'] == '平衡' else -0.15),axis =1)
    data_single['contri_th'] = data_single.apply(lambda x:0.4 if x['EQUITY_STYLE'] == '债性' else (0.5 if x['EQUITY_STYLE'] == '平衡' else 0.6),axis =1)
    # data_single['contri_th'] = data_single.apply(lambda x:0.1 if x['EQUITY_STYLE'] == '债性' else (0.1 if x['EQUITY_STYLE'] == '平衡' else 0.1),axis =1)

    idx_unitnv = cb_idx.pivot(index = 'TRADINGDAY',columns='SECUCODE',values='CHANGEPCT')/100+1
    idx_unitnv = idx_unitnv.cumprod(axis=0)
    idx_ret_until_basedt = (idx_unitnv.loc[self.basedate] / idx_unitnv.loc[:self.basedate])-1
    # idx_ret_until_basedt = idx_ret_until_basedt.shift(1) 
    idx_ret_until_basedt.columns = ['idx_ret_until_basedt']

    mf_cb_cumret = mf_cb_ret_past1m.pivot(index = 'D_DATE',columns=['C_FUNDCODE','CBOND_WINDCODE'],values='TWRR')+1
    mf_cb_cumret = mf_cb_cumret.cumprod(axis= 0 )
    mf_cb_cumret = mf_cb_cumret.loc[self.basedate].div(mf_cb_cumret,axis =0)
    mf_cb_cumret = mf_cb_cumret.unstack().reset_index()
    mf_cb_cumret.columns =['C_FUNDCODE','CBOND_WINDCODE','BEGIN_DATE','TWRR_CUM']
    mf_cb_cumret = mf_cb_cumret.dropna()
    mf_cb_cumret['TWRR_CUM'] = mf_cb_cumret['TWRR_CUM']-1
    mf_cb_cumret = pd.merge(cb_begin_dt,mf_cb_cumret,on=['C_FUNDCODE', 'CBOND_WINDCODE', 'BEGIN_DATE'],how ='left')
    mf_cb_cumret['BEGIN_DATE'] = pd.to_datetime(mf_cb_cumret['BEGIN_DATE'])

    data_single = data_single.merge(cb_begin_dt.rename(columns={'C_FUNDCODE':'PORTFOLIO_CODE'}),how ='left',on =['PORTFOLIO_CODE','CBOND_WINDCODE'] )
    data_single['BEGIN_DATE'] = pd.to_datetime(data_single['BEGIN_DATE'])
    data_single = data_single.merge(idx_ret_until_basedt,left_on='BEGIN_DATE',right_index=True)
    data_single = data_single.merge(mf_cb_cumret.rename(columns={'C_FUNDCODE':'PORTFOLIO_CODE'}),on = ['PORTFOLIO_CODE','CBOND_WINDCODE','BEGIN_DATE'],how = 'left')
    data_single['EX_CUM_RET'] = data_single['TWRR_CUM']-data_single['idx_ret_until_basedt']

    mf_cb_ret_past1y = mf_cb_ret_past1y.merge(mf_ranking_list_dd_mid.rename(columns = {'PORTFOLIO_CODE':'C_FUNDCODE'})[['C_FUNDCODE','RANKING_LIST']],on = 'C_FUNDCODE',how = 'left')
    mf_cb_ret_past1y = mf_cb_ret_past1y.merge(mf_rank_daily[['C_DATE','RANKING_LIST','DAILY_RETURN']].rename(columns = {'C_DATE':'D_DATE','DAILY_RETURN':"LIST_RET"}),how = 'left',on = ['D_DATE','RANKING_LIST'])
    mf_cb_ret_past1y['contribution_cb'] = (mf_cb_ret_past1y['TWRR'])*mf_cb_ret_past1y['WEIGHT_PROD']
    mf_cb_ret_past1y['contribution_lst'] = (mf_cb_ret_past1y['LIST_RET'])*mf_cb_ret_past1y['WEIGHT_PROD']

    contribution_cb = (mf_cb_ret_past1y.pivot(index ='D_DATE',columns=['C_FUNDCODE','CBOND_WINDCODE'],values='contribution_cb').fillna(0)+1).cumprod(axis =0 ) 
    contribution_cb= contribution_cb.loc[self.basedate].div(contribution_cb,axis= 0)-1
    contribution_lst = (mf_cb_ret_past1y.pivot(index ='D_DATE',columns=['C_FUNDCODE','CBOND_WINDCODE'],values='contribution_lst').fillna(0)+1 ).cumprod(axis =0 )
    contribution_lst= contribution_lst.loc[self.basedate].div(contribution_lst,axis= 0)-1

    rank_list = mf_ranking_list_dd_mid.rename(columns = {'PORTFOLIO_CODE':'C_FUNDCODE'})[['C_FUNDCODE','RANKING_LIST']]
    rank_list['LIST_DD_BEGDT'] = rank_list['RANKING_LIST'].map(mf_rank_daily_cum.idxmax())
    rank_list['PFL_DD_BEGDT'] = rank_list['C_FUNDCODE'].map(mf_cb_dd.set_index("PORTFOLIO_CODE")['DD_BEG_DT'])
    rank_list = rank_list.dropna( )
    rank_list['BENCH_DT'] =rank_list.apply(lambda x:min(x['LIST_DD_BEGDT'],x['PFL_DD_BEGDT']), axis =1)

    contribution_cb = contribution_cb.unstack().reset_index().rename(columns={0:'contribution_cb'}).merge(rank_list.rename(columns={'BENCH_DT':'D_DATE'})[['C_FUNDCODE','D_DATE']],on = ['C_FUNDCODE','D_DATE'],how = 'inner')
    contribution_lst = contribution_lst.unstack().reset_index().rename(columns={0:'contribution_lst'}).merge(rank_list.rename(columns={'BENCH_DT':'D_DATE'})[['C_FUNDCODE','D_DATE']],on = ['C_FUNDCODE','D_DATE'],how = 'inner')
    contribution = pd.merge(contribution_cb,contribution_lst,how='left',on =['C_FUNDCODE','D_DATE','CBOND_WINDCODE'] )
    contribution['contribution'] = contribution['contribution_cb']- contribution['contribution_lst']

    contribution = contribution.merge(data[['EX_DD','PORTFOLIO_CODE']].rename(columns={'PORTFOLIO_CODE':'C_FUNDCODE'}),on = ['C_FUNDCODE'],how ='left')#.dropna(subset = 'EX_DD')
    contribution['contribution'] /= contribution['EX_DD']

    data_single = data_single.merge(contribution[['C_FUNDCODE','CBOND_WINDCODE','D_DATE','contribution']].rename(columns = {'C_FUNDCODE':'PORTFOLIO_CODE','D_DATE':"BENCH_DATE"}),how ='left',on = ['PORTFOLIO_CODE','CBOND_WINDCODE'])
    data_single = pd.merge(data_single,data[['PORTFOLIO_CODE','EX_DD','DD_THOLD']],how='left')
    data_single.columns = [i.upper() for i in data_single.columns]
    data_single['ALART'] =  data_single.apply(lambda x:'个券单日跌幅早期预警' if (((x['F_ASSETRATIO'] >= x['CB_SING_UPPER']) & (x['TWRR'] <x['DAILY_DD_TH'])) ) else  None  ,axis =1)
    data_single['ALART'] =  data_single.apply(lambda x:'个券月度超额跌幅早期预警' if (((x['F_ASSETRATIO'] >= x['CB_SING_UPPER']) & (x['EX_CUM_RET'] <x['MONTHLY_EXDD_TH'])) ) else  x['ALART']  ,axis =1)
    data_single['ALART'] =  data_single.apply(lambda x:'个券超额回撤贡献重点预警' if (((abs(x['EX_DD'])>0.2*x['DD_THOLD']) & (x['CONTRIBUTION']>=x['CONTRI_TH'])) ) else  x['ALART']  ,axis =1)
    data_single['D_DATE'] = pd.to_datetime(self.basedate)
    data_single = data_single[['ALART','D_DATE','PORTFOLIO_CODE','C_FUNDNAME','CBOND_WINDCODE','SEC_ABBR','EQUITY_STYLE','F_ASSETRATIO','CB_SING_UPPER','TWRR','DAILY_DD_TH','TWRR_CUM','IDX_RET_UNTIL_BASEDT','EX_CUM_RET','MONTHLY_EXDD_TH','BENCH_DATE','CONTRIBUTION','CONTRI_TH','EX_DD','DD_THOLD']]
    data_single= data_single.rename(columns ={'TWRR_CUM':'TWRR_MONTHLY','IDX_RET_UNTIL_BASEDT':'IDX_MONTHLY','EX_CUM_RET':'EX_RET_MONTHLY','MONTHLY_EXDD_TH':'EX_RET_MONTHLY_TH'})
    data_single = data_single.replace([-np.inf,np.inf],np.nan)
    data_single_output = data_single.query("ALART.notna()")

    data_single_output = data_single_output.merge(rank_list.rename(columns={'C_FUNDCODE':'PORTFOLIO_CODE'})[['PORTFOLIO_CODE','PFL_DD_BEGDT','LIST_DD_BEGDT']],how ='left',on = 'PORTFOLIO_CODE')
    data_single_output['CONTRIBUTION'] = data_single_output['CONTRIBUTION']*data_single_output['EX_DD']


    data['ALART'] =  data.apply(lambda x:'早期预警' if (((abs(x['CRT_DD'])>= x['DD_THOLD']*0.4) & (abs(x['EX_DD']) > 0.2*x['DD_THOLD'])) ) else  None  ,axis =1)
    data['ALART'] =  data.apply(lambda x:'重点预警' if (((abs(x['CRT_DD'])>= x['DD_THOLD']*0.6) & (abs(x['EX_DD']) > 0.4*x['DD_THOLD'])) ) else  x['ALART']  ,axis =1)
    data_output = data.query("ALART.notna()").drop(['RETURN','CRTDD_DURA','RANKING_LIST'],axis =1)
    if len(data_output)>0:
        data_output = pd.concat([data_output[['ALART']],data_output.iloc[:,:-1]],axis= 1)
        data_output = data_output.merge(rank_list.rename(columns={'C_FUNDCODE':'PORTFOLIO_CODE'})[['PORTFOLIO_CODE','PFL_DD_BEGDT']],how ='left',on = 'PORTFOLIO_CODE')
        data_output.columns = ['ALART','D_DATE','PORTFOLIO_CODE','C_FUNDNAME','DD','RET_BGT','LST_DD_MID','EX_DD','DD_BEG_DT']
        data_output['DD_ALART'] = data_output.apply(lambda x:x['RET_BGT']*0.4 if x['ALART'] == '早期预警' else (x['RET_BGT']*0.6 if x['ALART'] == '重点预警' else np.nan ),axis= 1)
        data_output['EX_DD_ALART'] = data_output.apply(lambda x:x['RET_BGT']*0.2 if x['ALART'] == '早期预警' else (x['RET_BGT']*0.4 if x['ALART'] == '重点预警' else np.nan ),axis= 1)
        data_output = data_output[['ALART','D_DATE','PORTFOLIO_CODE','C_FUNDNAME','DD','DD_ALART','LST_DD_MID','EX_DD','EX_DD_ALART','DD_BEG_DT','RET_BGT']]
        data_output.columns = ['ALART_TYPE','D_DATE','PORTFOLIO_CODE','PORTFOLIO_NAME','DD','DD_ALART','LST_DD_MID','EX_DD','EX_DD_ALART','DD_BEG_DT','RET_BGT']
        data_output['ALART'] = '是'
        data_output['DD']*=100
        data_output['DD_ALART']*=100
        data_output['LST_DD_MID']*=100
        data_output['EX_DD']*=100
        data_output['EX_DD_ALART']*=100
        data_output['RET_BGT']*=100
        data_output['DD_ALART'] = -data_output['DD_ALART']
        data_output['EX_DD_ALART'] = -data_output['EX_DD_ALART']
        data_output['RET_BGT'] = -data_output['RET_BGT']
        data_output = data_output.merge(rm_threshold_cbond['PORTFOLIO_CODE'], how='inner', on='PORTFOLIO_CODE')
        data_output = data_output.merge(mf_basic_info, on='PORTFOLIO_CODE', how='left')
        data_output = data_output[
            ['ALART_TYPE', 'D_DATE', 'PORTFOLIO_CODE', 'PORTFOLIO_NAME', 'DD', 'DD_ALART', 'LST_DD_MID', 'EX_DD',
             'EX_DD_ALART', 'DD_BEG_DT', 'RET_BGT', 'ALART', 'PORTFOLIO_TYPE', 'PORTFOLIO_MANAGER']]
        data_output = data_output.replace('永赢合嘉一年持有期混合', '永赢合嘉一年持有混合')
        self.insert2db_single("cb_dd_alart", data_output, t=self.basedate)
        data_output = data_output[
            ['ALART_TYPE', 'D_DATE', 'PORTFOLIO_CODE', 'PORTFOLIO_NAME', 'DD', 'DD_ALART', 'LST_DD_MID', 'EX_DD',
             'EX_DD_ALART', 'DD_BEG_DT', 'RET_BGT', 'PORTFOLIO_TYPE', 'PORTFOLIO_MANAGER']]
        data_output.columns = ['预警类别', '日期', '组合代码', '组合名称', '当前回撤', '回撤预警线', '竞品回撤中位数', '超额回撤', '超额回撤预警线', '回撤开始日期',
                               '回撤预算', '组合类型', '基金经理']
    else:
        data_output = pd.DataFrame(columns = ['预警类别', '日期', '组合代码', '组合名称', '当前回撤', '回撤预警线', '竞品回撤中位数', '超额回撤', '超额回撤预警线', '回撤开始日期',
                               '回撤预算', '组合类型', '基金经理'])
    data_output.to_excel(os.path.join(
        r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\日频数据\DailyIndicators\DailyResult',
        '{}_可转债回撤预警.xlsx'.format(datestr)), index=False, sheet_name='可转债回撤预警')

    data_single_output.columns = ['ALART_TYPE','D_DATE','PORTFOLIO_CODE','PORTFOLIO_NAME','CBOND_CODE','CBOND_NAME','CBOND_TYPE','WEIGHT','MANAGER_WEIGHT_UB','CB_RET','CB_RET_ALART','CB_RET_MONTHLY','INDEX_RET_MONTHLY','EX_RET_MONTHLY','EX_RET_MONTHLY_ALART','CONTRI_BASEDATE','CBOND_CONTRI','CBOND_CONTRI_ALART','EX_RET','RET_BGT','PFL_DD_DT','LST_DD_DT']
    data_single_output['ALART'] = '是'
    data_single_output['CB_RET']*=100
    data_single_output['CB_RET_ALART']*=100
    data_single_output['CB_RET_MONTHLY']*=100
    data_single_output['INDEX_RET_MONTHLY']*=100
    data_single_output['EX_RET_MONTHLY']*=100
    data_single_output['EX_RET_MONTHLY_ALART']*=100
    data_single_output['CBOND_CONTRI']*=100
    data_single_output['CBOND_CONTRI_ALART']*=100
    data_single_output['EX_RET']*=100
    data_single_output['RET_BGT']*=100
    data_single_output['RET_BGT'] = -data_single_output['RET_BGT']
    data_single_output = data_single_output.merge(rm_threshold_cbond['PORTFOLIO_CODE'],how = 'inner',on = 'PORTFOLIO_CODE')
    data_single_output = data_single_output.merge(mf_basic_info,on = 'PORTFOLIO_CODE',how ='left')
    data_single_output = data_single_output[['ALART_TYPE','D_DATE','PORTFOLIO_CODE','PORTFOLIO_NAME','CBOND_CODE','CBOND_NAME','CBOND_TYPE','WEIGHT','MANAGER_WEIGHT_UB','CB_RET','CB_RET_ALART','CB_RET_MONTHLY','INDEX_RET_MONTHLY','EX_RET_MONTHLY','EX_RET_MONTHLY_ALART','CONTRI_BASEDATE','CBOND_CONTRI','CBOND_CONTRI_ALART','EX_RET','RET_BGT','PFL_DD_DT','LST_DD_DT','ALART', 'PORTFOLIO_TYPE', 'PORTFOLIO_MANAGER']]
    data_single_output=data_single_output.replace('永赢合嘉一年持有期混合','永赢合嘉一年持有混合')	
    self.insert2db_single("cb_dd_single_alart", data_single_output, t=self.basedate)

    data_single_output = data_single_output[['ALART_TYPE','D_DATE','PORTFOLIO_CODE','PORTFOLIO_NAME','CBOND_CODE','CBOND_NAME','CBOND_TYPE','WEIGHT','MANAGER_WEIGHT_UB','CB_RET','CB_RET_ALART','CB_RET_MONTHLY','INDEX_RET_MONTHLY','EX_RET_MONTHLY','EX_RET_MONTHLY_ALART','CONTRI_BASEDATE','CBOND_CONTRI','CBOND_CONTRI_ALART','EX_RET','RET_BGT','PFL_DD_DT','LST_DD_DT', 'PORTFOLIO_TYPE', 'PORTFOLIO_MANAGER']]
    data_single_output.columns = ['预警类别','日期','组合代码','组合名称','转债代码','转债简称','转债性质','权重','经理权重上限','转债单日跌幅','转债单日跌幅预算','转债月度跌幅','转债指数月度收益','转债月度超额跌幅','转债月度超额跌幅预算','回撤贡献计算基日','回撤贡献','回撤贡献百分比预警线','超额回撤','回撤预算','组合回撤开始日','竞品回撤开始日','组合类型','基金经理']
    data_single_output.to_excel(os.path.join(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\日频数据\DailyIndicators\DailyResult','{}_可转债回撤预警_个券.xlsx'.format(datestr)),index=False,sheet_name = '可转债预警-个券回撤')