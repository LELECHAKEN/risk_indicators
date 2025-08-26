"""
单一接口主程序
"""
import os
import pandas as pd

from scripts.utils.log_utils import logger
from scripts.settings import config, DIR_OF_MAIN_PROG
from scripts.db import OracleDB, sqls_config
from scripts.db.db_utils import convert_columns
from scripts import demo_code

from scripts.updateFundNav import IntegrateNav
from scripts.BasicIndicators import BasicIndicators
from scripts.ConcentrationIndicators import ConcentrationIndicators
from scripts.MarketIndicators import MarketIndicators
from scripts.MarketIndicators_b import BenchmarkRiskReturn
from scripts.CreditIndicators import CreditIndicators
from scripts.LiquidityIndicators import LiquidityIndicators
from scripts.LiquidityIndicators_core import LiquidityIndicators_core
from scripts.RiskIndicator_mg import RiskIndicator_mg
from scripts.RiskIndicator_mismatch import RiskIndicator_mismch
from scripts.CBIndicators import CBIndicators
from scripts.derivatives import Derivatives
from scripts.risk_warning import PortfolioWarning

def saveData(data_path, Indicators):
    writer = pd.ExcelWriter(data_path + 'Holdings.xlsx')
    Indicators.bond_holdings.to_excel(writer, sheet_name='bond_holdings', index=False)
    Indicators.stock_holdings.to_excel(writer, sheet_name='stock_holdings', index=False)
    Indicators.holdings.to_excel(writer, sheet_name='holdings', index=False)
    writer.save()

    writer = pd.ExcelWriter(data_path + 'data_db.xlsx')
    Indicators.data_jy.to_excel(writer, sheet_name='data_jy', index=False)
    Indicators.data_wind.to_excel(writer, sheet_name='data_wind', index=False)
    writer.save()

    Indicators.data_wind_equity.to_excel(data_path + 'data_wind_equity.xlsx', index=False)

    writer = pd.ExcelWriter(data_path + 'data_benchmark.xlsx')
    Indicators._benchmark.to_excel(writer, sheet_name='benchmark', index=False)
    Indicators._benchmark_gk.to_excel(writer, sheet_name='benchmark_gk', index=False)
    Indicators._benchmark_ind.to_excel(writer, sheet_name='benchmark_ind', index=False)
    Indicators._benchmark_rating.to_excel(writer, sheet_name='benchmark_rating', index=False)
    writer.save()

    Indicators._yield_map.to_excel(data_path + 'yield_map.xlsx', index=False)
    Indicators.data_fund.to_excel(data_path + 'data_fund.xlsx', index=False)

    Indicators._lev_all.to_excel(data_path + '_lev_all.xlsx', index=False)
    Indicators._repo_all.to_excel(data_path + '_repo_all.xlsx', index=False)

def check_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == '__main__':
    t = '2025-03-07'
    # data_path = r'E:\RiskQuant\风险指标\DailyIndicators\\'
    # save_path = r'E:\RiskQuant\风险指标\DailyIndicators\%s\\'%t.replace('-', '')
    demo_code.insert_liq_factor(t)                        # 仅在流动性模型系数更新后调用

    # 删除某一天的数据
    # t = '2021-06-08'
    # table_names = ['RC_CONCENTRATION', 'RC_STYLE', 'RC_MR_RETURN', 'RC_MR_RETURN_BCH', 'RC_MR_HOLDING', 'RC_CREDIT', 'RC_LIQUIDITY', 'RC_LIQ_CORE']
    # demo_code.delete_t_sheet(table_names, t, 'quant')

    # Const data insertion
    # file_list = ['DPE_SPREAD_AD', 'DPE_SPREADIND_MAP', 'DPE_YIELDCURVE_MAP']
    # for file_name in file_list:
    #     data = pd.read_excel(r'E:\RiskQuant\风险指标\系统化\数据库梳理\%s.xlsx'%file_name, engine='openpyxl')
    #     data = demo_code._format_string(data)
    #     demo_code.insert_table(file_name, data, 'quant', if_exists='replace')
    # dd_map = pd.read_excel(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\PortfolioType\回撤阈值.xlsx',
    #                         engine='openpyxl').rename(columns={'基金名称': 'C_FUNDNAME', '回撤阈值': 'dd_threshold'})
    # demo_code.insert_table('manual_dd_threshold', dd_map, 'quant')