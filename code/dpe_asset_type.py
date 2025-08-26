
import pandas as pd
from datetime import datetime
from scripts.utils.log_utils import logger
from scripts.asset_type import AllAsset, retrieve_n_tradeday
from scripts.LiquidityHolders import LiquidityAsset, LiquidityLiablility


if __name__ == '__main__':
    today = datetime.today().strftime('%Y-%m-%d')

    all_asset = AllAsset(today)  # dpe_asset_type 表
    liq_asset = LiquidityAsset(today)  # 流动性模块补充
    liq_liability = LiquidityLiablility(today)

    for t in [retrieve_n_tradeday(today, i) for i in [0, 1, 2]]:   # 过去1、2、3个交易日
        all_asset.dpe_asset_type(t)
        liq_asset.rc_lr_asset_realization(t)
        liq_asset.rc_lr_liquidity_level(t)
        liq_liability.rc_lr_holders(t)

    for t in [(datetime.today() - pd.Timedelta(i, 'd')).strftime('%Y-%m-%d') for i in [1, 2, 3]]:  # 过去1、2、3个自然日
        logger.info('开始计算%s日的资产类型' % t)
        all_asset.dpe_asset_type_nottd(t)

    # # 回跑交易日历史数据
    # tds = liq_asset.get_period_tradedays(t0='2024-06-01', t1='2024-06-06')
    # for t in tds:
    #     liq_liability = LiquidityLiablility(t=t)
    #     liq_liability.rc_lr_holders()
    #     liq_asset.rc_lr_asset_realization(t=t)
    #     liq_asset.rc_lr_liquidity_level(t=t)

    # tds = [t.strftime('%Y-%m-%d') for t in pd.date_range('2025-04-15', '2025-05-06')]
    # for t in tds:
    #     all_asset.dpe_asset_type(t)
    #     all_asset.dpe_asset_type_nottd(t)