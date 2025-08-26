'''
@Description: including individual secs and industry concentration
@Author: Wangp
@Date: 2020-03-16 11:24:02
LastEditTime: 2021-06-15 15:48:51
LastEditors: Wangp
'''

import pandas as pd
from .utils_ri.RiskIndicators import RiskIndicators


class ConcentrationIndicators(RiskIndicators):
    def __init__(self, t, ptf_codes=None):
        RiskIndicators.__init__(self, t, ptf_codes)
        self.HoldingsCleaning()
        self.RepoCleaning()

    def _calcTopN_secs(self, data, n=3, method='Total'):
        '''
        计算前N大股票的名称及占比\n
        :param data: DataFrame, 需要计算前n大口径的持仓证券表
        :param n: int, 前N大
        :param method: string, 计算总资产/净资产占比, 默认总资产
        :return: DataFrame
        '''
        res0 = data.sort_values(by=['F_ASSETRATIO'], ascending=False).iloc[:n, :].copy()
        topN_secs = ','.join(res0['C_SUBNAME_BSH'].tolist())
        if method == 'Total':
            topN_secRatio = res0['F_ASSETRATIO'].sum() / data['F_ASSETRATIO'].sum()
        else:
            topN_secRatio = res0['F_ASSETRATIO'].sum() / 100
        res = pd.DataFrame([topN_secs, topN_secRatio], index=['top%d_sec'%n, 'top%d_secRatio'%n]).T

        return res

    def getIndividualConcentration(self, n=3, method='Total', data=''):
        '''
        计算个券前N大\n
        :param n: int, 前N大
        :param method: string, 计算总资产/净资产占比, 默认总资产
        :param data: DataFrame, 默认取当日全量债券持仓表
        :return: DataFrame
        '''
        if type(data) == str:
            bond_holingds_adj = self.bond_holdings[self.bond_holdings['INDUSTRY_SW'] != '行业_利率债'].copy()
            holdings = pd.concat([bond_holingds_adj, self.stock_holdings], ignore_index=True)
        else:
            holdings = data.copy()

        res_cols = ['C_FUNDNAME', 'D_DATE', 'top%d_sec' % n, 'top%d_secRatio' % n]
        if holdings.empty:
            return pd.DataFrame(columns=res_cols)

        res = holdings.groupby(['C_FUNDNAME', 'D_DATE']).apply(self._calcTopN_secs, n, method).reset_index()
        if 'level_2' in res.columns:
            res = res.drop(columns=['level_2'])

        if len(res) == 1: # 如果仅有一条持仓数据，groupby的结果中，没有c_fundname和d_date
            res[['C_FUNDNAME', 'D_DATE']] = [holdings['C_FUNDNAME'].iloc[0], holdings['D_DATE'].iloc[0]]
            res = res.reindex(columns=res_cols)

        return res
        
    # 获取组合前n大行业集中度持仓占比，剔除利率债，股债统一口径
    def getConcentration(self, n=1):
        '''
        获取组合前n大行业集中度持仓占比，剔除利率债，股债统一口径\n
        :param n: int, 前N大
        :return: DataFrame
        '''
        bond_holdings_adj = self.bond_holdings[self.bond_holdings['INDUSTRY_SW'] != '行业_利率债'].copy()
        holdings = pd.concat([bond_holdings_adj, self.stock_holdings], ignore_index=True)
        if holdings.empty:
            return pd.DataFrame(columns=['D_DATE', 'C_FUNDNAME', 'Industry_%d' % n, 'Industry_%d_asset'%n])

        holdings['INDUSTRY_SW'] = holdings['INDUSTRY_SW'].replace('城投债', '城投')

        data = holdings.groupby(['D_DATE', 'C_FUNDNAME', 'INDUSTRY_SW'])['F_ASSET'].sum().sort_values(ascending=False).reset_index()
        if data.empty:
            return pd.DataFrame(columns=['D_DATE', 'C_FUNDNAME', 'Industry_%d' % n, 'Industry_%d_asset' % n])

        data['Order'] = data.groupby(['D_DATE', 'C_FUNDNAME'])['F_ASSET'].rank(method='first', ascending=False)
        res_asset = data.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: (x.loc[x['Order'] <= n, 'F_ASSET']).sum()/x['F_ASSET'].sum()).reset_index().rename(columns={0: 'Industry_%d_asset'%n})
        res_name = data.groupby(['D_DATE', 'C_FUNDNAME']).apply(lambda x: ','.join(x.loc[x['Order'] <= n, 'INDUSTRY_SW'])).reset_index().rename(columns={0: 'Industry_%d'%n})

        res = pd.merge(res_name, res_asset, on=['D_DATE', 'C_FUNDNAME'], how='outer')

        return res
    
    def CalculateAll(self):
        # 个券集中度，前N大
        self.top3_sec = self.getIndividualConcentration(3)
        self.top5_sec = self.getIndividualConcentration(5)

        stock_holdings = self.stock_holdings.copy()
        self.top10_stock = self.getIndividualConcentration(n=10, method='AssetRatio', data=stock_holdings)
        self.top10_stock.columns = [x.replace('sec', 'stock') for x in self.top10_stock.columns]

        bond_holdings = self.bond_holdings[self.bond_holdings['WINDL1TYPE'] != '资产支持证券'].copy()
        self.top10_bond = self.getIndividualConcentration(n=10, method='AssetRatio', data=bond_holdings)
        self.top10_bond.columns = [x.replace('sec', 'bond') for x in self.top10_bond.columns]
        
        abs_holdings = self.bond_holdings[self.bond_holdings['WINDL1TYPE'] == '资产支持证券'].copy()
        self.top10_abs = self.getIndividualConcentration(n=10, method='AssetRatio', data=abs_holdings)
        self.top10_abs.columns = [x.replace('sec', 'abs') for x in self.top10_abs.columns]

        # 行业集中度，前N大
        self.ind_1 = self.getConcentration(n=1)                                  # 前1大行业持仓占比，剔除利率债
        self.ind_3 = self.getConcentration(n=3)                                  # 前3大行业持仓占比，剔除利率债
        self.ind_5 = self.getConcentration(n=5)                                  # 前5大行业持仓占比，剔除利率债

        res_list = [self.top3_sec, self.top5_sec, self.ind_1, self.ind_3, self.ind_5, self.top10_stock, self.top10_bond, self.top10_abs]
        key_cols = ['C_FUNDNAME', 'D_DATE']
        res = self.holdings[key_cols].drop_duplicates()
        for temp in res_list:
            if temp.empty:
                new_cols = [i for i in list(temp.columns) if i not in key_cols]
                res = res.reindex(columns=list(res.columns)+new_cols)
            else:
                res = pd.merge(res, temp, on=key_cols, how='left')

        res['PORTFOLIO_CODE'] = res['C_FUNDNAME'].map(self.fundname_to_code)
        self.res_all = res.copy()
        return res


if __name__ == '__main__':
    t = '2021-06-15'
    ConcenIdx = ConcentrationIndicators(t)
    res = ConcenIdx.CalculateAll()
    res.to_excel(r'C:\Users\wangp\Desktop\ConcentrationIndex.xlsx', index=False)
    print('Done.')