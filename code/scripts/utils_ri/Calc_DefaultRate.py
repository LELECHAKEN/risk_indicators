'''
@Description: Calc default rate of individual bond based on credit transition matrix.
@Author: Wangp
@Date: 2020-03-10 09:38:18
@LastEditTime: 2020-04-28 18:22:00
@LastEditors: Wangp
'''
import pandas as pd
from math import floor
import numpy as np

class credit_transition_matrix(object):
    """
    该对象用于计算违约率。

    函数：
    -------
    calc_defaultRate : 计算违约率的函数

    """
    def __init__(self):
        tb = pd.read_excel(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\CreditTransitionMatrix.xlsx', engine='openpyxl').dropna(thresh=14)
        cols_notna = [x for x in tb.columns if 'Unnamed' not in x]
        self.tb=tb.reindex(columns=cols_notna).set_index('Final')

    def calc_defaultRate(self,rating,ttm):
        '''
        ---calculating credit transition rate and default rate---
        :param rating: credit rating
        :param ttm: time to maturity
        :return: transition rate, default rate
        '''
        rl=self.tb.columns.to_list() #rating list
        matrix_tn = pd.Series(np.zeros(14), index=rl)
        matrix_tn[rating]=1
        #计算0年至整数年的累计违约率(ie. 0yr-2yr)
        for i in range(floor(ttm)):
            temp=pd.Series(np.zeros(14),index=rl)
            for r in rl:
                matrix_rating=self.tb.loc[r,:].copy()  #该评级的1年迁移概率
                temp=matrix_tn[r]*matrix_rating+temp
            matrix_tn=temp.copy()
        #计算整数年至期限的累计违约率（ie. 2yr-2.3yr)
        temp=pd.Series(np.zeros(14),index=rl)
        for r in rl:
            matrix_rating = self.tb.loc[r, :].copy()  # 该评级的1年迁移概率
            for r_tmp in rl:
                transition_rating_adj=1-(1-matrix_rating[r_tmp])**(ttm-floor(ttm)) #该评级调整后的迁移概率
                matrix_rating[r_tmp]=transition_rating_adj
            matrix_rating[r]=1-sum(matrix_rating[matrix_rating.index!=r])
            temp = matrix_tn[r] * matrix_rating + temp
        #加总得到累计迁移概率
        matrix_tn=temp.copy()
        #累计违约率
        dr=matrix_tn['D']
        return matrix_tn,dr

# #使用方法
# ctm=credit_transition_matrix()
# # matrix_tn,default_rate=ctm.calc_defaultRate('AA-',2.3)  #第一个参数为中债隐含评级，第二个参数为到期日
# ratings = ['AAA+','AAA','AAA-','AA+','AA','AA-','A+','A','A-','BBB+','BBB','BB','B','CCC','CC']
# ttm_list = [0.6, 1, 2, 3, 5]
# dr_list = []
# for rating in ratings:
#     dr_temp = []
#     for ttm in ttm_list:
#         dr = ctm.calc_defaultRate(rating, ttm)[1]
#         dr_temp.append([rating, ttm, dr])
#     dr_list.append(dr_temp)
# cols = ['6个月', '1年', '2年', '3年', '5年']
# res = pd.DataFrame(dr_list, columns=cols, index=ratings)