#!/usr/bin/env python
#!-*-coding:utf-8 -*-
#!@Time   : 2020/3/5 16:09
#!@Author : Zhang,Yifan
#!@File   : Calc_AssetVaR.py

import pandas as pd
import numpy as np
from .var import get_tb_params_ir, get_tb_params_slope



class model_VaR(object):
    """
    该对象用于管理各类VaR模型，以及计算资产相应的VaR值。

    函数：
    -------
    __equityVaR : 私有函数，用于计算股票的VaR
    __irbondVaR : 私有函数，用于计算利率债的VaR
    __creditbondVaR : 私有函数，用于计算信用债的VaR
    __convertibleVaR : 私有函数，用于计算可转债的VaR
    __absVaR : 私有函数，用于计算ABS的VaR
    calc_VaR：调用各资产的VaR模型，并输出VaR值
    """
    def __init__(self):
        self.tb_params_ir = get_tb_params_ir()
        self.tb_params_slope = get_tb_params_slope()

    def __equityVaR(self):
        '''
        计算股票的VaR，模型暂无
        :return: None
        '''
        return None

    def __irbondVaR(self,t,timeToMaturity,rating=None):
        '''
        计算利率债的VaR
        :param t: TimeStamp,analysis day
        :param timeToMaturity: float,time to maturity
        :return: 
        '''
        def factor_matrix():
            row_name=['平','中度','陡']
            col_name=['25分位以下','25~50分位','50~75分位','75分位以上']
            # 0-1年
            irbond_1yr=[0.19,0.28,0.36,0.40,0.44,0.36,0.34,0.33,0.48,0.47,0.36,0.28]
            irbond_1yr=np.reshape(irbond_1yr,(3,4))
            irbond_1yr=pd.DataFrame(irbond_1yr,index=row_name,columns=col_name)
            # 1-3年
            irbond_3yr=[0.21,0.3,0.37,0.40,0.41,0.33,0.32,0.29,0.44,0.43,0.36,0.25]
            irbond_3yr=np.reshape(irbond_3yr,(3,4))
            irbond_3yr=pd.DataFrame(irbond_3yr,index=row_name,columns=col_name)
            # 3-5年
            irbond_5yr=[0.23,0.33,0.38,0.41,0.36,0.31,0.3,0.25,0.42,0.38,0.31,0.23]
            irbond_5yr=np.reshape(irbond_5yr,(3,4))
            irbond_5yr=pd.DataFrame(irbond_5yr,index=row_name,columns=col_name)
            # 5-10年以上
            irbond_10yr=[0.25,0.35,0.39,0.45,0.29,0.27,0.27,0.24,0.40,0.33,0.26,0.18]
            irbond_10yr=np.reshape(irbond_10yr,(3,4))
            irbond_10yr=pd.DataFrame(irbond_10yr,index=row_name,columns=col_name)
            factors_dict = {'1年': irbond_1yr,
                            '3年': irbond_3yr,
                            '5年': irbond_5yr,
                            '10年': irbond_10yr}
            return factors_dict
        def reclassify_ttm(timeToMaturity):
            if timeToMaturity <= 1.5:
                label = '1年'
            elif timeToMaturity > 1.5 and timeToMaturity <= 3.5:
                label = '3年'
            elif timeToMaturity > 3.5 and timeToMaturity <= 5.5:
                label = '5年'
            else:
                label = '10年'
            return label
        rrm=reclassify_ttm(timeToMaturity)
        param_level='利率债：'+rrm
        param_slope='利率债'
        ir_level=self.tb_params_ir[param_level][t]  #利率债的收益率水平
        slope_level=self.tb_params_slope[param_slope][t]  #利率债的斜率水平
        factors_dict=factor_matrix()
        VaR=factors_dict[rrm][ir_level][slope_level]  #计算VaR
        return VaR    

    def __creditbondVaR(self,t,timeToMaturity,rating):
        '''
        计算信用债的VaR，包括同业存单，可转债
        :param t: TimeStamp,analysis day
        :param timeToMaturity: float,time to maturity
        :param rating: string,china bond implicit rating
        :return:
        '''
        def factor_matrix():
            row_name = ['平', '中度', '陡']
            col_name = ['25分位以下', '25~50分位', '50~75分位', '75分位以上']
            # 信用AAA 1年
            cb_AAA_1yr = [0.76, 0.65, 0.58, 0.60, 0.30, 0.34, 0.63, 0.62, 0.43, 0.39, 0.33, 0.22]
            cb_AAA_1yr = np.reshape(cb_AAA_1yr, (3, 4))
            cb_AAA_1yr = pd.DataFrame(cb_AAA_1yr, index=row_name, columns=col_name)
            # 信用AAA 3年
            cb_AAA_3yr = [0.71, 0.59, 0.53, 0.55, 0.29, 0.33, 0.55, 0.54, 0.38, 0.34, 0.27, 0.20]
            cb_AAA_3yr = np.reshape(cb_AAA_3yr, (3, 4))
            cb_AAA_3yr = pd.DataFrame(cb_AAA_3yr, index=row_name, columns=col_name)
            # 信用AAA 5年
            cb_AAA_5yr = [0.66, 0.54, 0.49, 0.50, 0.28, 0.32, 0.50, 0.51, 0.36, 0.32, 0.25, 0.18]
            cb_AAA_5yr = np.reshape(cb_AAA_5yr, (3, 4))
            cb_AAA_5yr = pd.DataFrame(cb_AAA_5yr, index=row_name, columns=col_name)
            # 信用AA+ 1年
            cb_AAp_1yr = [0.96, 0.89, 0.78, 0.67, 0.20, 0.34, 0.84, 0.89, 0.56, 0.55, 0.45, 0.26]
            cb_AAp_1yr = np.reshape(cb_AAp_1yr, (3, 4))
            cb_AAp_1yr = pd.DataFrame(cb_AAp_1yr, index=row_name, columns=col_name)
            # 信用AA+ 3年
            cb_AAp_3yr = [0.83, 0.71, 0.60, 0.50, 0.17, 0.29, 0.72, 0.78, 0.53, 0.52, 0.35, 0.16]
            cb_AAp_3yr = np.reshape(cb_AAp_3yr, (3, 4))
            cb_AAp_3yr = pd.DataFrame(cb_AAp_3yr, index=row_name, columns=col_name)
            # 信用AA+ 5年
            cb_AAp_5yr = [0.70, 0.56, 0.48, 0.42, 0.16, 0.24, 0.59, 0.60, 0.50, 0.45, 0.31, 0.12]
            cb_AAp_5yr = np.reshape(cb_AAp_5yr, (3, 4))
            cb_AAp_5yr = pd.DataFrame(cb_AAp_5yr, index=row_name, columns=col_name)
            # 信用AA 1年
            cb_AA_1yr = [0.51, 0.48, 0.46, 0.39, 0.40, 0.44, 0.53, 0.50, 0.56, 0.49, 0.34, 0.24]
            cb_AA_1yr = np.reshape(cb_AA_1yr, (3, 4))
            cb_AA_1yr = pd.DataFrame(cb_AA_1yr, index=row_name, columns=col_name)
            # 信用AA 3年
            cb_AA_3yr = [0.50, 0.47, 0.41, 0.34, 0.36, 0.44, 0.47, 0.48, 0.54, 0.46, 0.32, 0.20]
            cb_AA_3yr = np.reshape(cb_AA_3yr, (3, 4))
            cb_AA_3yr = pd.DataFrame(cb_AA_3yr, index=row_name, columns=col_name)
            # 信用AA 5年
            cb_AA_5yr = [0.45, 0.42, 0.36, 0.29, 0.35, 0.40, 0.44, 0.44, 0.52, 0.45, 0.24, 0.19]
            cb_AA_5yr = np.reshape(cb_AA_5yr, (3, 4))
            cb_AA_5yr = pd.DataFrame(cb_AA_5yr, index=row_name, columns=col_name)
            factors_dict_cb_AAA = {'1年': cb_AAA_1yr,
                                   '3年': cb_AAA_3yr,
                                   '5年': cb_AAA_5yr}
            factors_dict_cb_AAp = {'1年': cb_AAp_1yr,
                                   '3年': cb_AAp_3yr,
                                   '5年': cb_AAp_5yr}
            factors_dict_cb_AA = {'1年': cb_AA_1yr,
                                  '3年': cb_AA_3yr,
                                  '5年': cb_AA_5yr}
            factors_dict = {'AAA': factors_dict_cb_AAA,
                            'AA+': factors_dict_cb_AAp,
                            'AA': factors_dict_cb_AA}
            return factors_dict
        def reclassify_ttm(timeToMaturity):
            if timeToMaturity <= 1.5:
                label = '1年'
            elif timeToMaturity > 1.5 and timeToMaturity <= 3.5:
                label = '3年'
            elif timeToMaturity > 3.5 and timeToMaturity <= 5.5:
                label = '5年'
            else:
                label = '5年'
            return label
        def reclassify_rating(rating):
            def impRat_mapping():
                # implicit rating
                impRat = {'AAA+': 'AAA',
                          'AAA': 'AAA',
                          'AAA-': 'AAA',
                          'AA+': 'AAA',
                          'AA': 'AA+',
                          'AA(2)': 'AA+',
                          'AA-': 'AA',
                          'A+': 'AA',
                          'A': 'AA',
                          'A-': 'AA',
                          'BBB+': 'AA',
                          'BBB': 'AA',
                          'BB': 'AA',
                          'B': 'AA',
                          'CCC': 'AA',
                          'CC': 'AA',
                          'C': 'AA',
                          'D': 'AA',
                          'NR': 'AA'}
                return impRat
            irmap=impRat_mapping()  #隐含评级与评级的映射关系
            rat=irmap[rating]  #映射
            return rat
        rrm=reclassify_ttm(timeToMaturity)          #剩余期限分类
        rat=reclassify_rating(rating)               #隐含评级分类
        param_level='信用债'+rat+'：'+rrm
        param_slope='信用债'+rat
        ir_level=self.tb_params_ir[param_level][t]  #利率债的收益率水平
        slope_level=self.tb_params_slope[param_slope][t]  #利率债的斜率水平
        factors_dict=factor_matrix()
        VaR=factors_dict[rat][rrm][ir_level][slope_level]  #计算VaR
        return VaR

    def __convertibleVaR(self,t,timeToMaturity,rating):
        '''
        计算可转债的VaR，暂时沿用信用债VaR模型
        :param t: TimeStamp,analysis day
        :param timeToMaturity: float,time to maturity
        :param rating: string,china bond implicit rating
        :return:
        '''
        VaR=self.__creditbondVaR(t,timeToMaturity,rating)
        return VaR

    def __absVaR(self,t,timeToMaturity,rating):
        '''
        计算ABS的VaR，暂时沿用信用债VaR模型
        :param t: TimeStamp,analysis day
        :param timeToMaturity: float,time to maturity
        :param rating: string,china bond implicit rating
        :return:
        '''
        VaR=self.__creditbondVaR(t,timeToMaturity,rating)
        return VaR

    def calc_VaR(self,underlyingType,t,timeToMaturity,rating=None):
        '''
        资产与对应的var模型的映射关系
        :param underlyingType:string,includes:'股票','利率债', '信用债', '转债','ABS'
        :return:function
        '''
        if underlyingType not in ['股票', '利率债', '信用债', '可转债', 'ABS']:
            return 0
        model_mapping={'股票':self.__equityVaR,
               '利率债':self.__irbondVaR,
               '信用债':self.__creditbondVaR,
               '可转债':self.__convertibleVaR,
               'ABS':self.__absVaR}
        model=model_mapping[underlyingType]
        VaR=model(t,timeToMaturity,rating)
        return VaR


# 模型用法：
# t=pd.Timestamp(2022,9,16)  #t日
# model=model_VaR(t)         #创建对象
# VaR=model.calc_VaR('信用债',t,3.3,'AA+')  #输入参数，t日，剩余期限和隐含评级。对于利率债，隐含评级参数可以留空，或者输任何评级


