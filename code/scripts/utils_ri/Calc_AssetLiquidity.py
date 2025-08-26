#!/usr/bin/env python
#!-*-coding:utf-8 -*-
#!@Time   : 2020/3/5 16:08
#!@Author : Zhang,Yifan
#!@File   : Calc_AssetLiquidity.py

import pandas as pd
import numpy as np
from pathlib import Path
from WindPy import w
import warnings
warnings.filterwarnings('ignore')

class model_Liqudity(object):
    def __init__(self):
        self.tb_td = pd.read_excel(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\TradeDayDate\chinabond_tradedaydate.xlsx', engine='openpyxl').loc[:, '交易日（TIMESTAMP）':].copy()  # 交易日数据

    def get_data(self,t):
        '''
        该函数用于读取计算流动性所需的公用数据
        :param t: TimeStamp，分析日
        :return: None
        '''
        self.t=t
        self.__data_processing()  #数据处理

    def __data_processing(self):
        '''
        该函数用于对数据做初步转换
        :param t: TimeStamp，分析日
        :return: None
        '''
        self.t0=self.date_format(self.t) #日期格式转换为TimeStamp
        self.t0_str=self.t0.strftime('%Y%m%d') #日期格式转换为字符串格式YYYYMMDD
        #计算第前20个交易日的日期
        row=self.tb_td.loc[self.tb_td['交易日（TIMESTAMP）']<=self.t0,:].shape[0]
        self.t_prev=self.tb_td.iloc[row-20,0]  #第前20个交易日日期 TimeStamp
        self.t_prev_str=self.t_prev.strftime('%Y%m%d') #字符串格式YYYYMMDD

    def date_format(self,t):
        '''
        该函数将输入日期转换为TimeStamp格式
        :param t: String/Datetime/Date/TimeStamp/Most others，分析日
        :return: TimeStamp
        '''
        t0=pd.to_datetime(t)
        return t0

    def Wind_Connect(self):
        '''
        检查万德API连接状态
        :return: None
        '''
        if w.isconnected() == False:
            w.start()

class liquidity_Collateral(model_Liqudity):
    def __init__(self,t):
        super().__init__()
        self.file_path_collateral=Path(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\Collaterals')  #质押券数据目录地址
        self.file_path_repo=Path(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\Collaterals')  #回购到期浏览表
        self.file_path_coding=Path(r'\\shaoafile01\RiskManagement\1. 基础数据')  #代码匹配表
        self.get_data(t)

    def get_data(self,t):
        '''
        该函数用于读取计算质押券所需的数据
        :return: None
        '''
        super().get_data(t)
        #质押券
        file_path_tb_ib=self.file_path_collateral / self.t0_str / '银行间回购质押券.xls' #路径
        file_path_tb_exchg = self.file_path_collateral / self.t0_str / '综合信息查询_质押券.xls'
        file_path_tb_exchg_p = self.file_path_collateral / self.t0_str / '综合信息查询_标准券.xls'
        self.tb_ib=pd.read_excel(str(file_path_tb_ib),skip_footer=1)  #银行间质押券
        self.tb_exchg=pd.read_excel(str(file_path_tb_exchg),skip_footer=1)     #交易所质押券
        self.tb_exchg_p=pd.read_excel(str(file_path_tb_exchg_p),skip_footer=1)   #交易所组合层标准券
        #限售股名单
        file_path_tb_equity = self.file_path_collateral / self.t0_str / '限售股名单.xlsx'
        self.tb_equity = pd.read_excel(str(file_path_tb_equity),sheet_name='限售股名单', engine='openpyxl')  # 银行间质押券
        #逆回购
        t=self.t0.strftime('%Y-%m-%d')  #字符串日期格式YYYY-MM-DD
        filename=''.join(['回购资产交易到期浏览表_[',t,']','.xls'])
        file_path_tb_repo=self.file_path_repo / self.t0_str / filename
        self.tb_repo=pd.read_excel(file_path_tb_repo,skiprows=3,skipfooter=3)
        #代码匹配表
        file_path_tb_coding=self.file_path_coding / '代码匹配表.xlsx'
        self.tb_coding=pd.read_excel(file_path_tb_coding, engine='openpyxl')
        #数据处理
        self.__data_processing()
        print('读取质押券数据完成！')

    def __data_processing(self):
        '''
        该函数用于对质押券数据做初步处理，以便于后续计算
        :return: None
        '''
        # 银行间质押券
        tb_ib = self.tb_ib.copy()
        tb_ib = tb_ib.loc[~(tb_ib['委托方向'] == '融券回购'), :]  # 删除融券回购
        tb_ib['质押结束日期'] = pd.to_datetime(tb_ib['质押结束日期'])  #日期格式转换为TimeStamp
        tb_ib['CONC'] = tb_ib['基金名称'] + tb_ib['证券名称']    #创建unique id
        tb_ib.loc[tb_ib['质押结束日期'] <= self.t0, 'CONC'] = ''  # 质押结束日期大于今天，创建unique key=基金名称+证券名称
        tb_ib.set_index('CONC', inplace=True)  # 设为key
        self.tb_ib = tb_ib.copy()
        # 交易所质押券
        tb_exchg = self.tb_exchg
        tb_exchg['CONC'] = tb_exchg['基金名称'] + tb_exchg['证券名称']   #创建unique id
        tb_exchg.loc[tb_exchg['已质押数量'] <= 0, 'CONC'] = ''  # 质押量大于0，创建unique key=基金名称+证券名称
        tb_exchg.set_index('CONC', inplace=True)  # 设为key
        self.tb_exchg = tb_exchg.copy()
        # 组合交易所标准券
        tb_exchg_p = self.tb_exchg_p
        tb_exchg_p['总可融资量'] = tb_exchg_p['基金名称'].apply(lambda x: tb_exchg_p.loc[tb_exchg_p['基金名称'] == x, '可用数量'].sum())  #按基金名称加总可用质押量
        tb_exchg_p.drop_duplicates('基金名称', inplace=True)  # 删除重复行
        tb_exchg_p['总标准券'] = tb_exchg_p['基金名称'].apply(lambda x: tb_exchg.loc[tb_exchg['基金名称'] == x, '已转标准券数量'].sum())  #按基金名称加总标准券数量
        tb_exchg_p['已质押比率'] = 1 - tb_exchg_p['总可融资量'] / tb_exchg_p['总标准券']
        tb_exchg_p['已质押比率'].fillna(0, inplace=True)
        tb_exchg_p.set_index('基金名称', inplace=True)
        self.tb_exchg_p = tb_exchg_p.copy()
        # 限售股名单
        tb_equity = self.tb_equity.copy()
        tb_equity['限售数量'] = tb_equity['实际中签数量'] * tb_equity['限售比例'] #创建unique id
        tb_equity['CONC'] = tb_equity['基金名称'] + tb_equity['证券名称']
        tb_equity.set_index('CONC', inplace=True)
        self.tb_equity = tb_equity.copy()
        # 逆回购表
        self.tb_repo['查询标识'] = self.tb_repo['业务类别']  #数据处理
        self.tb_repo.loc[self.tb_repo['查询标识'].isin(['融资回购', '融券回购']), '查询标识'] = np.nan   #数据处理
        self.tb_repo['查询标识'].fillna(method='ffill', inplace=True)  #数据处理
        tb_mapping = self.tb_coding[['产品名称', 'O32产品名称']]  #数据处理
        tb_mapping.set_index('产品名称', inplace=True)
        self.tb_repo['基金简称'] = self.tb_repo['查询标识'].map(tb_mapping['O32产品名称'])  #数据处理
        self.tb_repo['到期日期'].fillna(0, inplace=True)
        self.tb_repo['日期'] = pd.to_datetime(self.tb_repo['到期日期'])

    def calc_collaterals(self,fundname,assetname):
        '''
        用于计算资产的质押量/限售量（张数/股数/数量）
        :param type: string，证券类型，股/债
        :param fundname: string，基金名称
        :param assetname: string，证券简称
        :return: float，质押量（数量）
        '''
        #债券质押
        CONC=''.join([fundname,assetname])
        collateral_volume_ib=self.tb_ib.loc[self.tb_ib.index==CONC,'质押数量']
        collateral_ratio=self.tb_exchg_p.loc[self.tb_exchg_p.index==fundname,'已质押比率']
        collateral_volume_ib=0 if collateral_volume_ib.empty else collateral_volume_ib.sum(skipna=True)#如果为空，则为0
        collateral_ratio = 0 if collateral_ratio.empty else collateral_ratio.sum(skipna=True) #如果为空，则为0
        collateral_volume_exch= self.tb_exchg.loc[self.tb_exchg.index == CONC, '已质押数量']
        collateral_volume_exch = 0 if collateral_volume_exch.empty else collateral_volume_exch.iloc[0]  #如果为空，则为0
        #股票限售
        collateral_volume_equity=self.tb_equity.loc[self.tb_equity.index==CONC,'限售数量']
        collateral_volume_equity = 0 if collateral_volume_equity.empty else collateral_volume_equity.sum(skipna=True)  # 如果为空，则为0
        #汇总：债+股
        collateral_volume_total=collateral_volume_ib+collateral_volume_exch*collateral_ratio+collateral_volume_equity
        return collateral_volume_total

    def calc_reverse_repo(self,fundname,days):
        '''
        该函数用于计算组合层面的days内到期的逆回购规模
        :param fundname: string,基金名称
        :param days: int,单位：天
        :return: float,N日内逆回购到期量
        '''
        count=0
        tn=self.t0
        while count<days:
            tn=tn+pd.Timedelta(days=1)
            if tn.weekday()==5:
                tn = tn + pd.Timedelta(days=2)
            elif tn.weekday()==6:
                tn = tn + pd.Timedelta(days=1)
            count+=1
        reverse_repo_amt=self.tb_repo.loc[(self.tb_repo['基金简称']==fundname) & (self.tb_repo['日期']<=tn) & (self.tb_repo['日期']>self.t0),'融出资金'].sum()
        return reverse_repo_amt

    def __leverage_haircut(self):
        '''
        定义债券的抵押率
        :return:
        '''
        hc={'AAA+':0.95,
            'AAA':0.95,
            'AAA-':0.9,
            'AA+':0.85,
            'AA':0.8,
            'AA(2)':0.7,
            'AA-':0.5,
            'NR':0.95}
        return hc

    def calc_leverage_space(self,fundname,assetname,asset_amt,impRat):
        '''
        计算债券的可加杠杆率
        :param fundname:string,基金名称
        :param assetname:string,证券简称
        :param asset_amt:float,证券持仓数量
        :param impRat:string,中债隐含评级
        :return:float,数量（可加杠杆数量0
        '''
        try:
            haircut=self.__leverage_haircut()[impRat]
        except:
            haircut=0  #隐含评级小于AA-的质押率为0
        coll_amt=self.calc_collaterals(fundname,assetname)  #计算质押数量
        leverage_amt=(asset_amt-coll_amt)*haircut  #杠杆空间：（持仓数量-质押数量）*质押比例
        return leverage_amt

class liquidity_IRbond(model_Liqudity):
    def __init__(self,t):
        super(liquidity_IRbond, self).__init__()
        self.get_data(t)
        self.dict=self.__define_factors()

    def __define_factors(self):
        '''
        定义流动性因子
        :return:
        '''
        row_name=['1年','3年','5年','7年','10年','10年以上']
        col_name=['新券','次新券','老券']
        # 国债
        treasury=[5.84,2.76,0.65,4.59,4.05,0.64,4.15,3.10,0.93,9.35,2.21,0.55,14.39,5.13,0.31,1.13,0.4,0.4]
        treasury=np.reshape(treasury,(6,3))
        treasury=pd.DataFrame(treasury,index=row_name,columns=col_name)
        # 国开
        cdb=[31.95,18.65,3.48,69.99,59.27,1.76,65.61,45.75,2.28,21.78,13.33,7.62,240.78,186.10,4.52,0.63,0.63,0.63]
        cdb=np.reshape(cdb,(6,3))
        cdb=pd.DataFrame(cdb,index=row_name,columns=col_name)
        # 非国开
        non_cdb=[14.76,13.58,1.55,20.67,16.79,1.74,16.18,12.87,0.82,10.09,8.64,2.84,21.63,20.65,7.56,1.62,1.62,1.62]
        non_cdb=np.reshape(non_cdb,(6,3))
        non_cdb=pd.DataFrame(non_cdb,index=row_name,columns=col_name)

        factors_dict = {'国债': treasury,
                        '国开': cdb,
                        '农发': non_cdb,
                        '进出':non_cdb}
        return factors_dict

    def __reclassification_ttm(self,ttm):
        '''
        对剩余期限分类
        :param ttm: float,time to maturity
        :return: string, classified time to maturity
        '''
        def reclassify(ttm):
            if ttm <= 1.5:
                ttm_reclassified = '1年'
            elif ttm > 1.5 and ttm <= 3.5:
                ttm_reclassified = '3年'
            elif ttm > 3.5 and ttm <= 5.5:
                ttm_reclassified = '5年'
            elif ttm > 5.5 and ttm <= 7.5:
                ttm_reclassified = '7年'
            elif ttm > 7.5 and ttm <= 10:
                ttm_reclassified = '10年'
            elif ttm > 10:
                ttm_reclassified = '10年以上'
            return ttm_reclassified
        factor_ttm=reclassify(ttm)
        return factor_ttm

    def __reclassification_OnOffRun(self,tfi):
        def reclassify(tfi):
            if tfi<=0.75:
                tfi_reclassified='新券'
            elif tfi>0.75 and tfi<=1.5:
                tfi_reclassified = '次新券'
            elif tfi > 1.5:
                tfi_reclassified = '老券'
            return tfi_reclassified
        factor_tfi=reclassify(tfi)
        return factor_tfi

    def calc_liquidity(self,ir_type,ttm,tfi):
        '''
        计算利率债的流动性
        :param ir_type: string,利率债类型
        :param ttm: float,剩余期限
        :param tfi: float,距离发行日
        :return: float,流动性
        '''
        ttm=self.__reclassification_ttm(ttm)
        acitivity=self.__reclassification_OnOffRun(tfi)
        liq=self.dict[ir_type][acitivity][ttm]*1000000 #亿元单位转换为张数单位
        return liq

class liquidity_Creditbond(model_Liqudity):
    def __init__(self,t):
        super(liquidity_Creditbond, self).__init__()
        self.get_data(t)
        self.dict=self.__define_factors()

    def __reclassification_corpAttr(self,ca):
        '''
        对公司属性进行分类
        :param ca: string,company attribute
        :return: string,relcassified company attribute
        '''
        dict = {'中央国有企业': '央企',
                '地方国有企业': '国企',
                '国有企业': '国企',
                '公众企业': '上市企业',
                '民营企业': '民企',
                '其他企业': '民企',
                '集体企业': '民企',
                '外商独资企业': '外资',
                '中外合资企业': '外资',
                '外资企业': '外资'}
        factor_companyAttribute=dict[ca]
        return factor_companyAttribute

    def __reclassification_ttm(self,ttm):
        '''
        对剩余期限分类
        :param ttm: float,time to maturity
        :return: string, classified time to maturity
        '''
        def reclassify(ttm):
            if ttm <= 1.08:
                ttm_reclassified = '1年以下'
            elif ttm > 1.08 and ttm <= 2.5:
                ttm_reclassified = '1~2.5年'
            elif ttm > 2.5 and ttm <= 4.0:
                ttm_reclassified = '2.5~4年'
            elif ttm > 4.0 and ttm <= 6.0:
                ttm_reclassified = '4~6年'
            else:
                ttm_reclassified = '6年以上'
            return ttm_reclassified
        factor_ttm=reclassify(ttm)
        return factor_ttm

    def __define_factors(self):
        '''
        计算信用债流动性所用的因子及因子值
        :return: Dict,因子字典
        '''
        # implicit rating
        impRat = {'AAA+': 0.575,'AAA': 0.391,'AAA-': 0.219,'AA+': 0.134,'AA': 0.075,'AA(2)': 0.06,'AA-': 0.05, \
                  'A+': 0.04,'A': 0.036, 'A-': 0.034,'BBB+': 0.03,'BBB': 0.029,'BB': 0.025,'B': 0.02, 'CCC': 0.0001, \
                  'CC': 0.0001,'C':0.0001,'D': 0.0,'NR':0.05}
        # remaining maturity
        ttm = {'1年以下': 1.0, '1~2.5年': 0.468, '2.5~4年': 0.435, '4~6年': 0.236,'6年以上': 0.133}
        # corporation attributes
        corpAttr = {'央企': 1.36, '上市企业': 1.24, '国企': 1.2,'外资': 0.75,'民企': 0.72}
        # issuing method
        issMethod = {'公募': 1.0,'私募': 0.1}
        # industry type
        indType = {'产业债': 1.0, '城投债': 0.9}
        # payment sequence
        pmtSeq = {'普通': 1.0,'次级': 6.5}
        # clause content
        clause = {'普通': 1.0, '永续': 0.5}
        factors_dict = {'隐含评级': impRat,
                        '剩余期限分类':ttm,
                        '公司属性': corpAttr,
                        '发行方式': issMethod,
                        '行业类别': indType,
                        '偿还顺序': pmtSeq,
                        '含权条款': clause}
        return factors_dict

    def calc_liquidity(self,impRat,ttm,corpAttr,issMethod,indType,pmtSeq,clause):
        '''
        用于计算信用债的流动性
        :param impRat: string,中债隐含评级
        :param ttm: float,剩余期限
        :param corpAttr: string,公司属性
        :param issMethod: string,发行方式
        :param indType: string，申万一级行业
        :param pmtSeq: string,偿付顺序
        :param clause: string,含权条款
        :return: float,单日变现规模
        '''
        val_impRat=self.dict['隐含评级'][impRat]
        ttm=self.__reclassification_ttm(ttm)
        val_ttm=self.dict['剩余期限分类'][ttm]
        corpAttr=self.__reclassification_corpAttr(corpAttr)
        val_corpAttr=self.dict['公司属性'][corpAttr]
        val_issMethod=self.dict['发行方式'][issMethod]
        val_indType=self.dict['行业类别'][indType]
        val_pmtSeq=self.dict['偿还顺序'][pmtSeq]
        val_clause=self.dict['含权条款'][clause]
        liq=val_impRat*val_ttm*val_corpAttr*val_issMethod*val_indType*val_pmtSeq*val_clause
        liq=liq*1000000 #亿元单位转换为张数单位
        return liq

class liquidity_NCD(model_Liqudity):
    def __init__(self,t):
        super(liquidity_NCD, self).__init__()
        self.get_data(t)
        self.dict=self.__define_factors()

    def __define_factors(self):
        '''
        定义流动性因子
        :return: None
        '''
        # implicit rating
        impRat = {'AAA': 3.0, 'AAA-': 2.25, 'AA+': 0.90, 'AA': 0.45, 'AA-': 0.21, \
                  'A+': 0.10, 'A': 0.05, 'A-': 0.04, 'BBB+': 0.03, 'BBB': 0.0, 'BB': 0.01, 'B': 0.001, 'CCC': 0.0001, \
                  'CC': 0.0001,'C':0.0001, 'D': 0.0}
        factors_dict = {'隐含评级': impRat}
        return factors_dict

    def calc_liquidity(self,impRat):
        '''
        计算同业存单的流动性
        :param impRat: string,中债隐含评级
        :return: float,流动性
        '''
        liq=self.dict['隐含评级'][impRat]
        liq=liq*1000000 #亿元单位转换为张数单位
        return liq

class liquidity_ABS(model_Liqudity):
    def __init__(self,t):
        super(liquidity_ABS, self).__init__()
        self.get_data(t)
        self.dict=self.__define_factors()

    def __define_factors(self):
        '''
        定义流动性因子
        :return: None
        '''
        # implicit rating
        impRat = {'AAA+': 0.575,'AAA': 0.391,'AAA-': 0.219,'AA+': 0.134,'AA': 0.075,'AA(2)': 0.06,'AA-': 0.05, \
                  'A+': 0.04,'A': 0.036, 'A-': 0.034,'BBB+': 0.03,'BBB': 0.029,'BB': 0.025,'B': 0.02, 'CCC': 0.02, \
                  'CC': 0.01,'C':0.0001,'D': 0.0,'NR':0.05}
        # remaining maturity
        ttm = {'1年以下': 1.0, '1~2.5年': 0.468, '2.5~4年': 0.435, '4~6年': 0.236,'6年以上': 0.133}
        # issuing method
        issMethod = {'公募': 0.1,'私募': 0.1}

        factors_dict = {'隐含评级': impRat,
                        '剩余期限分类': ttm,
                        '发行方式':issMethod}
        return factors_dict

    def __reclassification_ttm(self,ttm):
        '''
        对剩余期限分类
        :param ttm: float,time to maturity
        :return: string, classified time to maturity
        '''
        def reclassify(ttm):
            if ttm <= 1.08:
                ttm_reclassified = '1年以下'
            elif ttm > 1.08 and ttm <= 2.5:
                ttm_reclassified = '1~2.5年'
            elif ttm > 2.5 and ttm <= 4.0:
                ttm_reclassified = '2.5~4年'
            elif ttm > 4.0 and ttm <= 6.0:
                ttm_reclassified = '4~6年'
            else:
                ttm_reclassified = '6年以上'
            return ttm_reclassified
        factor_ttm=reclassify(ttm)
        return factor_ttm

    def calc_liquidity(self,impRat,ttm,issMethod):
        '''
        计算ABS的流动性
        :param impRat: string,中债隐含评级
        :param ttm: float,剩余期限
        :param issMethod: string,发行方式
        :return: float,流动性
        '''
        val_impRat=self.dict['隐含评级'][impRat]
        ttm=self.__reclassification_ttm(ttm)
        val_ttm=self.dict['剩余期限分类'][ttm]
        val_issMethod=self.dict['发行方式'][issMethod]
        liq=val_impRat*val_ttm*val_issMethod
        liq=liq*1000000 #亿元单位转换为张数单位
        return liq

class liquidity_Convertible(model_Liqudity):
    def __init__(self,t):
        super(liquidity_Convertible, self).__init__()
        self.get_data(t)
        self.wss_fields, self.wss_args = self.__define_factors()

    def __define_factors(self):
        '''
        定义计算ABS流动性所用的因子
        :return:Wind Object
        '''
        startDate=self.t_prev_str
        endDate=self.t0_str
        wss_fields="".join(["vol_per"])
        wss_args="".join(["unit=1;startDate=",startDate,";endDate=",endDate])
        return wss_fields,wss_args

    def calc_liquidity(self,windcode):
        '''
        计算ABS流动性
        :param windcode:
        :return:
        '''
        self.Wind_Connect()
        wss=w.wss(windcode, self.wss_fields,self.wss_args)
        try:
            liq = wss.Data[0][0]*0.05/20
        except:
            liq=0
        return liq

class liquidity_Equity(model_Liqudity):
    def __init__(self,t):
        super(liquidity_Equity, self).__init__()
        self.get_data(t)
        self.wss_fields,self.wss_args=self.__define_factors()

    def __define_factors(self):
        '''
        定义计算股票流动性所用的因子
        :return:Wind Object
        '''
        startDate=self.t_prev_str
        endDate=self.t0_str
        wss_fields="".join(["vol_per"])
        wss_args="".join(["unit=1;startDate=",startDate,";endDate=",endDate])
        return wss_fields,wss_args

    def calc_liquidity(self,windcode):
        '''
        计算股票流动性
        :param windcode:string,万德代码
        :return:float,流动性
        '''
        self.Wind_Connect()
        wss=w.wss(windcode, self.wss_fields,self.wss_args)
        try:
            liq = wss.Data[0][0] * 0.05 / 20
        except:
            liq = 0
        return liq

class Calc_AssetLiquidity(object):
    def __init__(self,t):
        #建立对象
        self.irbond=liquidity_IRbond(t)
        self.creditbond=liquidity_Creditbond(t)
        self.ncd=liquidity_NCD(t)
        self.convertible=liquidity_Convertible(t)
        self.abs=liquidity_ABS(t)
        self.equity=liquidity_Equity(t)
        self.collateral=liquidity_Collateral(t)

    def calc_liquidity(self,securitytype,args):
        '''
        计算流动性
        :param securitytype:string,证券类型：股票/利率债/信用债/同业存单/ABS/可转债
        :param args:dict,根据不同证券类型，输入不同参数
            {windcode:string,万德代码
            irbond_type: string,利率债类型：国债/国开/农发/口行
            impRat: string,中债隐含评级
            ttm: float,剩余期限
            tfi: float,距离发行日
            corpAttr: string，公司属性
            issMethod: string,发行方式：公募/私募
            indType: string,行业：产业债/城投债
            pmtSeq:string,偿付顺序：普通/次级
            clause:string,含权条款：普通/永续}
        :return:float,流动性（张数）
        '''
        #根据资产类型筛选所需要变量
        if securitytype=='股票':
            windcode=args['windcode']
            liq = self.equity.calc_liquidity(windcode)
        elif securitytype=='利率债':
            irbondtype=args['irbond_type']
            ttm=args['ttm']
            tfi=args['tfi']
            liq=self.irbond.calc_liquidity(irbondtype,ttm,tfi)
        elif securitytype=='信用债':
            impRat=args['impRat']
            ttm = args['ttm']
            corpAttr=args['corpAttr']
            issMethod=args['issMethod']
            indType=args['indType']
            pmtSeq=args['pmtSeq']
            clause=args['clause']
            liq=self.creditbond.calc_liquidity(impRat,ttm,corpAttr,issMethod,indType,pmtSeq,clause)
        elif securitytype=='同业存单':
            impRat=args['impRat']
            liq=self.ncd.calc_liquidity(impRat)
        elif securitytype=='ABS':
            impRat=args['impRat']
            ttm=args['ttm']
            issMethod=args['issMethod']
            liq=self.abs.calc_liquidity(impRat,ttm,issMethod)
        elif securitytype=='可转债':
            windcode=args['windcode']
            liq=self.convertible.calc_liquidity(windcode)
        else:
            print("证券类型输入错误")
            liq = np.nan
        return liq

    def calc_collateral(self,fn,an):
        '''
        计算质押券量（张数）
        :param fn: string,基金名称
        :param an: string,证券简称
        :return: float,质押券量（张数）
        '''
        coll_amt=self.collateral.calc_collaterals(fn,an)  #质押券量
        return coll_amt

    def calc_liquidity_1d(self,securitytype,args,fn,an,asset_amt):
        '''
        1日变现规模减去质押券
        :param securitytype: string,证券类型：股票/利率债/信用债/同业存单/ABS/可转债
        :param args: dict,参数表
        :param fn: string,基金简称
        :param an: string,证券简称
        :param asset_amt: float,证券数量（张数）
        :return:float,1日可变现规模
        '''
        coll_amt = self.calc_collateral(fn, an)  #逆回购量
        aval_amt = asset_amt - coll_amt  # 可用量
        liq_amt=self.calc_liquidity(securitytype,args)  #资产单日可变现量
        sellable_amt=max(min(liq_amt,aval_amt),0)  #持仓资产单日可变现量
        return sellable_amt

    def calc_liquidity_5d(self,securitytype,args,fn,an,asset_amt):
        '''
        5日变现规模减去质押券
        :param securitytype: string,证券类型：股票/利率债/信用债/同业存单/ABS/可转债
        :param args: dict,参数表
        :param fn: string,基金简称
        :param an: string,证券简称
        :param asset_amt: float,证券数量（张数）
        :return:float,5日可变现规模
        '''
        coll_amt = self.calc_collateral(fn, an)  #逆回购量
        aval_amt = asset_amt - coll_amt  # 可用量
        liq_amt=self.calc_liquidity(securitytype,args)  #资产单日可变现量
        sellable_amt=max(min(liq_amt*5,aval_amt),0)  #持仓资产单日可变现量
        return sellable_amt

    def calc_reverse_repo(self,fundname,days):
        '''
        该函数用于计算组合层面逆回购days日内的到期量
        :param fundname:string,基金名称
        :param days:int,天
        :return:float,逆回购到期量
        '''
        amt=self.collateral.calc_reverse_repo(fundname,days)  #逆回购量
        return amt


'''
#-------------使用方法----------------
t='2020-02-28'   #分析日
cal=Calc_AssetLiquidity(t) #建立对象

#计算个券流动性
tb=pd.read_excel(r'Y:\27. RiskQuant\Data\Collaterals\20200228\测试数据.xlsx',sheet_name='债券持仓') #估值表（需替换,此处仅为样例）
#在估值表内按行循环计算
tb_out=['基金名称','证券简称','持仓数量','资产单日变现规模','资产质押量','单日可变现规模','五日可变现规模']
tb_out=pd.DataFrame(columns=tb_out)
for i in range(len(tb)):
    security_type=tb['债券内部分类二'][i]
    fn=tb['基金名称'][i]
    an=tb['证券名称'][i]
    asset_amt=tb['持仓'][i]
    print('(%i/%i)，%s，%s，%s' % (i,len(tb),security_type,an,fn))
    if tb['债券内部分类二'][i]=='铁道债':
        security_type='信用债'
    args={'windcode': tb['万德代码'][i],
          'irbond_type': tb['流动性风险：利率债：券种'][i],
          'impRat': tb['隐含评级'][i],
          'ttm':tb['剩余期限'][i],
          'tfi': tb['流动性风险：利率债：距离起息日'][i],
          'corpAttr': tb['公司属性'][i],
          'issMethod': tb['发行方式'][i],
          'indType': tb['是否城投'][i],
          'pmtSeq': tb['是否次级债'][i],
          'clause': tb['是否永续'][i]}
    #----------------------具体要使用的函数-----------------------------
    liq_asset=cal.calc_liquidity(security_type,args)
    liq_coll=cal.calc_collateral(fn,an)
    liq_sellable_1d=cal.calc_liquidity_1d(security_type,args,fn,an,asset_amt)
    liq_sellable_5d=cal.calc_liquidity_5d(security_type,args,fn,an,asset_amt)
    # -------------------------------------------------------------------
    tb_appd={'基金名称':fn,
             '证券简称':an,
             '持仓数量':asset_amt,
             '资产单日变现规模':liq_asset,
             '资产质押量':liq_coll,
             '单日可变现规模':liq_sellable_1d,
             '五日可变现规模':liq_sellable_5d}
    tb_out=tb_out.append(tb_appd,ignore_index=True)
tb_out.to_excel(r'Y:\27. RiskQuant\Data\Collaterals\20200228\测试结果（个券）.xlsx') #输出

#计算个股流动性
tb=pd.read_excel(r'Y:\27. RiskQuant\Data\Collaterals\20200228\测试数据.xlsx',sheet_name='股票持仓') #估值表（需替换,此处仅为样例）
#在估值表内按行循环计算
tb_out=['基金名称','证券简称','持仓数量','资产单日变现规模','资产质押量','单日可变现规模','五日可变现规模']
tb_out=pd.DataFrame(columns=tb_out)
for i in range(len(tb)):
    security_type='股票'
    fn=tb['基金名称'][i]
    an=tb['证券名称'][i]
    asset_amt=tb['持仓'][i]
    print('(%i/%i)，%s，%s，%s' % (i,len(tb),security_type,an,fn))
    args={'windcode': tb['万德代码'][i]}
    #----------------------具体要使用的函数-----------------------------
    liq_asset=cal.calc_liquidity(security_type,args)
    liq_coll=cal.calc_collateral(fn,an)
    liq_sellable_1d=cal.calc_liquidity_1d(security_type,args,fn,an,asset_amt)
    liq_sellable_5d=cal.calc_liquidity_5d(security_type,args,fn,an,asset_amt)
    # -------------------------------------------------------------------
    tb_appd={'基金名称':fn,
             '证券简称':an,
             '持仓数量':asset_amt,
             '资产单日变现规模':liq_asset,
             '资产质押量':liq_coll,
             '单日可变现规模':liq_sellable_1d,
             '五日可变现规模':liq_sellable_5d}
    tb_out=tb_out.append(tb_appd,ignore_index=True)
tb_out.to_excel(r'Y:\27. RiskQuant\Data\Collaterals\20200228\测试结果（个股）.xlsx') #输出

#计算组合层面逆回购到期量
tb=pd.read_excel(r'Y:\1. 基础数据\代码匹配表.xlsx') #估值表（需替换,此处仅为样例）
tb_out=['基金名称','1日逆回购到期量','5日逆回购到期量']
tb_out=pd.DataFrame(columns=tb_out)
fundlist=tb['O32产品名称'].drop_duplicates().to_list()
for fn in fundlist:
    print(fn)
    #----------------------具体要使用的函数-----------------------------
    repo_1day_amt=cal.calc_reverse_repo(fn,1)
    repo_5day_amt = cal.calc_reverse_repo(fn,5)
    # -------------------------------------------------------------------
    tb_appd={'基金名称':fn,
             '1日逆回购到期量':repo_1day_amt,
             '5日逆回购到期量':repo_5day_amt}
    tb_out=tb_out.append(tb_appd,ignore_index=True)
tb_out.to_excel(r'Y:\27. RiskQuant\Data\Collaterals\20200228\测试结果（逆回购）.xlsx') #输出
'''