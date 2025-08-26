'''
Description:
Author: Wangp
Date: 2020-09-22 16:30:31
LastEditTime: 2020-11-09 11:23:26
LastEditors: Wangp
'''
import numpy as np
import pandas as pd

#%%
def calcKeyYearRatio(key_year, key_dura):
    if key_year[0] < 0:
        idx = [i for i in range(len(key_year)) if key_year[i] < 0]
        key_year = key_year[idx[-1]+1:]
    if key_year[-1] == np.inf:
        key_year = key_year[:-1]
        key_dura = key_dura[:-1]
    
    if len(key_year) != len(key_dura):
        print('Length mismatch.')
        return np.nan
    
    key_year = np.array(key_year)
    key_dura = np.array(key_dura)

    ratios = np.zeros(shape=key_dura.shape)
    idx_list = [i for i in range(len(key_dura)) if np.isnan(key_dura)[i] == False]
    for i in range(key_dura.shape[0]):
        if i in idx_list:
            ratios[i] = key_dura[i] / key_year[i]            
    if ratios.sum() < 1:
        ratios[0] = 1 - ratios[1:].sum()
   
    return ratios

def calcKeyYearRatio_n(key_year, key_dura):
    if np.array(key_dura).ndim > 1:
        ratios = np.zeros(shape=key_dura.shape)
        for i in range(np.array(key_dura).shape[0]):
            ratios[i] = calcKeyYearRatio(key_year, np.array(key_dura)[i])
    else:
        ratios = calcKeyYearRatio(key_year, key_dura)
    
    return ratios


#%%
# key_dura = [0, 0.48, np.nan]
# key_year = [0, 1, 10]
# ratios = calcKeyYearRatio(key_year, key_dura)
# ratios

#%%
# data = pd.DataFrame([[0, 0.1, 0.5], [0, 0.5, np.nan]], columns=['0Y', '1Y', '3Y'])
# key_dura = data.loc[:, '0Y':'3Y'].values
# key_year = [0, 1, 3]
# ratios = calcKeyYearRatio_n(key_year, key_dura)
# ratios