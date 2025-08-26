# @Time : 2022/6/22 14:00 
# @Author : for wangp
# @File : data_transfer.py 
# @Software: PyCharm
# @Description: 将公盘的组合风险指标同步至云文档
import os
import shutil

t = '2023-02-08'
path_share = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\日频数据'
path_cloud = r'E:\ShareCache\SHAShare\MWFiles\风险管理部 - 组合风险指标'
folders = os.listdir(path_cloud)

for folder in folders:
    manager_i = os.path.join(path_share, t.replace('-', ''), folder)
    manager_i_cloud = os.path.join(path_cloud, folder)
    files = os.listdir(manager_i)
    for file in files:
        file_share = os.path.join(manager_i, file)
        shutil.copy(file_share, manager_i_cloud)
print('==='*3, 'done.', '==='*3)