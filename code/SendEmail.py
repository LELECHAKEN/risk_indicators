'''
Description: 
Author: Wangp
Date: 2021-05-18 11:02:24
LastEditTime: 2021-08-30 11:02:44
LastEditors: Wangp
'''
import os
import win32com.client as win32
import numpy as np
import pandas as pd


def checkProds(file_path):
    files = os.listdir(file_path)
    if len(files) == 1:
        if files[0].split('.')[0][:5] == '风险管理部':
            return 0
        else:
            return len(files)
    else:
        return len(files) - 1


def send_email(receiver,c_receiver,title,body,att):
    outlook = win32.Dispatch('outlook.application')    #启动outlook进程
    mail = outlook.CreateItem(0)       #新建一封邮件
    mail.GetInspector
    mail.To = receiver                     #将收件人设为变量receiver
    mail.CC = c_receiver                   #将抄送人设为变量c_receiver
    mail.Subject = title                   #将邮件主题设为变量title
    mail.HTMLBody = body + mail.HTMLBody   #添加邮件签名

    if type(att) == list:                  #将邮件附件设为变量att,att必须为绝对路径
        for i in range(len(att)):
            mail.Attachments.Add(att[i])
    else:
        mail.Attachments.Add(att)

    if receiver in ['yangy@maxwealthfund.com', 'yangfy@maxwealthfund.com']:
        mail.Display(True)                     #True则脚本停止，直到窗口关闭；False则继续下一个命令
    
    mail.Send()


baseDate = '20220617'
file_path = r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\日频数据\%s\\'%baseDate

send_df = pd.read_excel(r'\\shaoafile01\RiskManagement\27. RiskQuant\Data\RiskIndicators\日频数据\基金经理清单.xlsx', engine='openpyxl')
send_dict = send_df.set_index(['基金经理']).to_dict(orient='index')
send_list = send_df.loc[send_df['分类'] == '固收', '基金经理'].tolist()
fm_paths = [x for x in os.listdir(file_path) if '.' not in x]
fms = list(set(fm_paths) & set(send_list))


# 邮件正文必须用html语言写才能自带邮件签名
mail_body = '''
<!DOCTYPE html><html><head><meta charset="utf-8" /><title>html<br>组合风险指标正文</title></head><body bgcolor="white">
您好，<br/><br/>附件是%s的组合风险指标，请您查收。<br/><br/>谢谢！
</body></html>'''%baseDate

for fm in fms:
    # check该基金经理是否有管理产品
    cnt = checkProds(file_path + fm)
    if cnt == 0:
        continue
    else:
        folder_path = file_path + fm + r'\\'
        receiver = send_dict[fm]['邮箱']
        c_receiver = send_dict[fm]['抄送']
        title = '组合风险指标%s'%baseDate
        body = mail_body
        att = [folder_path + x for x in os.listdir(folder_path)]

        send_email(receiver, c_receiver, title, body, att)
        print(fm, 'sent.')