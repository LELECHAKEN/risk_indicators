# @Time : 2021/12/16 10:37 
# @Author : for wangp
# @File : send_email_standard.py 
# @Software: PyCharm
import os
import win32com.client as win32

import datetime
import numpy as np
import pandas as pd

def convert_table_content(df):
    if df.shape[0] > 0:
        for col in df.columns:
            if type(df[col].iloc[0]) == pd.Timestamp:
                df[col] = [x.strftime('%Y-%m-%d') for x in df[col]]
            if type(df[col].iloc[0]) == np.float64:
                df[col] = [f'{round(x*100,2)}%' for x in df[col]]
    else:
        pass
    return df

def format_table_html(df):
    '''
    按照html格式拼接表格各行列的内容

    :param df: DataFrame
    :return: html
    '''
    table_html = ''
    for idx, row_i in df.iterrows():
        table_html += '''
        <tr>
          <td width="10" align="center">%s</td>
          '''%str(idx) + '<td width="400" align="center">%s</td>'*df.shape[1]%(tuple(str(x) if type(x) != str else x for x in row_i)) + '</tr>'
    return table_html

def format_table_style(table_name, df, table_html):
    '''
    添加html的表格样式，如标题加粗、列间距等

    :param table_name: 展示在邮件正文中的表格名称
    :param df: DataFrame，需展示在正文中的表格数据
    :param table_html: 已拼接完毕的html
    :return: html
    '''
    html_style = '''
        <p><strong>%s:</strong></p>
        <div id="content">
        <table border="1" cellspacing="0" cellpadding="0">
        <tr>
          <td width="90"><strong>序号</strong></td>'''%table_name + '<td width="400" align="center"><strong>%s</strong></td>'*df.shape[1]%(tuple(x for x in df.columns)) + '</tr>'+table_html+'''
        </table>
        </div>
        </div>'''
    return html_style

def format_body(df_list, name_list):
    html_list = []
    for df, name in zip(df_list, name_list):
        table_html_temp = format_table_html(df)
        table_style_temp = format_table_style(name, df, table_html_temp)
        html_list.append(table_style_temp)

    html = '''
        <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <body>
        <div id="container">''' + ''.join(x for x in html_list) + '''
        </div>
        </body>
        </html>'''
    return html

def send_email(receiver,c_receiver,title,body,att=''):
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
    elif att != '':
        mail.Attachments.Add(att)
    else:
        pass

    mail.Send()