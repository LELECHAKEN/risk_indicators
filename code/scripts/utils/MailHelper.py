#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time   : 2024/3/22 14:14
# !@File   : MailHelper.py
# !@Author : shiyue


import os.path
import smtplib

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from scripts.settings import config


class MailHelper:
    __mail_server = None
    __msg = None
    __sender_address = None
    __receiver_address = None

    def __init__(self):
        self.__mail_host = config['email']['addr']
        self.__mail_port = config['email']['port']
        self.__postfix = config['email']['postfix']
        self.__mail_sender = config['email']['sender']
        if not self.__mail_sender.__contains__('@'):
            self.__mail_sender = self.__mail_sender + self.__postfix
        self.__mail_password = config['email']['password']
        self.__mail_server = smtplib.SMTP(self.__mail_host, self.__mail_port)  # 初始化邮箱服务器
        self.__msg = MIMEMultipart()  # 初始化邮件内容主体对象

    def mail_sender_register(self):
        if self.__mail_password:
            print(self.__mail_password)
            print(self.__mail_server.ehlo())
            print(self.__mail_server.starttls())
            print(self.__mail_server.helo(self.__mail_server.helo))
            print((self.__mail_sender, self.__mail_password))
            self.__mail_server.login(self.__mail_sender, self.__mail_password)  # 发送人邮箱 授权码

    def msg_sender_address(self, mail_sender=None):
        if not mail_sender:
            mail_sender = self.__mail_sender
        if not mail_sender.__contains__('@'):
            mail_sender = mail_sender + self.__postfix
        self.__sender_address = mail_sender
        self.__msg["From"] = self.__mail_sender

    def msg_receiver_address(self, receiver_address):
        # self.__receiver_address = receiver_address.split(",")
        receiver_address_list = [r if r.__contains__('@') else r + self.__postfix for r in receiver_address.split(",")]
        self.__receiver_address = receiver_address_list
        self.__msg["To"] = ','.join(receiver_address_list)

    def msg_cc_address(self, cc_address):
        # self.__receiver_address = self.__receiver_address + cc_address.split(",")
        cc_address_list = [r if r.__contains__('@') else r + self.__postfix for r in cc_address.split(",")]
        self.__receiver_address = self.__receiver_address + cc_address_list
        self.__msg["Cc"] = ','.join(cc_address_list)

    def msg_bcc_address(self, bcc_address):
        # self.__receiver_address = self.__receiver_address + bcc_address.split(",")
        bcc_address_list = [r if r.__contains__('@') else r + self.__postfix for r in bcc_address.split(",")]
        self.__receiver_address = self.__receiver_address + bcc_address_list
        self.__msg["Bcc"] = ','.join(bcc_address_list)

    def msg_subject(self, subject=None):
        if not subject:
            subject = config['message']['subject']
        self.__msg["Subject"] = subject

    def msg_content(self, content=None):
        if not content:
            content = config['message']['content']
        self.__msg.attach(MIMEText(content, "html", "utf-8"))

    def msg_attach(self, file_paths):
        file_paths = file_paths.split(",")
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            file = MIMEApplication(open(file_path, 'rb').read())
            file.add_header('Content-Disposition', 'attachment', filename=file_name)  # 设置附件信息
            self.__msg.attach(file)

    def send(self):
        return self.__mail_server.sendmail(self.__sender_address, self.__receiver_address, self.__msg.as_string())

    def quit(self):
        self.__mail_server.quit()

    def format_table_html(self, df):
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
              ''' % str(idx) + '<td width="400" align="center">%s</td>' * df.shape[1] % (
                tuple(str(x) if type(x) != str else x for x in row_i)) + '</tr>'
        return table_html

    def format_table_style(self, table_name, df, table_html):
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
              <td width="90"><strong>序号</strong></td>''' % table_name + '<td width="400" align="center"><strong>%s</strong></td>' * \
                     df.shape[1] % (tuple(x for x in df.columns)) + '</tr>' + table_html + '''
            </table>
            </div>
            </div>'''
        return html_style

    def format_body(self, df_list, name_list):
        html_list = []
        for df, name in zip(df_list, name_list):
            table_html_temp = self.format_table_html(df)
            table_style_temp = self.format_table_style(name, df, table_html_temp)
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


if __name__ == '__main__':
    try:
        mail = MailHelper()  # 邮箱服务器类型
        # mail.mail_sender_register()
        mail.msg_sender_address()  # 发件人姓名
        mail.msg_subject()  # 邮件标题
        content = mail.format_body([], [])
        mail.msg_content(content)  # 邮件内容
        mail.msg_receiver_address("shiy02")  # 收件人账号，为多个时英文逗号隔开
        # mail.msg_cc_address("huay,fugh")  # 收件人账号，为多个时英文逗号隔开
        # mail.msg_bcc_address("yinjz")  # 收件人账号，为多个时英文逗号隔开
        # 邮件附件，传入附件路径，路径多个时英文逗号隔离
        # mail.msg_attach(
        #     r"C:/Users/pc/git/ServiceEmail/附件/ReferenceCard.pdf,C:/Users/pc/git/ServiceEmail/附件/永赢测试系统.docx")
        # mail.msg_attach(
        #     r"C:/Users/pc/git/ServiceEmail/附件/ReferenceCard.pdf")

        print(mail.send())
        mail.quit()
    except Exception as e:
        print(e)