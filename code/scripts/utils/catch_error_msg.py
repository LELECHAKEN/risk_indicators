# @Time : 2021/11/4 14:54 
# @Author : for wangp
# @File : catch_error_msg.py 
# @Software: PyCharm
from tkinter import messagebox

def show_catch_message(title, message):
    status = messagebox.askyesno(title=title, message=message)
    if not status:
        exit()
    else:
        pass