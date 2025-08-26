# -*- coding: utf-8 -*-
'''
# @Desc    : 日志工具：在主程序目录logs文件夹下记录日志，
# @Author  : zhouyy
# @Time    : 2021/5/10
'''
import logging
from logging.handlers import RotatingFileHandler # 按文件大小滚动备份
import colorlog  # 控制台日志输入颜色
import time
import datetime
import os
from .. import settings
log_path = settings.log_path

class Log:
    log_colors_config = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }

    if not os.path.exists(log_path):
        os.mkdir(log_path)  # 如果不存在这个logs文件夹，就自动创建一个
    logName = os.path.join(log_path, '%s.log' % time.strftime('%Y-%m-%d'))  # 文件的命名

    def __init__(self, logName=logName):
        self.logName = logName
        self.logPath = os.path.dirname(logName)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(levelname)s: %(message)s',
            log_colors=Log.log_colors_config)  # 日志输出格式
        self.handle_logs()

    def get_file_sorted(self, file_path):
        """最后修改时间顺序升序排列 os.path.getmtime()->获取文件最后修改时间"""
        dir_list = os.listdir(file_path)
        if not dir_list:
            return
        else:
            dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
            return dir_list

    def TimeStampToTime(self, timestamp):
        """格式化时间"""
        timeStruct = time.localtime(timestamp)
        return str(time.strftime('%Y-%m-%d', timeStruct))

    def handle_logs(self):
        """处理日志过期天数和文件数量"""
        dir_list = ['']  # 要删除文件的目录名
        for dir in dir_list:
            dirPath = self.logPath + '\\' + dir  # 拼接删除目录完整路径
            file_list = self.get_file_sorted(dirPath)  # 返回按修改时间排序的文件list
            if file_list:  # 目录下没有日志文件
                for i in file_list:
                    file_path = os.path.join(dirPath, i)  # 拼接文件的完整路径
                    t_list = self.TimeStampToTime(os.path.getctime(file_path)).split('-')
                    now_list = self.TimeStampToTime(time.time()).split('-')
                    t = datetime.datetime(int(t_list[0]), int(t_list[1]),
                                          int(t_list[2]))  # 将时间转换成datetime.datetime 类型
                    now = datetime.datetime(int(now_list[0]), int(now_list[1]), int(now_list[2]))
                    if (now - t).days > 360:  # 创建时间大于 天的文件删除
                        self.delete_logs(file_path)
                if len(file_list) > 50:  # 限制目录下记录文件数量
                    file_list = file_list[0:-20]
                    for i in file_list:
                        file_path = os.path.join(dirPath, i)
                        print(file_path)
                        self.delete_logs(file_path)

    def delete_logs(self, file_path):
        try:
            os.remove(file_path)
        except PermissionError as e:
            Log().warning('删除日志文件失败：{}'.format(e))

    def __console(self, level, message):
        # 创建一个FileHandler，用于写到本地
        fh = RotatingFileHandler(filename=self.logName, mode='a', maxBytes=1024 * 1024 * 5, backupCount=5,
                                 encoding='utf-8')  # 使用RotatingFileHandler类，滚动备份日志
        fh.setLevel(logging.INFO)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

        # 创建一个StreamHandler,用于输出到控制台
        ch = colorlog.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        # 这两行代码是为了避免日志输出重复问题
        self.logger.removeHandler(ch)
        self.logger.removeHandler(fh)
        fh.close()  # 关闭打开的文件

    def debug(self, message):
        self.__console('debug', message)

    def info(self, message):
        self.__console('info', message)

    def warning(self, message):
        self.__console('warning', message)

    def error(self, message):
        self.__console('error', message)

logger = Log()

if __name__ == "__main__":
    logger = Log()
    logger.debug("---测试开始----")
    logger.info("操作步骤")
    logger.warning("----测试结束----")
    logger.error("----测试错误----")