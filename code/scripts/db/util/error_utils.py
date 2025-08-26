#!/bin/python
# -*- coding: utf-8 -*-
# __author__ = sunsn
# __datetime__ = 2021/1/20 17:21
"""
模块注释
"""


class ParamError(Exception):
    pass


class DBError(Exception):
    pass


class MysqlError(Exception):
    pass


class OracleError(Exception):
    pass


class PostgresError(Exception):
    pass


class SQLError(Exception):
    pass


class RedisError(Exception):
    pass


class MongoError(Exception):
    pass


class NotSupportError(Exception):
    pass


class DoesNotExist(Exception):
    pass
