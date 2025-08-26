#!/bin/python
# -*- coding: utf-8 -*-
# __author__ = sunsn
# __datetime__ = 2021/1/20 16:46
"""
Oracle数据库封装
"""

import os
import traceback
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, types
from sqlalchemy.engine import Engine

from .base import CRUD
from ..util.error_utils import DBError, ParamError

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.AL32UTF8'


class OracleDB(CRUD):

    def __init__(self, db_url: str, pool_size: Optional[int] = 5):

        if "oracle" not in db_url:
            raise ParamError("请填写正确的 Oracle 地址, 例如: oracle+cx_oracle://username:password@tnsname")

        if not isinstance(pool_size, int) or pool_size < 1:
            raise ParamError("pool_size 必须大于 0 的整数")

        self.pool_size = pool_size
        super(OracleDB, self).__init__(db_url)

    def _create_engine(self, dsn: str) -> Engine:

        try:
            engine = create_engine(
                dsn,
                pool_size=self.pool_size,  # 连接池大小
                max_overflow=1,  # 超过连接池大小外最多创建的连接
                pool_timeout=30,  # 池中没有可用线程在队列最多等待的时间，否则报错
                pool_recycle=600,  # 多久之后对线程池中的线程进行一次连接的回收（重置）
                pool_pre_ping=True,
                echo=False,
                max_identifier_length=128,
            )
        except Exception as e:
            msg = f"创建数据库 engine 失败, 错误为: {traceback.format_exc()}"
            raise DBError(msg) from e
        else:
            return engine

    def insert_dataframe(
            self,
            table: str,
            data: pd.DataFrame,
            schema: Optional[str] = None,
            if_exists: Optional[str] = 'append',
            chunksize: Optional[int] = 10000,
    ) -> None:
        """
        通过 pandas.DataFrame 插入数据
        重写 gjdata.core.relation_db.base.insert_dataframe, 提高插入效率

        :param table:
        :param data:
        :param schema:
        :param chunksize: 分批插入条数, 每次插入的条目数量
        :return:
        """

        if not isinstance(data, pd.DataFrame):
            raise ParamError("insert_dataframe 只支持 pandas.DataFrame")

        dtype = {
            c: types.VARCHAR(4000)
            for c in data.columns[data.dtypes == "object"]
        }

        with self._engine.begin() as connection:
            data.to_sql(table, connection, schema=schema, if_exists=if_exists, index=False,
                        dtype=dtype, chunksize=chunksize)

    def truncate(self, table: str, schema: Optional[str] = None) -> None:
        """
        清空表数据

        :param table:
        :param schema:
        :return:
        """

        if schema is not None:
            table = f"{schema}.{table}"

        clause = f"TRUNCATE TABLE {table}"

        self._execute(clause)
