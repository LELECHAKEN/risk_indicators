#!/bin/python
# -*- coding: utf-8 -*-
# __author__ = sunsn
# __datetime__ = 2021/1/20 16:47
"""
关系型数据库操作基类
"""
import traceback
from collections import abc
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import pandas as pd
from sqlalchemy import MetaData, Table, column
from sqlalchemy.engine import Engine, ResultProxy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.elements import BooleanClauseList, and_

from ..util.error_utils import (DBError, DoesNotExist, NotSupportError,
                                ParamError)



Base = declarative_base()


class RelationDBEngine(object):

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._engine = self._create_engine(self.db_url)
        self._metadata = self.__create_metadata(self._engine)

    def _create_engine(self, dsn: str):
        raise NotImplementedError

    def __create_metadata(self, engine: Engine) -> MetaData:
        try:
            metadata = MetaData(bind=engine)
        except Exception as e:
            msg = f"创建数据库metadata失败，类名为：{self.__name__}, 错误为：{traceback.format_exc()}"
            raise DBError(msg) from e
        else:
            return metadata

    def get_engine(self) -> Engine:
        return self._engine

    def get_metadata(self) -> MetaData:
        return self._metadata

    def _execute(self, clause) -> ResultProxy:
        return self._engine.execute(clause)

    def _reflect_table(self, table: str, schema: Optional[str] = None) -> Table:
        """
        反射表
        :param table:
        :return:
        """
        return Table(table, self._metadata, autoload=True, autoload_with=self._engine, schema=schema)

    def close(self) -> None:
        """
        关闭所有闲置的连接

        :return:
        """

        self._engine.dispose()


class CRUD(RelationDBEngine):

    def _insert(self, table: Table, data: Union[Dict, Sequence]) -> None:
        """
        在 SQLAlchemy 中, table.insert().values(data) 不支持 Oracle 数据库的
        单行多条插入, 使用 Engine.execute(table.insert(), data) 方式可默认调用
        cx_Oracle 扩展的 executemany 函数, 执行批量插入

        :param table:
        :param data:
        :return:
        """

        self._engine.execute(table.insert(), data)

    def _select(self, table: Table, condition: Optional[BooleanClauseList] = None) -> ResultProxy:

        clause = table.select()

        if condition is not None:
            clause = clause.where(condition)

        return self._execute(clause)

    def _delete(self, table: Table, condition: BooleanClauseList) -> None:

        clause = table.delete().where(condition)
        self._execute(clause)

    def _update(
            self,
            table: Table,
            data: Dict[str, Any],
            condition: BooleanClauseList,
    ) -> None:

        clause = table.update().where(condition).values(data)
        self._execute(clause)

    def insert_one(
            self,
            table: str,
            data: Dict[str, Any],
            schema: Optional[str] = None
    ) -> None:
        """
        插入一条数据

        :param table:
        :param data:
            e.g.: {"id": 7, "name": "somename7"},
        :param schema:
        :return:
        """

        if not isinstance(data, dict):
            raise ParamError("insert_one 只支持 dict")

        obj_tb = self._reflect_table(table, schema)
        self._insert(table=obj_tb, data=data)

    def insert_many(
            self,
            table: str,
            data: Sequence[Dict],
            schema: Optional[str] = None,
            chunksize: Optional[int] = 10000,
    ) -> None:
        """
        在一个事务内分批插入多条数据

        :param table:
        :param data:
            e.g.:
                [
                    {"id": 7, "name": "somename7"},
                    {"id": 8, "name": "somename8"},
                    {"id": 9, "name": "somename9"}
                ]
        :param schema:
        :param chunksize: 单次插入条目数量
        :return:
        """

        if not isinstance(data, abc.Sequence) or isinstance(data, str):
            raise ParamError("insert_many 只支持含有多条数据的序列, 例如 list, tuple 等")

        obj_tb = self._reflect_table(table, schema)
        nrows = len(data)
        chunks = int(len(data) / chunksize) + 1

        with self._engine.begin() as connection:

            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, nrows)
                if start_i >= end_i:
                    break

                connection.execute(obj_tb.insert(), data[start_i:end_i])

    def insert_dataframe(
            self,
            table: str,
            data: pd.DataFrame,
            schema: Optional[str] = None,
            chunksize: Optional[int] = 10000,
    ) -> None:
        """
        通过 pandas.DataFrame 插入数据

        :param table:
        :param data:
        :param schema:
        :param chunksize: 分批插入条数, 每次插入的条目数量
        :return:
        """

        if not isinstance(data, pd.DataFrame):
            raise ParamError("insert_dataframe 只支持 pandas.DataFrame")

        data.to_sql(table, self._engine, schema=schema, if_exists="append", index=False, chunksize=chunksize)

    def insert(
            self,
            table: str,
            data: Union[Dict[str, Any], Sequence[Dict[str, Any]], pd.DataFrame],
            schema: Optional[str] = None
    ) -> None:
        """
        插入数据
        （需要根据data的类型做判断,一条或者多条）

        :param table:
        :param data:
        :param schema:
        :return:
        """
        if isinstance(data, dict):
            self.insert_one(table=table, data=data, schema=schema)
        elif isinstance(data, abc.Sequence):
            self.insert_many(table=table, data=data, schema=schema)
        elif isinstance(data, pd.DataFrame):
            self.insert_dataframe(table=table, data=data, schema=schema)
        else:
            supported_data_type = ["dict", "list", "pandas.DataFrame"]
            raise ParamError(f"参数错误, data 只支持: {', '.join(supported_data_type)}")

    def select_one(
            self,
            table: str,
            schema: Optional[str] = None,
            condition: Optional[BooleanClauseList] = None,
    ) -> Dict:
        """
        查询满足查询条件的一条数据

        :param table:
        :param condition:
        :param schema:
        :return:
            e.g.:
                {
                    "name": "xiaoming",
                    "age": 20
                }
        """

        obj_tb = self._reflect_table(table, schema)
        columns = obj_tb.columns
        data = self._select(table=obj_tb, condition=condition).fetchone()

        if data is None:
            return dict()

        return dict(zip([str(c).split(".")[1] for c in columns], data))

    def select_all(
            self,
            table: str,
            schema: Optional[str] = None,
            condition: Optional[BooleanClauseList] = None,
    ) -> List[Dict]:
        """
        查询满足查询条件的所有数据

        :param table:
        :param condition:
        :param schema:
        :return:
        """
        obj_tb = self._reflect_table(table, schema)
        columns = obj_tb.columns
        data = self._select(table=obj_tb, condition=condition).fetchall()

        if not data:
            return list()

        result = [
            dict(zip([str(c).split(".")[1] for c in columns], single_data))
            for single_data in data
        ]

        return result

    def select_many(
            self,
            table: str,
            limit: int,
            schema: Optional[str] = None,
            condition: Optional[BooleanClauseList] = None,
    ) -> List[Dict]:
        """
        查询满足查询条件的一些数据数据, 根据 limit 限制

        :param table:
        :param condition:
        :param schema:
        :param limit:  TODO .limit() or .fetchmany(size)
        :return:
        """

        if not isinstance(limit, int) or limit < 1:
            raise ParamError("limit shuold be int and must greater than 0")

        obj_tb = self._reflect_table(table, schema)
        columns = obj_tb.columns
        data = self._select(table=obj_tb, condition=condition).fetchmany(limit)

        if not data:
            return list()

        result = [
            dict(zip([str(c).split(".")[1] for c in columns], single_data))
            for single_data in data
        ]

        return result

    def select_sql(self, sql: str) -> List[Dict]:
        """
        通过原生 SQL 查询

        :param sql:
        :return:
        """

        result = self.read_sql(sql).to_dict(orient="records")

        return result

    def read_sql(self, sql: str, chunksize: Optional[int] = None) -> Union[pd.DataFrame, Iterator]:
        """
        执行原生 SQL 并返回 Pandas.DataFrame 或一个迭代器 (Iterator), 参考 pandas.read_sql

        :param sql:
        :param chunksize:
        :return:
        """
        tmp_sql = sql.strip(" ").lower()

        if not (tmp_sql.startswith("select") or tmp_sql.startswith("with")):
            raise NotSupportError("只支持以 select, with 开头的查询语句")

        return pd.read_sql(sql, self._engine, chunksize=chunksize)

    def delete(
            self,
            table: str,
            condition: BooleanClauseList,
            schema: Optional[str] = None
    ) -> None:
        """
        删除数据, 必须传递筛选条件, 防止删除所有数据

        :param table:
        :param condition:
        :param schema:
        :return:
        """
        if not self.is_exist(table=table, condition=condition, schema=schema):
            raise DoesNotExist("数据不存在")

        obj_tb = self._reflect_table(table, schema)
        self._delete(table=obj_tb, condition=condition)

    def update(
            self,
            table: str,
            condition: BooleanClauseList,
            data: Dict[str, Any],
            schema: Optional[str] = None
    ) -> None:
        """
        更新数据

        :param table:
        :param condition:
        :param data:
        :param schema:
        :return:
        """
        if not self.is_exist(table=table, condition=condition, schema=schema):
            raise DoesNotExist("数据不存在")

        obj_tb = self._reflect_table(table, schema)
        self._update(obj_tb, data, condition)

    def is_exist(
            self,
            table: str,
            condition: BooleanClauseList,
            schema: Optional[str] = None
    ) -> bool:
        """
        根据条件判断数据是否存在

        :return:
        """

        if self.select_all(table, schema=schema, condition=condition):
            return True

        return False

    def insert_or_update(
            self,
            table: str,
            primary_columns: List[str],
            data: Dict[str, Any],
            schema: Optional[str] = None
    ) -> None:
        """
        根据 data 中 primary_columns 对应的值判断数据是否存在
        若存在, 更新除 primary_columns 对应的字段之外的值
        若不存在, 插入 data 的值

        最终结果, 数据库有 data

        :param table:
        :param primary_columns: 用来筛选数据是否存在, 必须被 data 包含
        :param data: 数据库最终结果
        :param schema:
        :return:
        """

        if not set(primary_columns).issubset(set(data.keys())):
            raise ParamError("data 中的字段必须包含 primary_columns 中的字段")

        condition = and_(column(key) == data[key] for key in primary_columns)

        if self.is_exist(table=table, condition=condition, schema=schema):
            self.update(table=table, condition=condition, data=data, schema=schema)
        else:
            self.insert(table=table, data=data, schema=schema)
