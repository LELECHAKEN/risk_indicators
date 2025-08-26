# 数据库操作模块
from sqlalchemy import and_, column, or_
from .relation_db import OracleDB, MySQLDB
import yaml
from pathlib import Path

# 解析sqls
with open(str(Path(__file__).resolve().parent.joinpath("sqls.yaml")), encoding='utf-8') as cfg:
    sqls_config = yaml.safe_load(cfg)

__all__ = ['OracleDB', 'MySQLDB', 'and_', 'or_', 'column', 'sqls_config']