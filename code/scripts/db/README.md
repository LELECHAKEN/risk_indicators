# relation_db
## Examples

### 使用准备

**初始化**

```python
from relation_db import OracleDB

# 关系型数据库
# PostgreSQL: postgresql+psycopg2://username:password@host:port/database
# MySQL: mysql+pymysql://username:password@host:port/database
# Oracle: oracle+cx_oracle://username:password@tnsname
# SQLite: sqlite:///path/test.relation_db

relation_relation_db = OracleDB("oracle+cx_oracle://username:password@tnsname")
```

**释放数据库连接**

```python
relation_relation_db.close()
```

### Insert

**insert_one**

插入一条数据

```python
data = {"name": "zhangsan", "age": 66}

relation_db.insert_one(table="test_table", data=data, schema="public")

None  # 返回值为 None
```

**insert_many**

插入多条数据

```python
data = [{"name": "zhangsan", "age": 66}, {"name": "lisi", "age": 62}]

relation_db.insert_many(table="test_table", data=data, schema="public")

None # 返回值为 None
```

**insert_dataframe**

通过 `pandas.DataFrame` 插入数据

```python
data = [{"name": "zhangsan", "age": 66}, {"name": "lisi", "age": 62}]
df = pd.DataFrame(data=data, chunksize=10)  # 若数据量很大, 可指定 chunksize, 分批插入, 默认为 1000

relation_db.insert_dataframe(table="test_table", data=df, schema="public")

None # 返回值为 None
```

**insert**

根据 `data` 的类型判断使用哪种方式插入数据

```python
data = {"name": "zhangsan", "age": 66}
data1 = [{"name": "zhangsan", "age": 66}, {"name": "lisi", "age": 62}]
data2 = pd.DataFrame(data=data1)

relation_db.insert(table="test_table", data=data, schema="public")
relation_db.insert(table="test_table", data=data1, schema="public")
relation_db.insert(table="test_table", data=data2, schema="public")

None  # 返回值为 None
None  # 返回值为 None
None  # 返回值为 None
```

### Select

通过 `select` 相关方法及 `condition` 参数实现复杂查询

**构造 condition **

```python
from db import and_, or_, column
column("name") == "zhangsan"

# where (name='zhangsan' and age=66) or (name='lisi' and age=62);
or_(and_(column("name") == "zhangsan", column("age") == 66), and_(column("name") == "lisi", column("age") == 62))
```

**select_one**

查询满足条件的一条数据

```python
condition = column("name") == "zhangsan"

relation_db.select_one(table="test_table", condition=condition, schema="public")

{'name': 'zhangsan', 'age': 66}  # 返回值为字典
```

**select_all**

查询满足条件的所有数据

```python
condition = column("name") == "zhangsan"

relation_db.select_all(table="test_table", condition=condition, schema="public")

# 返回值为列表嵌套字典
[
    {'name': 'zhangsan', 'age': 66}, 
    {'name': 'zhangsan', 'age': 66}, 
    {'name': 'zhangsan', 'age': 66}
]
```

**select_many**

查询满足条件的一部分数据

```python
condition = column("name") == "zhangsan"

relation_db.select_many(table="test_table", condition=condition, limit=2, schema="public")

# 返回值为列表嵌套字典
[
    {'name': 'zhangsan', 'age': 66}, 
    {'name': 'zhangsan', 'age': 66}
]
```

**read_sql**

通过原生 sql 查询, 返回值格式为 `pandas.DataFrame`, *注意: 只支持以 select 开头的查询语句*

```python
sql = "select * from test_table where name = 'zhangsan'"

relation_db.read_sql(sql=sql)

# 返回值为 pandas.DataFrame
       name  age
0  zhangsan   66
1  zhangsan   66
2  zhangsan   66


relation_db.read_sql(sql=sql, chunksize=10)

# chunksize 默认为 None, 若指定值, 该函数返回一个迭代器 (Iterator)
for df in relation_db.read_sql(sql=sql, chunksize=1):
    print("------")
    print(df)

------
       name  age
0  zhangsan   66
------
     name  age
0  zhangsan   66
------
     name  age
0  zhangsan   66

```

**select_sql**

通过原生 sql 查询, 返回值格式为 `List[Dict]` *注意: 只支持以 select 开头的查询语句*

```python
sql = "select * from test_table where name = 'zhangsan'"

relation_db.select_sql(sql=sql)

# 返回值为列表嵌套字典
[
    {'name': 'zhangsan', 'age': 66}, 
    {'name': 'zhangsan', 'age': 66}
]

```

### Delete

**delete**

删除满足条件的数据

```python
from db import and_, or_, column
condition = column("name") == "zhangsan"

relation_db.delete(table="test_table", condition=condition, schema="public")

DoesNotExist  # 数据不存在, 抛出 DoesNotExist 异常
None  # 返回结果为 None
```

### Update

**update**

更新满足条件的数据

```python
from db import and_, or_, column
condition = column("name") == "zhangsan"

relation_db.update("test_table", condition=condition, data={"age": 22}, schema="public")

DoesNotExist  # 数据不存在, 抛出 DoesNotExist 异常
None  # 返回结果为 None
```

### Others

**is_exist**

根据条件判断数据是否存在

```python
from db import and_, or_, column
relation_db.is_exist("test_table", condition=column("name") == "xiaoming", schema="public")

True  # 存在则返回 True
False  # 不存在则返回 False
```

**insert_or_update**

更新或插入, 通过 `primary_columns` 指定的字段名及其值来判断数据是否存在

若存在, 则更新, 不存在, 则新增. 执行此函数的最终结果是数据库至少有一条与 `data` 一致的数据

*注意: `primary_columns` 指定的字段名必须是 `data` 中出现的字段名的子集*

```python
relation_db.insert_or_update(
    table="test_table",
    primary_columns=["name"],  # 通过 name 字段及其值判断是否存在
    data={"name": "zhangsan", "age": 80},
    schema="public"
)

None  # 返回值为 None
```

**truncate**

清空表数据

```python
relation_db.truncate(table="test_table", schema="public")

None  # 返回值为 None
```
