# risk indicators

## 投资风险框架
|**风险指标**|**对应类**|
|:---:|:---:|  
|风险指标基础框架(母类)|Risk Indicators|  
|基础指标| Basic Indicators|
|集中度指标| Concentration Indicators|
|市场风险指标| Market Indicators|
|市场风险指标(比较基准) |Market Indicators_b|
|信用风险指标|Credit Indicators|
|流动性风险指标|Liquidity Indicators|
|核心流动性指标|Liquidity Indicators core|
|可转债风险指标|CB Indicators|
|期限错配指标|Risk Indicators mismatch|
|衍生品相关指标|derivatives|
|管理层专项指标|Risk Indicators mg|


***每日运行的操作手册见：./doc/日频风险指标准备v4.one***

## Main Function
1. DailyRun.py
    主程序，风险指标每日运行程序。
    * SplitFundManager.py  待vba将单产品的指标excel运行完后，该程序会将各产品的指标文件移动复制至对应经理的文件夹内。
    * data_transfer.py  将公盘里各经理文件夹内的文件同步至云文档对应文件夹。

2. daily_check.py  
    单位净值数据每日清洗及检查程序。经复权调整后的单位净值数据落入数据库dpe_portfolionav表，检查结果会落入数据库dpe_nav_check表。
   
3. daily_check_mail.py
    单位净值检查结果每日自动邮件发送提醒。
   
4. risk_monitor_basic_data.py
    投委会相关监控指标监控的前置程序，包括过去一年的最大回撤情况、滚动一年胜率计算等。

5. update_bond_index_dura.py
    每月更新债券指数基金对应的指数久期，需从wind下载各指数对应的成分券文件。

## 结构说明
-code
  -main：程序运行总接口
  -scripts：程序脚本（存放个性化代码）
    -config：路径、数据库配置
    -db:  数据库相关操作模块
      -sqls.yaml:  sql语句统一存放地址
      -ConnectingDatabase：数据库操作类
    -utils：工具类模块
      -log_utils:  日志操作工具

-data: 存放数据

-doc：存放项目相关文档

-logs：日志存放地址

README.md

requirement.txt: 记录所有依赖包及其精确的版本号，以便新环境部署。生成方式：pip install pipreqs；pipreqs . --encoding=utf8 --force；