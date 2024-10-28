# Introduction | 介绍

Project Name：Kotone Trader  
项目名：小琴音交易Bot

# Work in Progress | 目前需求

 - 需要对`fetch_yh_data.py`做测试.
 - 需要完成`riskAsm_MonteCarlo.py`的各项功能.
 - 需要完成回测功能.
 - 需要完成对不同策略给出template基本框架.

 - `Data_Analysis\*`暂时没啥用

 - `IBKR_API`是核心内容，需要找ib拿一个papertrade certification用于测试.
  

## `fetch_yh_data.py`

- 改善API的使用方法
   - 有默认输出的功能，每一次调用（如果成功）自动存储股票代码进watchlist
   - 改善数据库逻辑，创建视图（view）