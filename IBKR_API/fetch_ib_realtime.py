# 安装必要的包
# pip install ib_insync
# pip install sqlite3

import sqlite3
from ib_insync import *
import re

# 设置连接信息（确保TWS正在运行，并且已成功登录）
ib = IB()

# 尝试连接到TWS以确认其正常运行
try:
    ib.connect('127.0.0.1', 7496, clientId=1)  # 修改端口为TWS当前的监听端口7496
    print("TWS 连接成功！")
except Exception as e:
    print(f"无法连接到TWS: {e}")
    exit()

# 创建数据库连接
conn = sqlite3.connect('ib_mkt_data.db')
c = conn.cursor()

def create_table_for_ticker(ticker):
    # 将表名中的特殊字符替换为下划线，确保表名以字母开头，避免SQLite的解析错误
    table_name = 't' + re.sub(r'[^a-zA-Z0-9]', '_', ticker.lower()) + "_5sec_bars"
    c.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            time TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    ''')
    conn.commit()
    return table_name

# 定义函数来获取市场数据并存储到数据库
def onBarUpdate(bars, hasNewBar, table_name):
    if hasNewBar:
        bar = bars[-1]
        try:
            # 实时输出5秒K线数据
            print(f"新5秒K线: 时间={bar.time}, 开盘价={bar.open_}, 最高价={bar.high}, 最低价={bar.low}, 收盘价={bar.close}, 成交量={bar.volume}")
            
            # 存储到SQLite数据库
            c.execute(f'''
                INSERT OR REPLACE INTO {table_name} (time, open, high, low, close, volume) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (bar.time, bar.open_, bar.high, bar.low, bar.close, bar.volume))
            conn.commit()
        except AttributeError as e:
            print(f"属性访问错误: {e}")

# 主函数，用于监控多个标的的实时数据
def main(tickers):
    contracts = []
    for ticker in tickers:
        # 创建合约（改为港股）
        contract = Stock(ticker, 'SEHK', 'HKD')
        ib.qualifyContracts(contract)
        contracts.append(contract)

        # 创建数据库表
        table_name = create_table_for_ticker(ticker)

        # 请求5秒实时K线数据
        bars = ib.reqRealTimeBars(contract, 5, 'MIDPOINT', False)
        bars.updateEvent += lambda bars, hasNewBar, table_name=table_name: onBarUpdate(bars, hasNewBar, table_name)

    # 保持连接状态以便持续获取数据
    ib.run()

# 指定标的的ticker列表（例如港股腾讯：700）
if __name__ == "__main__":
    tickers = ["700", "981"]  # 可以在此处添加多个ticker，例如港股代码
    main(tickers)

# 断开数据库连接
conn.close()
