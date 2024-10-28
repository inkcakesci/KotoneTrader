import pandas as pd
import sqlite3
import yfinance as yf

# 快速将股票数据存入SQLite数据库

def download_and_store_data_quick(tickers, db_name='stocks_data.db'):
    # 连接SQLite数据库
    conn = sqlite3.connect(db_name)
    for ticker in tickers:
        # 下载数据
        data = yf.download(ticker, period='1y', interval='1d')
        data.reset_index(inplace=True)
        # 将Datetime列转换为字符串格式
        data['Date'] = data['Date'].astype(str)
        # 去除股票代码中的后缀
        table_name = f"stock_{ticker.split('.')[0]}"
        # 创建表（如果不存在）
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            Datetime TEXT PRIMARY KEY,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Adj_Close REAL,
            Volume INTEGER
        )
        """
        conn.execute(create_table_query)
        conn.commit()
        # 插入数据
        for _, row in data.iterrows():
            insert_query = f"""
            INSERT OR REPLACE INTO {table_name} (Datetime, Open, High, Low, Close, Adj_Close, Volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            conn.execute(insert_query, (
                row['Date'], row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume']
            ))
        conn.commit()
    conn.close()

if __name__ == '__main__':
    # 股票代码列表
    tickers = ["300418.SZ", "300059.SZ", "300015.SZ", "300014.SZ", "300033.SZ"]
    # 下载并快速存储数据
    download_and_store_data_quick(tickers)
