import yfinance as yf
import sqlite3
import time
from datetime import datetime, timedelta

# 初始化SQLite数据库连接
conn = sqlite3.connect('btc_60min_data.db')
c = conn.cursor()

# 创建表格用于存储60分钟数据，增加日期时间一列（如果不存在）
c.execute('''
    CREATE TABLE IF NOT EXISTS btc_60min (
        timestamp INTEGER PRIMARY KEY,
        datetime TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL
    )
''')
conn.commit()

# 将UNIX时间戳转换为可读的日期时间格式
def format_datetime(unix_time):
    return datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d-%H')

# 存储数据到SQLite数据库
def store_data(data):
    for index, row in data.iterrows():
        timestamp = int(index.timestamp())  # 转换为UNIX时间戳
        datetime_str = format_datetime(timestamp)  # 转换为日期时间字符串
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        volume = row['Volume']
        
        # 插入数据到数据库，覆盖相同时间戳的数据
        c.execute('''
            INSERT OR REPLACE INTO btc_60min (timestamp, datetime, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, datetime_str, open_price, high_price, low_price, close_price, volume))
    
    conn.commit()

# 获取最近7天的比特币数据
def get_recent_btc_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # 获取最近7天的数据
    print(f"Fetching data from {start_date} to {end_date}...")
    
    # 获取最近7天的数据
    btc_data = yf.download(tickers="BTC-USD", interval='60m', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    # 如果有数据，存储到数据库
    if not btc_data.empty:
        store_data(btc_data)
        print("数据更新完成。")
    else:
        print("未获取到数据。")

# 执行更新操作
get_recent_btc_data()

# 关闭数据库连接
conn.close()

print("数据库连接关闭。")
