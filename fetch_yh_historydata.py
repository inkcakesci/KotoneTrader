import yfinance as yf
import pandas as pd
import sqlite3
import logging
import configparser
from typing import Optional

class StockDataManager:
    def __init__(self, config_file="config.ini"):
        # 配置文件读取
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        # 从配置文件中获取数据库名称和股票列表文件名
        self.db_name = self.config.get("DATABASE", "db_name", fallback="stocks_data.db")
        self.watchlist_file = self.config.get("STOCKS", "watchlist_file", fallback="watchlist.txt")

        # 日志记录设置
        log_level = self.config.get("LOGGING", "level", fallback="INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

        # 数据库连接
        self.conn = self.setup_database()

    def setup_database(self) -> sqlite3.Connection:
        """初始化数据库连接"""
        try:
            conn = sqlite3.connect(self.db_name)
            logging.info(f"成功连接到数据库: {self.db_name}")
            return conn
        except sqlite3.Error as e:
            logging.error(f"数据库连接失败: {e}")
            raise

    def remove_suffix(self, ticker: str) -> str:
        """
        移除股票代码中的市场后缀 (.SS, .SZ, .HK)
        :param ticker: 股票代码
        :return: 没有后缀的股票代码
        """
        suffixes = ['.HK', '.SZ', '.SS', '.SH']
        for suffix in suffixes:
            if ticker.endswith(suffix):
                return ticker.replace(suffix, '')
        return ticker

    def create_table_if_not_exists(self, ticker: str):
        """如果表不存在，为给定的股票代码创建一个SQLite表。"""
        try:
            # 移除股票后缀来创建表名
            table_name = f"stock_{self.remove_suffix(ticker)}"

            cursor = self.conn.cursor()
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
            cursor.execute(create_table_query)
            self.conn.commit()
            logging.info(f"表 {table_name} 创建成功或已存在.")
        except sqlite3.Error as e:
            logging.error(f"创建表时出错: {e}")

    def download_stock_data(self, ticker: str, period: str = '2y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """从Yahoo Finance下载股票数据"""
        try:
            data = yf.download(tickers=ticker, period=period, interval=interval)
            if data.empty:
                logging.warning(f"未能下载到{ticker}的股票数据.")
                return None
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            logging.error(f"下载{ticker}数据时出错: {e}")
            return None

    def insert_data_into_table(self, ticker: str, data: pd.DataFrame):
        """将下载的股票数据插入SQLite数据库"""
        try:
            # 移除股票后缀来插入数据
            table_name = f"stock_{self.remove_suffix(ticker)}"

            cursor = self.conn.cursor()

            # 检查是否有 "Date" 列，并将其转换为 "Datetime" 列
            if 'Date' in data.columns:
                data.rename(columns={'Date': 'Datetime'}, inplace=True)

            # 确保 "Datetime" 列存在并正确格式化
            data['Datetime'] = pd.to_datetime(data['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')

            for _, row in data.iterrows():
                insert_query = f"""
                INSERT OR REPLACE INTO {table_name} (Datetime, Open, High, Low, Close, Adj_Close, Volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, (
                    row['Datetime'], row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume']
                ))
            self.conn.commit()
            logging.info(f"{ticker} 的数据插入成功.")
        except sqlite3.Error as e:
            logging.error(f"插入数据时出错: {e}")


    def full_update(self, ticker: str):
        """完整更新：从Yahoo Finance获取指定股票更长时间范围的数据（默认2年）"""
        self.create_table_if_not_exists(ticker)
        data = self.download_stock_data(ticker)
        if data is not None:
            self.insert_data_into_table(ticker, data)

    def update_recent_data(self, ticker: str):
        """更新数据库中指定股票最近5天的历史数据"""
        new_data = self.download_stock_data(ticker, period='5d', interval='1d')
        if new_data is not None:
            self.insert_data_into_table(ticker, new_data)

    def update_watchlist(self, mode: str = "full"):
        """根据监控列表更新股票数据"""
        try:
            with open(self.watchlist_file, "r") as file:
                watchlist = [line.strip() for line in file.readlines() if line.strip()]
        except FileNotFoundError:
            logging.error(f"未找到{self.watchlist_file}文件，请创建包含股票代码的文件，每行一个股票代码.")
            return
        
        for ticker in watchlist:
            if mode == "full":
                self.full_update(ticker)
                print(f"{ticker} 完整插入成功。")
            elif mode == "recent":
                self.update_recent_data(ticker)
                print(f"{ticker} 最近5天数据更新完毕。")
            else:
                logging.error("无效的更新模式。")

    def close_connection(self):
        """关闭数据库连接"""
        self.conn.close()

# Example usage:
if __name__ == "__main__":
    manager = StockDataManager()
    
    print("选择更新模式：\n1. 完整更新（2y）\n2. 快速更新（5d）")
    choice = input("请输入选项 (1 或 2): ")

    if choice == "1":
        manager.update_watchlist("full")
    elif choice == "2":
        manager.update_watchlist("recent")
    else:
        print("无效选项")

    manager.close_connection()
