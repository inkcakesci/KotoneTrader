# Introduction | 介绍

Project Name：Kotone Trader  
项目名：小琴音交易Bot

# Stock Data Manager | 股票数据管理器

This is a Python tool for downloading and managing stock data from Yahoo Finance (`yfinance`) and storing it in an SQLite database.

这是一个用于从Yahoo Finance下载和管理股票数据并存储到SQLite数据库的Python工具。

## Features | 功能
- Download historical stock data (2-year or 5-day options).
- Store data in SQLite.
- Configure via `config.ini`.

- 下载历史股票数据（2年或5天选项）。
- 存储数据到SQLite数据库。
- 通过`config.ini`进行配置。

## Requirements | 依赖
- Python 3.x
- `yfinance`, `pandas`

## Installation | 安装
1. Clone the repository or download the code.
2. Install the required packages:
   ```sh
   pip install yfinance pandas
   ```

1. 克隆仓库或下载代码。
2. 安装依赖包：
   ```sh
   pip install yfinance pandas
   ```

## Configuration | 配置
Create `config.ini` in the root directory:

在根目录创建 `config.ini` 文件：

```ini
[DATABASE]
db_name = stocks_data.db

[STOCKS]
watchlist_file = watchlist.txt

[LOGGING]
level = INFO
```

- `db_name`: SQLite database name.
- `watchlist_file`: File containing stock tickers.
- `level`: Logging level (`INFO`, `DEBUG`, etc.).

- `db_name`：SQLite数据库名称。
- `watchlist_file`：包含股票代码的文件。
- `level`：日志级别（如 `INFO`，`DEBUG` 等）。

## Usage | 使用
1. Create `watchlist.txt` with stock tickers (e.g., `AAPL`, `MSFT`, `300418.SZ`).
2. Run the script:
   ```sh
   python main.py
   ```
3. Choose update mode:
   - `1`: Full update (2 years)
   - `2`: Quick update (5 days)

1. 创建 `watchlist.txt` 文件，每行一个股票代码（如 `AAPL`, `MSFT`, `300418.SZ`）。
2. 运行脚本：
   ```sh
   python main.py
   ```
3. 选择更新模式：
   - `1`：完整更新（2年）
   - `2`：快速更新（5天）

## Example | 示例
```
---START-UP-(*´ε`*)---
---炒股千万条，安全第一条，风控不规范，账户两行泪.---
选择更新模式：
1. 完整更新（2y）
2. 快速更新（5d）
```

## Notes | 注意
- Ensure a stable internet connection.
- Only supports Yahoo Finance tickers.

- 请确保稳定的网络连接。
- 仅支持Yahoo Finance的股票代码格式。

## License | 许可证
MIT License.

