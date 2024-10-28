import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_hourly_data(conn, ticker, hours=720*24):
    """
    从SQLite数据库中获取小时线数据，并计算对数收益率
    :param conn: 数据库连接对象
    :param ticker: 股票代码
    :param hours: 获取数据的小时数（默认720天，每天24小时）
    :return: DataFrame，包含对数收益率的小时线数据
    """
    query = f"""
    SELECT Datetime, Close FROM {ticker} 
    ORDER BY Datetime DESC LIMIT {hours}
    """
    df = pd.read_sql(query, conn)
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # 按时间升序排列数据
    df = df.sort_values('Datetime')

    # 计算对数收益率（log returns）
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)  # 删除NaN值

    return df

def monte_carlo_simulation(df, hours=30*24, simulations=1000):
    """
    使用小时线数据进行蒙特卡罗模拟，未来30天
    :param df: 包含对数收益率的 DataFrame
    :param hours: 模拟的小时数（30天 x 24小时）
    :param simulations: 模拟路径的数量
    :return: 模拟的价格路径和月化波动率
    """
    last_price = df['Close'].iloc[-1]  # 当前价格作为起点
    log_returns_mean = df['Log_Returns'].mean()  # 平均对数收益率
    log_returns_std = df['Log_Returns'].std()  # 对数收益率的标准差

    prices = np.zeros((simulations, hours))
    prices[:, 0] = last_price

    dt = 1 / 24  # 每小时的时间步长
    for t in range(1, hours):
        z = np.random.standard_normal(simulations)
        prices[:, t] = prices[:, t - 1] * np.exp((log_returns_mean - 0.5 * log_returns_std**2) * dt
                                                 + log_returns_std * np.sqrt(dt) * z)

    # 计算模拟的月化波动率
    simulated_volatility = np.std(np.log(prices[:, -1] / prices[:, 0])) * np.sqrt(21 * 24)
    return prices, simulated_volatility

def plot_simulation(prices, actual_prices, title):
    """
    绘制蒙特卡罗模拟路径和实际价格对比图
    :param prices: 模拟的价格路径
    :param actual_prices: 实际历史价格
    :param title: 图表标题
    """
    plt.figure(figsize=(12, 6))

    # 绘制模拟路径
    for i in range(min(10, prices.shape[0])):  # 绘制前10条模拟路径
        plt.plot(prices[i], color='gray', alpha=0.3)

    # 绘制实际价格
    plt.plot(np.arange(-len(actual_prices), 0), actual_prices, color='blue', label='Actual Prices')

    plt.title(title)
    plt.xlabel('Time (Hourly Intervals)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(ticker):
    """
    主函数：传入股票代码，输出波动率并绘图
    :param ticker: 股票代码
    """
    conn = sqlite3.connect('stocks_data.db')

    # 获取小时线数据
    df = get_hourly_data(conn, ticker)

    # 运行蒙特卡罗模拟
    prices, simulated_vol = monte_carlo_simulation(df)

    # 打印模拟的月化波动率
    print(f'Simulated Monthly Volatility: {simulated_vol:.2%}')

    # 最近30天的实际价格用于比较
    actual_prices = df['Close'].values[-30 * 24:]

    # 绘制模拟路径与实际价格对比图
    plot_simulation(prices, actual_prices, title='Monte Carlo Simulation - 30 Days (Hourly)')

    conn.close()

if __name__ == "__main__":
    # 用户只需输入股票代码
    ticker = input("Enter the stock ticker: ").strip().upper()
    main(ticker)
