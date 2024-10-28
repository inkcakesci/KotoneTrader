import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# 从SQLite数据库读取数据，扩展到720天的数据
def load_data_from_db(days=720):
    conn = sqlite3.connect('btc_60min_data.db')
    cursor = conn.cursor()
    
    # 当前日期往前推720天的时间戳
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    start_timestamp = int(start_time.timestamp())
    
    # 只选择过去720天的数据
    cursor.execute("SELECT close, timestamp FROM btc_60min WHERE timestamp >= ? ORDER BY timestamp ASC", (start_timestamp,))
    data = cursor.fetchall()
    conn.close()
    
    df = pd.DataFrame(data, columns=['close', 'timestamp'])
    df['close'] = df['close'].astype(float)  # 确保close列是浮点数
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

# 生成AO和MACD以及其他技术特征
def generate_features(df):
    # 计算AO指标（SMA(5) - SMA(34)）
    df['sma_5'] = df['close'].rolling(window=5).mean()  # 5期均线
    df['sma_34'] = df['close'].rolling(window=34).mean()  # 34期均线
    df['ao'] = df['sma_5'] - df['sma_34']  # AO指标

    # 计算MACD指标
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()  # EMA 12
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()  # EMA 26
    df['macd'] = df['ema_12'] - df['ema_26']  # MACD线
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()  # 信号线
    df['macd_diff'] = df['macd'] - df['signal']  # MACD与信号线的差值

    # 计算20期简单均线
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # 计算波动率 (5期收盘价的标准差)
    df['volatility'] = df['close'].rolling(window=5).std()
    
    df.dropna(inplace=True)  # 删除缺失值
    return df

# 生成买入、卖出、持有信号的标签
def generate_labels(df, holding_period=10):
    df['future_returns'] = df['close'].shift(-holding_period) / df['close'] - 1  # 未来N天的收益
    df['signal'] = np.where(df['future_returns'] > 0.02, 0,  # 买入信号 (2% 以上收益)
                            np.where(df['future_returns'] < -0.02, 1, 2))  # 卖出信号 (-2% 以下损失)
    df.dropna(inplace=True)
    return df

# 加载数据并生成特征和标签，扩展时间到720天
df = load_data_from_db(days=720)
df = generate_features(df)
df = generate_labels(df, holding_period=10)

# 选择特征和标签
features = ['ao', 'macd', 'macd_diff', 'sma_20', 'volatility']  # 使用AO、MACD、SMA、波动率
X = df[features]
y = df['signal']

# 保持原始索引
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# 划分训练集和测试集（可以用于评估模型，但我们会对全数据进行预测）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 训练随机森林模型
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# 对全数据集进行预测
y_pred_all = rf_model.predict(X)

# 将预测结果放入数据集中
df['pred_signal'] = y_pred_all

def backtest_strategy(df):
    initial_cash = 10000  # 初始现金
    cash = initial_cash
    position = 0  # 当前持仓
    entry_price = 0  # 买入价格
    returns = []
    stop_loss = 0.05  # 5% 止损
    take_profit = 0.1  # 10% 止盈
    net_liquidation = initial_cash  # 总市值，初始为现金

    for i, row in df.iterrows():
        if row['pred_signal'] == 0 and position == 0:  # 买入
            position = cash / row['close']
            entry_price = row['close']
            cash = 0  # 买入后现金为0
            print(f"买入: {row['timestamp']}，价格: {entry_price:.2f}")
        
        elif row['pred_signal'] == 1 and position > 0:  # 卖出
            cash = position * row['close']
            profit = (row['close'] - entry_price) / entry_price
            returns.append(profit)
            position = 0
            print(f"卖出: {row['timestamp']}，价格: {row['close']:.2f}, 盈利: {profit * 100:.2f}%")
        
        # 判断止盈止损
        if position > 0:
            current_profit = (row['close'] - entry_price) / entry_price
            if current_profit >= take_profit or current_profit <= -stop_loss:
                cash = position * row['close']
                returns.append(current_profit)
                position = 0
                print(f"卖出(止盈/止损): {row['timestamp']}，价格: {row['close']:.2f}, 盈利: {current_profit * 100:.2f}%")

        # 计算当前总市值（现金 + 当前持仓价值）
        net_liquidation = cash + position * row['close'] if position > 0 else cash

    # 计算总收益率和胜率
    total_return = net_liquidation - initial_cash
    win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
    
    print(f"总收益率: {total_return / initial_cash * 100:.2f}%, 胜率: {win_rate * 100:.2f}%")
    print(f"最终净清算市值（总市值）: {net_liquidation:.2f}")

    
backtest_strategy(df)
