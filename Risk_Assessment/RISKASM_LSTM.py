import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# 从SQLite获取过去的数据，并计算移动平均线和成交量变化
def get_stock_data(conn, ticker, years=2):
    query = f"""
    SELECT Datetime, Open, High, Low, Close, Volume FROM {ticker} 
    WHERE Datetime >= datetime('now', '-{years} years')
    ORDER BY Datetime
    """
    df = pd.read_sql(query, conn)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 计算移动平均线 (MA)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()

    # 计算成交量变化 (Volume Change)
    df['Volume_Change'] = df['Volume'].pct_change()

    # 将 NaN 替换为 0，处理无效数据
    df['Volume_Change'].fillna(0, inplace=True)
    df.dropna(inplace=True)  # 去掉其他空值

    # 处理无穷大和 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=9, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, dropout=0.2)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(2, 1, self.hidden_layer_size).cuda(),
                            torch.zeros(2, 1, self.hidden_layer_size).cuda())

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def lstm_predict(data, n_predictions=30):
    # 创建一个 scaler 仅用于 Close 列
    close_scaler = MinMaxScaler(feature_range=(-1, 1))
    close_prices = data[:, 3].reshape(-1, 1)
    close_prices_normalized = close_scaler.fit_transform(close_prices)

    # 其他特征无需归一化，直接使用原始数据
    data_normalized = data.copy()

    # 构建训练数据
    train_window = 60
    train_inout = [(data_normalized[i:i+train_window], close_prices_normalized[i+train_window])
                   for i in range(len(data_normalized) - train_window)]

    model = LSTMModel().cuda()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 8
    for i in range(epochs):
        total_loss = 0.0
        with tqdm(total=len(train_inout), desc=f"Epoch {i+1}/{epochs}", unit="batch") as pbar:
            for seq, labels in train_inout:
                seq = torch.tensor(seq, dtype=torch.float32).cuda()
                labels = torch.tensor(labels, dtype=torch.float32).cuda()

                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(2, 1, model.hidden_layer_size).cuda(),
                                     torch.zeros(2, 1, model.hidden_layer_size).cuda())

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels.squeeze())
                single_loss.backward()

                # 加入梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += single_loss.item()
                pbar.update(1)

        mse = total_loss / len(train_inout)
        print(f"Epoch {i+1} MSE: {mse}")

    predictions = []
    with torch.no_grad():
        for _ in range(n_predictions):
            seq = torch.tensor(data_normalized[-train_window:], dtype=torch.float32).cuda()
            model.hidden_cell = (torch.zeros(2, 1, model.hidden_layer_size).cuda(),
                                 torch.zeros(2, 1, model.hidden_layer_size).cuda())
            predicted_close_price = model(seq).cpu().numpy()

            # 仅对 Close 列进行反归一化
            predicted_close_price = close_scaler.inverse_transform(predicted_close_price.reshape(-1, 1))[0][0]
            predictions.append(predicted_close_price)

            # 更新其他特征的值，保持它们不变
            new_row = data_normalized[-1].copy()
            new_row[3] = predicted_close_price  # 更新 Close 列
            data_normalized = np.vstack([data_normalized, new_row])

    return predictions


def main():
    conn = sqlite3.connect('stocks_data.db')
    ticker = 'TSM'
    data = get_stock_data(conn, ticker, years=2)

    ohlcv_ma_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'MA60', 'Volume_Change']].values

    actual_close_prices = data['Close'].values

    lstm_predictions = lstm_predict(ohlcv_ma_data, n_predictions=30)
    print("LSTM 预测结果:", lstm_predictions)

    conn.close()

    plt.figure(figsize=(10, 6))
    plt.plot(actual_close_prices, label='history', color='blue')

    future_index = range(len(actual_close_prices), len(actual_close_prices) + len(lstm_predictions))
    plt.plot(future_index, lstm_predictions, label=' close predict', color='red', linestyle='--')

    plt.title(f'{ticker} close predict')
    plt.xlabel('T')
    plt.ylabel('close')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
