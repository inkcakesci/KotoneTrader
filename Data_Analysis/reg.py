import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 定义美股交易时间
US_MARKET_OPEN = '10:00'
US_MARKET_CLOSE = '16:00'
US_EASTERN = pytz.timezone('US/Eastern')

# 获取历史数据
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, interval='1h')
    data.index = data.index.tz_convert(US_EASTERN)  # 转换为美东时间
    return data

# 过滤美股交易时间内的数据
def filter_us_market_hours(data):
    data = data.between_time(US_MARKET_OPEN, US_MARKET_CLOSE)
    return data

# 获取数据
start_date = '2023-10-01'
end_date = '2023-10-31'

coin_data = fetch_data('COIN', start=start_date, end=end_date)
btc_data = fetch_data('BTC-USD', start=start_date, end=end_date)
mara_data = fetch_data('MARA', start=start_date, end=end_date)

# 检查数据是否为空
if coin_data.empty or btc_data.empty or mara_data.empty:
    print("数据为空，请检查日期范围或数据源。")
else:
    # 将数据按美股交易时间过滤
    coin_data = filter_us_market_hours(coin_data)
    btc_data = filter_us_market_hours(btc_data)
    mara_data = filter_us_market_hours(mara_data)
    
    # 重新采样数据到相同的时间间隔（每小时）
    coin_data = coin_data.resample('1H').ffill()
    btc_data = btc_data.resample('1H').ffill()
    mara_data = mara_data.resample('1H').ffill()
    
    # 检查过滤后的数据是否为空
    if coin_data.empty or btc_data.empty or mara_data.empty:
        print("美股交易时间内的数据为空，请检查日期范围或数据源。")
    else:
        # 合并数据，确保时间对齐
        combined_data = pd.merge(coin_data['Close'], btc_data['Close'], left_index=True, right_index=True, suffixes=('_COIN', '_BTC'))
        combined_data = pd.merge(combined_data, mara_data['Close'], left_index=True, right_index=True, suffixes=('', '_MARA'))
        combined_data.rename(columns={'Close': 'Close_MARA'}, inplace=True)
        
        # 检查合并后的数据是否为空
        if combined_data.empty:
            print("合并后的数据为空，请检查美股交易时间内的数据是否存在重叠。")
        else:
            # 检查并处理缺失值
            combined_data = combined_data.dropna()
            
            # 检查处理后的数据是否为空
            if combined_data.empty:
                print("合并后的数据在去除缺失值后为空，无法进行回归分析。")
            else:
                # 对数据进行归一化处理
                scaler = StandardScaler()
                combined_data[['Close_COIN', 'Close_BTC', 'Close_MARA']] = scaler.fit_transform(combined_data[['Close_COIN', 'Close_BTC', 'Close_MARA']])
                
                # 使用非线性模型（随机森林回归）预测 COIN 和 MARA 的价格
                X = combined_data[['Close_BTC']]
                
                # 随机森林回归：BTC 预测 COIN
                y_coin = combined_data['Close_COIN']
                X_train, X_test, y_train, y_test = train_test_split(X, y_coin, test_size=0.2, random_state=42)
                rf_coin = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_coin.fit(X_train, y_train)
                y_pred_coin = rf_coin.predict(X_test)
                print("随机森林回归：BTC 对 COIN 的预测结果：")
                print(f"均方误差 (MSE): {mean_squared_error(y_test, y_pred_coin)}")
                print(f"R² 值: {r2_score(y_test, y_pred_coin)}")
                
                # 随机森林回归：BTC 预测 MARA
                y_mara = combined_data['Close_MARA']
                X_train, X_test, y_train, y_test = train_test_split(X, y_mara, test_size=0.2, random_state=42)
                rf_mara = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_mara.fit(X_train, y_train)
                y_pred_mara = rf_mara.predict(X_test)
                print("随机森林回归：BTC 对 MARA 的预测结果：")
                print(f"均方误差 (MSE): {mean_squared_error(y_test, y_pred_mara)}")
                print(f"R² 值: {r2_score(y_test, y_pred_mara)}")
                
                # 计算 MARA 和 BTC 的波动率及杠杆率
                combined_data['Return_BTC'] = combined_data['Close_BTC'].pct_change()
                combined_data['Return_MARA'] = combined_data['Close_MARA'].pct_change()
                
                btc_volatility = combined_data['Return_BTC'].std()
                mara_volatility = combined_data['Return_MARA'].std()
                
                if btc_volatility != 0:
                    leverage_ratio = mara_volatility / btc_volatility
                    print(f"BTC 的波动率: {btc_volatility}")
                    print(f"MARA 的波动率: {mara_volatility}")
                    print(f"MARA 对应于 BTC 的杠杆率估计: {leverage_ratio}")
                else:
                    print("BTC 的波动率为零，无法计算杠杆率。")
