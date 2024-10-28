import backtrader as bt
import yfinance as yf
from datetime import datetime

# 定义多头策略类
class LongStrategy(bt.Strategy):
    params = (
        ('short_ema', 5),   # 短期EMA
        ('long_ema', 20),   # 长期EMA
    )

    def __init__(self):
        # 指标初始化
        self.ema5 = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.short_ema)
        self.ema20 = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.long_ema)
        self.cross_up_confirmed = False  # 用于标记突破站稳

    def next(self):
        # 多头买入逻辑: K线突破EMA20，且第二根K线站稳
        if not self.position:  # 如果当前没有多头仓位
            if self.data.close[0] > self.ema20[0] and self.data.close[-1] > self.ema20[-1]:
                self.cross_up_confirmed = True
            if self.cross_up_confirmed and self.data.close[0] > self.ema20[0]:
                self.buy()  # 买入
                self.cross_up_confirmed = False

        # 多头卖出逻辑: EMA5下穿EMA20 或 K线跌破EMA20
        elif self.position.size > 0:  # 如果持有多头仓位
            if self.ema5[0] < self.ema20[0] or self.data.close[0] < self.ema20[0]:
                self.sell()  # 卖出

# 定义空头策略类
class ShortStrategy(bt.Strategy):
    params = (
        ('short_ema', 5),   # 短期EMA
        ('long_ema', 20),   # 长期EMA
    )

    def __init__(self):
        # 指标初始化
        self.ema5 = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.short_ema)
        self.ema20 = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.long_ema)
        self.cross_down_confirmed = False  # 用于标记跌破站稳

    def next(self):
        # 空头卖出逻辑: K线跌破EMA20，且第二根K线站稳
        if not self.position:  # 如果当前没有空头仓位
            if self.data.close[0] < self.ema20[0] and self.data.close[-1] < self.ema20[-1]:
                self.cross_down_confirmed = True
            if self.cross_down_confirmed and self.data.close[0] < self.ema20[0]:
                self.sell()  # 开空头
                self.cross_down_confirmed = False

        # 空头平仓逻辑: EMA5上穿EMA20 或 K线突破EMA20
        elif self.position.size < 0:  # 如果持有空头仓位
            if self.ema5[0] > self.ema20[0] or self.data.close[0] > self.ema20[0]:
                self.buy()  # 平空头

# 获取数据函数
def get_data(ticker):
    data = bt.feeds.YahooFinanceData(
        dataname=ticker,
        timeframe=bt.TimeFrame.Minutes,
        compression=60,  # 1小时
        fromdate=datetime.now() - bt.TimeDelta(days=720),  # 过去720天
        todate=datetime.now(),
        historical=True
    )
    return data

# 运行回测
def run_backtest():
    cerebro = bt.Cerebro()

    # 添加多头和空头策略
    cerebro.addstrategy(LongStrategy)
    cerebro.addstrategy(ShortStrategy)
    
    # 下载BTC数据并加载到backtrader中
    data = get_data('BTC-USD')  # 使用BTC-USD获取比特币的K线数据
    cerebro.adddata(data)

    # 设置初始资金
    cerebro.broker.set_cash(1000000)
    cerebro.broker.setcommission(commission=0.001)  # 佣金

    # 打印初始资金
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # 开始回测
    cerebro.run()

    # 打印最终资金
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # 绘图
    cerebro.plot()

if __name__ == '__main__':
    run_backtest()
