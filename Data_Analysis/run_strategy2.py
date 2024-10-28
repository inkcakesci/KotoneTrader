import backtrader as bt
import pandas as pd
import sqlite3

# 策略类
class MA13MACDStrategy(bt.Strategy):
    params = (
        ('ma_period', 13),  # MA周期
        ('macd_period_short', 12),  # MACD短周期
        ('macd_period_long', 26),  # MACD长周期
        ('macd_signal_period', 9),  # MACD信号线周期
        ('lookback_period', 30),  # 回看周期，用于寻找最高红柱
        ('ma5_period', 5),  # MA5周期
    )

    def __init__(self):
        # 添加MA13均线
        self.ma13 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.ma_period)
        # 添加MA5均线
        self.ma5 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.ma5_period)
        # 添加MACD指标
        self.macd = bt.indicators.MACD(self.data.close, 
                                       period_me1=self.params.macd_period_short, 
                                       period_me2=self.params.macd_period_long, 
                                       period_signal=self.params.macd_signal_period)
        # 计算MACD直方图
        self.macd_hist = self.macd.macd - self.macd.signal
        # 用于跟踪订单
        self.order = None

    def next(self):
        # 检查当前是否持仓
        if not self.position:
            # 寻找过去30天MACD柱状图的最高红柱对应的K线最低价
            max_macd_bar = None
            max_macd_bar_low = None
            for i in range(-self.params.lookback_period, 0):
                hist_value = self.macd_hist[i]
                if max_macd_bar is None or hist_value > max_macd_bar:
                    max_macd_bar = hist_value
                    max_macd_bar_low = self.data.low[i]
            
            # 买入条件：回踩MA13均线不破且股价突破前面MACD最高红柱对应K线的最低价
            if self.data.low[0] > self.ma13[0] and self.data.close[0] > max_macd_bar_low:
                # 计算可以购买的股数（全仓买入）
                size = int(self.broker.getcash() / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # 卖出条件：跌破MA5均线清仓
            if self.data.close[0] < self.ma5[0]:
                self.order = self.sell(size=self.position.size)

    def notify_trade(self, trade):
        # 记录交易结果
        if trade.isclosed:
            print('交易结果：利润 %.2f, 毛利润 %.2f, 净利润 %.2f' % (trade.pnl, trade.pnlcomm, trade.pnl - trade.commission))

# 从SQLite数据库获取数据
def fetch_data_from_db(ticker, db_name='stocks_data.db'):
    # 连接SQLite数据库
    conn = sqlite3.connect(db_name)
    # 格式化表名
    table_name = f"stock_{ticker.split('.')[0].replace('SZ', '')}"
    # 查询数据
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql(query, conn, parse_dates=['Datetime'])
    # 设置Datetime为索引
    data.set_index('Datetime', inplace=True)
    conn.close()
    return data

# 回测函数
def run_backtest(tickers):
    for ticker in tickers:
        # 获取回测数据
        data = fetch_data_from_db(ticker)
        # 创建Cerebro实体
        cerebro = bt.Cerebro()
        # 添加数据到Cerebro
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        # 添加策略到Cerebro
        cerebro.addstrategy(MA13MACDStrategy)
        # 设置初始资金
        cerebro.broker.set_cash(100000.0)
        # 设置手续费
        cerebro.broker.setcommission(commission=0.001)
        # 添加分析器（计算胜率和盈利成绩）
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        # 打印初始资金
        print(f'回测股票: {ticker}')
        print('初始投资组合价值: %.2f' % cerebro.broker.getvalue())
        # 运行回测
        result = cerebro.run()
        # 打印最终资金
        print('最终投资组合价值: %.2f' % cerebro.broker.getvalue())
        # 获取分析器结果
        trade_analyzer = result[0].analyzers.trade_analyzer
        trade_analysis = trade_analyzer.get_analysis()
        # 输出胜率和盈利成绩
        total_trades = trade_analysis.get('total', {}).get('closed', 0)
        profitable_trades = trade_analysis.get('won', {}).get('total', 0)
        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        total_profit = trade_analysis.get('pnl', {}).get('net', {}).get('total', 0)
        print('总交易次数: %d' % total_trades)
        print('盈利交易次数: %d' % profitable_trades)
        print('胜率: %.2f%%' % win_rate)
        print('总利润: %.2f' % total_profit)
        # 绘制回测结果
        cerebro.plot()

if __name__ == '__main__':
    # 股票代码列表
    tickers = ["300418SZ", "300059SZ", "300015SZ", "300014SZ", "300033SZ"]
    # 运行回测
    run_backtest(tickers)
