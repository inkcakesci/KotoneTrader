import backtrader as bt
import datetime
import pandas as pd
import sqlite3

# 策略类
class EMA20Strategy(bt.Strategy):
    params = (
        ('ema_period', 20),  # EMA周期
        ('ema5_period', 5),  # EMA5周期
        ('cooldown_period', 6),  # 冷静期，单位为K线数量（6根K线 = 30分钟）
    )

    def __init__(self):
        # 添加EMA指标
        self.ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_period)
        self.ema5 = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema5_period)
        # 用于跟踪订单
        self.order = None
        # 冷静期计数器
        self.cooldown = 0

    def next(self):
        # 跳过冷静期
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        # 检查当前是否持仓
        if self.position:
            # 卖出条件1：收盘价跌破EMA20
            if self.data.close[0] < self.ema[0]:
                self.order = self.sell(size=self.position.size)
            # 卖出条件2：EMA5与EMA20死叉
            elif self.ema5[0] < self.ema[0] and self.ema5[-1] > self.ema[-1]:
                self.order = self.sell(size=self.position.size)
            # 卖出条件3：第二根K线下破EMA20
            elif self.data.close[-1] < self.ema[-1]:
                self.order = self.sell(size=self.position.size)
        else:
            # 买入条件：开盘价和收盘价均高于EMA20，且连续两根K线满足条件
            if self.data.open[0] > self.ema[0] and self.data.close[0] > self.ema[0] and \
               self.data.open[-1] > self.ema[-1] and self.data.close[-1] > self.ema[-1]:
                # 计算可以购买的股数（全仓买入，一手为200股）
                size = int(self.broker.getcash() / self.data.close[0] / 200) * 200
                if size > 0:
                    self.order = self.buy(size=size)
                    # 设置冷静期
                    self.cooldown = self.params.cooldown_period

    def notify_trade(self, trade):
        # 记录交易结果
        if trade.isclosed:
            print('交易结果：利润 %.2f, 毛利润 %.2f, 净利润 %.2f' % (trade.pnl, trade.pnlcomm, trade.pnl - trade.commission))

# 回测函数
def run_backtest(data):
    # 创建Cerebro实体
    cerebro = bt.Cerebro()
    
    # 添加数据到Cerebro
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # 添加策略到Cerebro
    cerebro.addstrategy(EMA20Strategy)

    # 设置初始资金
    cerebro.broker.set_cash(100000.0)

    # 设置手续费
    cerebro.broker.setcommission(commission=0.001)

    # 添加分析器（计算胜率和盈利成绩）
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    # 打印初始资金
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

# 从SQLite数据库获取数据
def fetch_data_from_db(ticker, db_name='stocks_data.db'):
    # 连接SQLite数据库
    conn = sqlite3.connect(db_name)
    # 格式化表名
    if ".HK" in ticker:
        ticker = ticker.replace(".HK", "")
    table_name = f"stock_{ticker}"
    # 查询数据
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql(query, conn, parse_dates=['Datetime'])
    # 设置Datetime为索引
    data.set_index('Datetime', inplace=True)
    conn.close()
    return data

if __name__ == '__main__':
    # 获取回测数据
    ticker = "0981.HK"
    data = fetch_data_from_db(ticker)

    # 运行回测
    run_backtest(data)
