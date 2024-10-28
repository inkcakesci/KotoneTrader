### ！特别提醒！本程序设计程序交易，请自行承担风险

from ib_insync import *

# 设置连接信息（确保TWS正在运行，并且已成功登录）
ib = IB()

# 尝试连接到TWS以确认其正常运行
try:
    ib.connect('127.0.0.1', 7496, clientId=1)  # 修改端口为TWS当前的监听端口7496
    print("TWS 连接成功！")
except Exception as e:
    print(f"无法连接到TWS: {e}")
    exit()

# 创建麦当劳（MCD）的股票合约
mcd_contract = Stock('MCD', 'SMART', 'USD')
ib.qualifyContracts(mcd_contract)

# 定义监控价格并创建订单的函数
order_placed = False

def onMcdBarUpdate(bars, hasNewBar):
    global order_placed
    if hasNewBar:
        bar = bars[-1]
        try:
            # 实时输出5秒K线数据
            print(f"【市场数据】新5秒K线: 时间={bar.time}, 开盘价={bar.open_}, 最高价={bar.high}, 最低价={bar.low}, 收盘价={bar.close}, 成交量={bar.volume}")
            
            # 当价格低于313.8时，创建一个购买一股的市场订单（MKT）
            #if bar.close < 313.8 and not order_placed:
            #    print("【交易情况】触发买入麦当劳(MCD)信号，价格低于313.5!!!")
            #    order = MarketOrder('BUY', 1)
            #    trade = ib.placeOrder(mcd_contract, order)
            #    print(f"【交易情况】订单已创建: {trade}")
            #    order_placed = True  # 确保只下单一次
        except AttributeError as e:
            print(f"属性访问错误: {e}")

# 请求麦当劳的5秒实时K线数据
mcd_bars = ib.reqRealTimeBars(mcd_contract, 5, 'MIDPOINT', False)
mcd_bars.updateEvent += onMcdBarUpdate

# 保持连接状态以便持续获取数据
ib.run()
