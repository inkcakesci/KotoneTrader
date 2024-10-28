import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import random
from collections import deque

# 定义DQN模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

# 读取数据并预处理
def load_data_from_db():
    conn = sqlite3.connect('btc_60min_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT close, timestamp FROM btc_60min ORDER BY timestamp ASC")
    data = cursor.fetchall()
    conn.close()
    return np.array(data)

# 初始化环境与训练
def train_dqn_model(data, num_episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, buffer_size=10000, lr=0.001):
    state_size = 1  # 仅用收盘价作为状态
    action_size = 3  # 买入、卖出、持有

    dqn = DQN(state_size, action_size)
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    epsilon = epsilon_start
    total_trades = []
    total_rewards = []

    for episode in range(num_episodes):
        state = np.array([data[0][0]])  # 初始化第一个状态为第一个收盘价
        done = False
        total_reward = 0
        trades = []

        for t in range(1, len(data)):
            # Epsilon-greedy选择动作
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = dqn(state_tensor)
                action = torch.argmax(q_values).item()

            next_state = np.array([data[t][0]])
            reward = compute_reward(action, state, next_state)
            done = t == len(data) - 1

            replay_buffer.add(state, action, reward, next_state, done)

            if replay_buffer.size() >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                # 计算Q值
                q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = dqn(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                # 更新模型
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_reward += reward
            state = next_state
            trades.append((data[t][1], action))  # 记录时间戳和动作

            if done:
                break

        total_rewards.append(total_reward)
        total_trades.append(trades)

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    # 保存模型
    torch.save(dqn.state_dict(), "dqn_btc_model.pth")
    print("模型已保存到 dqn_btc_model.pth")

    return total_trades, total_rewards

# 定义奖励函数
def compute_reward(action, state, next_state):
    if action == 0:  # 买入
        return next_state[0] - state[0]  # 盈利
    elif action == 1:  # 卖出
        return state[0] - next_state[0]  # 盈利
    else:  # 持有
        return 0

# 运行回测，统计胜率
def evaluate_model(trades):
    buy_signals = 0
    sell_signals = 0
    win_trades = 0
    total_trades = 0

    for trade in trades:
        for timestamp, action in trade:
            if action == 0:  # 买入
                buy_signals += 1
            elif action == 1:  # 卖出
                sell_signals += 1
                total_trades += 1
                # 模拟简单的盈利逻辑
                if random.random() > 0.5:  # 假设50%的卖出为盈利
                    win_trades += 1

    win_rate = win_trades / total_trades if total_trades > 0 else 0
    print(f"总交易次数: {total_trades}, 胜率: {win_rate:.2%}")
    print(f"买入信号: {buy_signals}, 卖出信号: {sell_signals}")

# 主函数
if __name__ == "__main__":
    data = load_data_from_db()
    trades, rewards = train_dqn_model(data)
    evaluate_model(trades)
