import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import random
from collections import deque

# 自动检测GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

        # 将每个状态和下一状态转换为二维数组
        states = np.array([state.flatten() for state in states])
        next_states = np.array([state.flatten() for state in next_states])

        # 将actions, rewards, dones转为NumPy数组，保持一维结构
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

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

def train_dqn_model(data, num_episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, buffer_size=10000, lr=0.001):
    state_size = 1  # 仅用收盘价作为状态
    action_size = 3  # 买入、卖出、持有

    dqn = DQN(state_size, action_size).to(device)  # 将模型放到GPU
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    epsilon = epsilon_start
    total_trades = []
    total_rewards = []

    for episode in range(num_episodes):
        state = np.array([[data[0][0]]])  # 初始化第一个状态为第一个收盘价，形状为 (1, 1)
        state = torch.FloatTensor(state).to(device)  # 将state放到GPU
        done = False
        total_reward = 0
        trades = []

        for t in range(1, len(data)):
            # Epsilon-greedy选择动作
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                q_values = dqn(state)
                action = torch.argmax(q_values).item()

            next_state = np.array([[data[t][0]]])
            next_state = torch.FloatTensor(next_state).to(device)  # 将next_state放到GPU
            reward = compute_reward(action, state.cpu().numpy(), next_state.cpu().numpy())  # 奖励函数输入为numpy数组
            done = t == len(data) - 1

            replay_buffer.add(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

            if replay_buffer.size() >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # 将样本数据转换为 PyTorch 张量，并转移到 GPU
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # 计算Q值
                q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = dqn(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                # 更新模型
                loss = nn.MSELoss()(q_values, targets.detach())
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
    torch.save(dqn.state_dict(), "dqn_btc_model_gpu.pth")
    print("模型已保存到 dqn_btc_model_gpu.pth")

    return total_trades, total_rewards

# 修改后的奖励函数
def compute_reward(action, state, next_state):
    state_value = state.item()
    next_state_value = next_state.item()
    if action == 0:  # 买入
        return float(next_state_value - state_value)  # 盈利
    elif action == 1:  # 卖出
        return float(state_value - next_state_value)  # 盈利
    else:  # 持有
        return 0.0  # 确保返回浮点数

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
