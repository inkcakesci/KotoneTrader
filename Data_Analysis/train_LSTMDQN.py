import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os

# 创建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # 输出三个动作：买入、卖出、持有

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后时间步的输出
        return out

# 创建DQN模型
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

# 经验回放
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

# 训练函数
def train(env, model, dqn, num_episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                q_values = model(state)
                action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            buffer.add((state, action, reward, next_state, done))

            if buffer.size() > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = model(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    # 保存模型
    torch.save(model.state_dict(), "lstm_dqn_model.pth")
    print("模型训练完成并保存到 'lstm_dqn_model.pth' 文件中。")

# 假设你已经有一个简易的环境来测试
class DummyEnv:
    def __init__(self, data):
        self.data = data
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step:self.current_step+30]

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 30
        reward = 1 if action == 0 else 0  # 假设简单的奖励机制
        next_state = self.data[self.current_step:self.current_step+30]
        return next_state, reward, done

# 示例数据
data = np.random.rand(1000, 5)  # 模拟价格数据（open, high, low, close, volume）

# 初始化模型和环境
input_size = 5  # 假设输入5个特征
hidden_size = 128
num_layers = 2
env = DummyEnv(data)
model = LSTMModel(input_size, hidden_size, num_layers)
dqn = DQN(input_size, 3)

# 训练模型
train(env, model, dqn)
