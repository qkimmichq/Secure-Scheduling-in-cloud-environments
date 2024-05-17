import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """ 初始化q网络,为全连接网络
            state_dim: 输入的特征数即环境的状态维度
            action_dim: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        V = self.V(x)
        A = self.A(x)
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)

        return Q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量转移
        state, action, reward, next_state, done = zip(*batch)  # 解压成状态，动作等
        return state, action, reward, next_state, done

    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)


class D3QN:
    def __init__(self, state_dim, action_dim, cfg):

        self.action_dim = action_dim  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = DuelingDeepQNetwork(state_dim=state_dim, action_dim=action_dim,
                                          fc1_dim=cfg.fc1_dim, fc2_dim=cfg.fc2_dim)
        self.target_net = DuelingDeepQNetwork(state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=cfg.fc1_dim, fc2_dim=cfg.fc2_dim)
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)  # 优化器
        self.memory = ReplayBuffer(cfg.memory_capacity)  # 经验回放

        self.cost_his = []

    def choose_action(self, state):
        ''' 选择动作
        '''
        # self.frame_idx += 1
        # if random.random() > self.epsilon(self.frame_idx):
        #     state = torch.tensor([state], device=self.device, dtype=torch.float32)
        #     q_values = self.policy_net.forward(state)
        #     action = torch.argmax(q_values).item()  # 选择Q值最大的动作
        # else:
        #     action = random.randrange(self.action_dim)

        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        q_values = self.policy_net.forward(state)
        return q_values

    def update(self):
        if len(self.memory) < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # 转为张量
        state_batch = torch.tensor(state_batch, dtype=torch.float).to(self.device)
        action_batch = torch.tensor(action_batch).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float).to(self.device)
        done_batch = torch.tensor(done_batch).to(self.device)

        indices = np.arange(self.batch_size)

        q_values = self.policy_net(state_batch)[indices, action_batch]  # 计算当前状态(s_t,a)对应的Q(s_t, a)
        next_q_values = self.target_net(next_state_batch)  # 计算下一时刻的状态(s_t_,a)对应的Q值
        max_actions = torch.argmax(self.policy_net(next_state_batch), dim=1)
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values[indices, max_actions]
        next_q_values[done_batch] = 0.0

        loss = nn.MSELoss()(q_values, expected_q_values)  # 计算均方根损失
        #print(loss)
        #self.cost_his.append(loss)

        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():  # clip防止梯度爆炸
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('loss')
        plt.xlabel('training steps')
        plt.show()
