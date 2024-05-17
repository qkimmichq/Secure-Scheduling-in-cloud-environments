import sys
import os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

from Deep_learning.model_1.Workflow import Workflow
from Deep_learning.model_1.Processor import Processor
from Deep_learning.model_1.Processor import Service
from Deep_learning.model_1.Task import Task
from Deep_learning.resources_1.dax.DAX_parser import read_workflow
from Deep_learning.util_1.env_1 import Env
from data.data_1 import data  # cpu\ram\storage  数据存储位置
from data.data_1 import private_level  # 存储安全级别需求

import gym
import torch
import datetime
from common.utils import save_results, make_dir
from common.utils import plot_rewards, plot_rewards_cn
from Deep_learning.util_1.DQN_torch import DQN

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = "DQN"  # 算法名称
env_name = 'workflow'  # 环境名称


class DQNConfig:
    ''' 算法相关参数设置
    '''

    def __init__(self):
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 580  # 训练的回合数
        self.test_eps = 10  # 测试的回合数
        # 超参数
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.epsilon_start = 0.90  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 20000  # 经验回放的容量
        self.batch_size = 300  # mini-batch SGD中的批量大小
        self.target_update = 600  # 目标网络的更新频率
        self.hidden_dim = 256  # 网络隐藏层


class PlotConfig:
    ''' 绘图相关参数设置
    '''

    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        self.path_Montage_25 = curr_path + "/outputs/" + self.env_name + \
                       '/' + '20220711-202338' + '/models/'
        self.path_Sipht_30 = curr_path + "/outputs/" + self.env_name + \
                       '/' + '20220711-181756' + '/models/'
        self.path_Epigenomics_24 = curr_path + "/outputs/" + self.env_name + \
                       '/' + '20220710-155430' + '/models/'


def env_agent_config(cfg, wf, Task_1, processors, budget):
    ''' 创建环境和智能体
    '''
    env = Env(wf, Task_1, processors, budget)

    state_dim = env.n_features  # 状态维度
    action_dim = env.n_actions  # 动作维度
    agent = DQN(state_dim, action_dim, cfg)  # 创建智能体
    return env, agent


def train(cfg, env, agent):
    ''' 训练
    '''
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done = env.step(action)  # 更新环境，返回transition
            agent.memory.push(state, action, reward, next_state, done)  # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))

    print('完成训练！')

    return rewards, ma_rewards


def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    # 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
        print(env.task_exec)
        print('cost:', env.cost, ' time:', max(env.vm_time))
        print(env.vm_time)
        for i in env.Pros:
            print(i.tasks)

    return rewards, ma_rewards


if __name__ == "__main__":

    dax_name = ['Epigenomics_24', 'Epigenomics_46', 'Epigenomics_100', 'Epigenomics_997',
                'Inspiral_30', 'Inspiral_50', 'Inspiral_100', 'Inspiral_1000',
                'Montage_25', 'Montage_50', 'Montage_100', 'Montage_1000',
                'Sipht_30', 'Sipht_60', 'Sipht_100', 'Sipht_1000']

    dax_filepath = "../resources_1/dax/{}.xml".format('Sipht_30')
    wf = read_workflow(dax_filepath, dax_name)  # wf --- workflow类

    # print(len(wf.tasks))
    id_task = dict()  # 创建空字典 对应的 task.id: task
    for i in wf.tasks:
        id_task[i.id] = i
        # print(i.id)
    # print(id_task)

    for i in wf.tasks:
        l1, l2 = list(i.parents), list(i.children)
        l1.sort(key=lambda t: t.id)
        l2.sort(key=lambda t: t.id)
        i.predecessors = l1  # 前置任务 按id排序
        i.successors = l2  # 后置任务 按id排序

    l3 = list(wf.head_task.children)
    l3.sort(key=lambda t: t.id)
    wf.head_task.successors = l3  # 设置头节点的后置任务

    Task_1 = list(wf.tasks)
    Task_1.sort(key=lambda t: t.id)  # 按id排序
    # print(Task_1)

    for i in range(len(Task_1)):
        Task_1[i].cpu = data['data_{}'.format(len(Task_1))].cpu[i]
        Task_1[i].ram = data['data_{}'.format(len(Task_1))].ram[i]
        Task_1[i].storage = data['data_{}'.format(len(Task_1))].storage[i]
        Task_1[i].private_level = private_level['private_{}'.format(len(Task_1))][i]

    Task_1.insert(0, wf.head_task)
    for i in range(len(Task_1)):
        Task_1[i].index = i
        # print(Task_1[i])

    prolevel = [[1, 3.75, 4, 0.067], [2, 7.5, 32, 0.133], [4, 15, 80, 0.266],
                [8, 30, 160, 0.532], [8, 15, 60, 0.65], [32, 60, 240, 2.6]]

    processors = []  # 初始化处理器
    for i in range(len(Task_1)):
        pro = Processor(i)
        for j in range(len(prolevel)):
            ser = Service(j)
            ser.CPU = prolevel[j][0]
            ser.Ram = prolevel[j][1]
            ser.Storage = prolevel[j][2]
            ser.Price = prolevel[j][3]
            pro.service.append(ser)
        processors.append(pro)

    budget = 10000000
    rewards = []

    cfg = DQNConfig()
    plot_cfg = PlotConfig()
    # 训练
    # env, agent = env_agent_config(cfg, wf, Task_1, processors, budget)
    # rewards, ma_rewards = train(cfg, env, agent)
    # make_dir(plot_cfg.result_path, plot_cfg.model_path)  # 创建保存结果和模型路径的文件夹
    # agent.save(path=plot_cfg.model_path)  # 保存模型

    env, agent = env_agent_config(cfg, wf, Task_1, processors, budget)
    # agent.load(path=plot_cfg.path_25)  # 导入模型
    agent.load(path=plot_cfg.path_Montage_25)  # 导入模型
    rewards, ma_rewards = test(cfg, env, agent)

