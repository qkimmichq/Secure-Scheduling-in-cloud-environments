import random
import matplotlib as plt
import numpy as np

from Deep_learning.model_1.Workflow import Workflow
from Deep_learning.model_1.Processor import Processor
from Deep_learning.model_1.Processor import Service
from Deep_learning.model_1.Task import Task
from Deep_learning.resources_1.dax.DAX_parser import read_workflow
from Deep_learning.util_1.DQN import DeepQNetwork
from Deep_learning.util_1.env_1 import Env
from Deep_learning.util_1.memory import Memory
from sklearn.preprocessing import MinMaxScaler, StandardScaler


EPISODES = 1400
MEMORY_SIZE = 10000



def run_env():
    step = 0
    for episode in range(EPISODES):
        rwd = 0.0
        # 初始化环境
        obs = env.reset()
        # print(episode)

        while True:

            # 根据观测值选择行为
            action = DQN.choose_action(obs)
            # 推进一个时间步长  环境根据行为给出下一个状态、奖励、是否终止
            obs_, reward, done = env.step(action)
            rwd += reward
            # DQN存储记忆
            DQN.store_transition(obs, action, reward, obs_)
            # 控制学习起始时间和频率（先积累一些记忆再开始学习）
            if (step > 200 ) and (step % 5 == 0):
                DQN.learn()
            # 将下一个状态值变为下次循环的状态值
            obs = obs_

            #print(done)
            if done:
                if episode == EPISODES - 1:
                    print(env.task_exec)
                    print(max(env.vm_time))
                    print(env.vm_time)
                    print(sum(env.vm_cost))
                    print(rwd)
                    for i in env.Pros:
                        print(i.tasks)
                if episode % 10 == 0:
                    print(
                        'episode:' + str(episode) + ' steps:' + str(step) + ' reward0:' + str(rwd) +
                         ' eps_greedy0:' + str(DQN.epsilon))

                # rewards.append(rwd)
                break
            step += 1
    DQN.plot_cost()
    print('Endding')

 
if __name__ == '__main__':

    # 读取xml文件
    dax_name = 'Montage_25'
    dax_filepath = "../resources_1/dax/{}.xml".format(dax_name)
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

    cpu__1 = [9, 5, 9, 2, 7, 2, 2, 6, 10, 8, 2, 7, 5, 1, 10, 2, 2, 9, 5, 10, 4, 8, 3, 7, 9]
    ram__1 = [10.65, 8.41, 8.79, 8.63, 11.33, 3.15, 5.65, 6.83, 1.17, 6.93, 8.57, 9.36, 6.99, 5.59, 1.12, 13.14, 6.89,
              10.71, 8.56, 10.97, 5.18, 10.05, 0.8, 12.56, 5.55]
    storage__1 = [18.01, 135.91, 135.01, 26.55, 18.31, 38.16, 43.39, 111.7, 139.07, 33.51, 96.7, 79.57, 64.08, 49.11,
                  107.72, 77.25, 138.12, 85.99, 88.47, 92.87, 123.35, 18.71, 65.92, 44.33, 119.74]

    for i in range(len(Task_1)):
        if i == 0:
            Task_1[i].cpu = 0
            Task_1[i].ram = 0
            Task_1[i].storage = 0
            continue
        Task_1[i].cpu = cpu__1[i-1]
        Task_1[i].ram = ram__1[i-1]
        Task_1[i].storage = storage__1[i-1]
    budget = 10000
    rewards = []
    scaler = StandardScaler()
    env = Env(wf, Task_1, processors, budget)

    # memories = [Memory(MEMORY_SIZE)]

    DQN = DeepQNetwork(env.n_actions, env.n_features,
                        learning_rate=0.001,
                        replace_target_iter=200,
                        e_greedy_increment=5e-5,
                        )

    run_env()



'''
plt.figure(1)
    plt.plot(np.arange(len(rewards)), rewards)
    plt.plot(np.arange(len(rewards)), [139 for i in range(len(rewards))])
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()

'''







