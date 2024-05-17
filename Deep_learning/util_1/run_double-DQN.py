import random
import matplotlib as plt
import numpy as np
import tensorflow as tf

from Deep_learning.model_1.Workflow import Workflow
from Deep_learning.model_1.Processor import Processor
from Deep_learning.model_1.Processor import Service
from Deep_learning.model_1.Task import Task
from Deep_learning.resources_1.dax.DAX_parser import read_workflow
from Deep_learning.util_1.double_DQN import DoubleDQN
from Deep_learning.util_1.env_1 import Env
from Deep_learning.util_1.memory import Memory
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data.data_1 import data   # cpu\ram\storage  数据存储位置


EPISODES = 2000
MINI_BATCH = 128
MEMORY_SIZE = 2000

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

    for i in range(len(Task_1)):
        Task_1[i].cpu = data['data_{}'.format(len(Task_1))].cpu[i]
        Task_1[i].ram = data['data_{}'.format(len(Task_1))].ram[i]
        Task_1[i].storage = data['data_{}'.format(len(Task_1))].storage[i]

    # 初始化处理器
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

    Task_1.insert(0, wf.head_task)
    for i in range(len(Task_1)):
        Task_1[i].index = i
        # print(Task_1[i])

    budget = 10000
    rewards = []
    scaler = StandardScaler()
    env = Env(wf, Task_1, processors, budget)
    memories = [Memory(MEMORY_SIZE)]

    sess = tf.Session()
    with tf.variable_scope('Natural_DQN'):
        natural_DQN = DoubleDQN(
            n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, double_q=False, sess=sess
        )

    with tf.variable_scope('Double_DQN'):
        double_DQN = DoubleDQN(
            n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

    sess.run(tf.global_variables_initializer())


    def train(RL):
        total_steps = 0

        for episode in range(EPISODES):
            rwd = 0.0
            observation = env.reset()
            while True:
                # if total_steps - MEMORY_SIZE > 8000: env.render()
                total_steps += 1
                action = RL.choose_action(observation)

                observation_, reward, done = env.step(action)
                rwd += reward
                #reward /= 10  # normalize to a range of (-1, 0). r = 0 when get upright
                # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
                # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

                RL.store_transition(observation, action, reward, observation_)

                if (total_steps > MEMORY_SIZE) and (total_steps % 5 == 0):  # learning
                    RL.learn()

                observation = observation_

                if done:
                    if episode == EPISODES - 1:
                        print(env.task_exec)
                        print(max(env.vm_time))
                        print(env.vm_time)
                        for i in env.Pros:
                            print(i.tasks)
                    if episode % 10 == 0:
                        print(
                            'episode:' + str(episode) + ' steps:' + str(total_steps) +
                            ' reward0:' + str(rwd) )

                    #rewards.append(rwd)
                    break

        #return RL.q


    q_natural = train(natural_DQN)
    q_double = train(double_DQN)


'''
    plt.plot(np.array(q_natural), c='r', label='natural')
    plt.plot(np.array(q_double), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()

'''











