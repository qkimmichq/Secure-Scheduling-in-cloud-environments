"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,  # 输出多少个action的值
            n_features,  # 接受多少个观测值的相关特征
            learning_rate=0.01,  # NN中learning_rate学习速率
            reward_decay=0.9,  # Q-learning中reward衰减因子
            e_greedy=0.7,
            e_greedy_max=0.95,
            replace_target_iter=300,
            memory_size=2000,
            batch_size=300,
            e_greedy_increment=1e-5,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy  # epsilon 的最大值
        self.epsilon_max = e_greedy_max
        self.replace_target_iter = replace_target_iter   # 更换 target_net 的步数 隔多少步后将target net 的参数更新为最新的参数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon_increment = e_greedy_increment  # epsilon 的增量  表示不断扩大epsilon，以便有更大的概率拿到好的值
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max   # 是否开启探索模式, 并逐步减少探索次数

        # total learning step  # 记录学习次数 (用于判断是否更换 target_net 参数)
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]   初始化全 0 记忆 [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))   # 和视频中不同, 因为 pandas 运算比较慢, 这里改为直接用 numpy
        # self.memory存储记忆的表
        # 行（高度）为存储记忆的数量
        # 列为(observation, action, reward, observation_)的长度
        # 对于一条记忆信息来说observation和observation_都有n_features的长度
        # 而action,reward都各自有一个单值信息
        # 则总列数为n_features+2+n_features

        # consist of [target_net, evaluate_net]  # 创建 [target_net, evaluate_net]
        self._build_net()

        # 替换 target net 的参数
        t_params = tf.get_collection('target_net_params')   # 替换 target net 的参数
        e_params = tf.get_collection('eval_net_params')   # 提取  eval_net 的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]   # 更新 target_net 参数
        # tf.assign(ref,value,validate_shape=None,use_locking=None,name=None)
        # 该操作在赋值后输出一个张量，该张量保存'ref'的新值。函数完成了将value赋值给ref的作用

        self.sess = tf.Session()

        # 输出 tensorboard 文件
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # 记录所有 cost 变化, 用于最后 plot 出来观看

    def _build_net(self):
        # ------------------ build evaluate_net 创建 eval 神经网络, 及时提升参数  -------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input  用来接收 observation，作为NN的输入
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss 用来接收 q_target 的值, 这个之后会通过计算得到
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到 ， 首先对图层进行配置，w,b初始化，第一层网络的神经元数n_l1
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            # c_name作为一个存储变量的集合，其名称为eval_net_params[],将q估计的参数都放入这个集合中
            # Variable()构造函数或get_variable()自动将新变量添加到图形集合：GraphKeys.GLOBAL_VARIABLES（默认）。
            # 这个方便函数返回该集合的内容。
            # 用于调用参数，将q估计的参数都放在collection这个集合当中
            # tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
            # 返回一个生成具有正态分布的张量的初始化器
            # tf.constant_initializer可以简写为tf.Constant()初始化为常数
            # tf.zeros_initializer()也可以简写为tf.Zeros()
            # tf.ones_initializer(), 也可以简写为tf.Ones()


            # first layer. collections is used later when assign to target net --eval_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                # 创建一个变量对于get_variable()，来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的。
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)  # l1输出维度[None,n_l1]

            # second layer. collections is used later when assign to target net  # eval_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2   # L2输出q估计维度[None,self_action]

        with tf.variable_scope('loss'):  # 求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))   # 基于Q估计与Q现实，构造loss-function
        with tf.variable_scope('train'):  # 梯度下降
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)    # 进行训练

        # ------------------ build target_net --创建 target 神经网络, 提供 target Q ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input # 接收下个 observation
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables  将q现实的参数都放入这个集合中
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net  # target_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net  # target_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):   # hasattr(object, name)判断一个对象里面是否有name属性或者name方法，返回BOOL值，初始不存在这个索引项，创建
            self.memory_counter = 0               # 判断self对象有name特性返回True， 否则返回False。即没有这个索引值memory_counter，则令self.memory_counter=0

        # 记录一条 [s, a, r, s_] 记录
        transition = np.hstack((s, [a, r], s_))   # numpy.hstack(tup)参数tup可以是元组，列表，或者numpy数组，返回结果为按顺序堆叠numpy的数组（按列堆叠一个）。

        # replace the old memory with new memory  # 总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
        index = self.memory_counter % self.memory_size   # index 取余  达到一个替换的效果
        self.memory[index, :] = transition    # 替换过程

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder  # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]    # 因为 observation 加入时是一维的数值，np.newaxis 为 numpy.ndarray（多维数组）增加一个轴,多加入了一个行轴

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)     # 随机选择
        return action

    def learn(self):
        # check to replace target parameters  # 检查是否替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:  # 隔self.replace_target_iter后将target net 的参数更新为最新的参数
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory   # 从 memory 中随机抽取 batch_size 这么多记忆
        if self.memory_counter > self.memory_size:  # 如果需要记忆的步数超过记忆库容量 #从给定的一维阵列self.memory_size生成一个随机样本，size为Output shape
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:  # 步数未超过记忆总容量，则最多在self.memory_counter个记忆值中选择32个索引数值
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)
        q_next, q_eval = self.sess.run(  # 运行这两个神经网络
            [self.q_next, self.q_eval],
            feed_dict={   # 前面store_transition(observation, action, reward, observation_)在run_this里都是参数
                self.s_: batch_memory[:, -self.n_features:],  # fixed params  q_next由目标值网络用记忆库中倒数n_features个列（observation_）的值做输入
                self.s: batch_memory[:, :self.n_features],  # newest params   q_eval由预测值网络用记忆库中正数n_features个列（observation）的值做输入
            })

        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        # 具体在下面还有一个举例说明.

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)  # 返回一个长度为self.batch_size的 “索引值列表” aray([0,1,2,...,31])
        eval_act_index = batch_memory[:, self.n_features].astype(int) # 返回一个长度为32的动作列表,从记忆库batch_memory中的标记的第2列，self.n_features=2
        # 即RL.store_transition(observation, action, reward, observation_)中的action，注意从0开始记，所以eval_act_index得到的是action那一列

        reward = batch_memory[:, self.n_features + 1]  # 返回一个长度为32奖励的列表，提取出记忆库中的reward

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        假如在这个 batch 中, 我们有2个提取的记忆, 根据每个记忆可以生产3个 action 的值:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        然后根据 memory 当中的具体 action 位置来修改 q_target 对应 action 上的值:
        比如在:
            记忆 0 的 q_target 计算值是 -1, 而且我用了 action 0;
            记忆 1 的 q_target 计算值是 -2, 而且我用了 action 2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        所以 (q_target - q_eval) 就变成了:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        最后我们将这个 (q_target - q_eval) 当成误差, 反向传递会神经网络.
        所有为 0 的 action 值是当时没有选择的 action, 之前有选择的 action 才有不为0的值.
        我们只反向传递之前选择的 action 的值,
        """

        # train eval network  # 训练 eval_net
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)    # 记录 cost 误差

        # increasing epsilon  # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)


