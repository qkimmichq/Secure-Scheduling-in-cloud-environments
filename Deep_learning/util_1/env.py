import numpy as np
import random
from Deep_learning.model_1.Workflow import Workflow

# 奖励值的设定

class Env:
    def __init__(self, wf, Tasks, Pros, budget):
        self.wf = wf                                    # 工作流--wf
        self.Tasks = Tasks                              # 任务--Tasks
        self.Pros = Pros                                # VM--Pros
        #print(len(Pros))
        self.budget = budget
        self.n_vm = len(Pros)                           # vm数量
        self.n_actions = len(Pros)         # 动作为 所有任务 与 VM 之间的映射关系，
        self.n_features = 1 + len(Pros)     # task_type and vm_state  观测特征数  --not sure
        self.n_task = len(Tasks)                        # 任务数
        self.dim_state = self.n_task

        self.now_time = None                            # 当前运行的时间
        self.add_time = None                            # 增加的最小时间
        self.workflow = None                            # 工作流
        self.task = None                                # 当前的任务
        self.vm_time = None                             # VM可用时间
        self.vm_cost = None                             # VM的开销
        self.released = None                            # 已完成的任务列表
        self.start_time = None                          # 任务可以开始执行的时间
        self.task_exec = None                           # 任务开始执行的时间，执行完毕的时间
        self.ready_task = None                          # 可以开始执行的任务列表  若所选的任务不在该列表中，奖励值 -10000 并跳出本次
        self.done = None                                # 是否执行完毕
        # self.reward = None

        self.reset()                                    # 初始化

    def reset(self):  # 重置环境  now_time设为0  将入口任务设为当前任务
        self.now_time = 0                                 # 当前运行的时间
        self.workflow = self.wf                           # 工作流
        self.vm_time = np.zeros(self.n_vm)    # VM_time --- not sure
        self.vm_cost = np.zeros(self.n_vm)                # VM_cost
        self.released = []                                # 已完成列表
        self.start_time = np.zeros(self.n_task)           # 任务可用开始执行的时间
        self.task_exec = []                               # 任务的 index ， 开始执行时间， 执行完毕的时间
        self.ready_task = [0]                              # 将头结点添加进  ready_task

        self.task = self.workflow.head_task.index          # 当前的任务设为--头结点,取的是 index值

        for i in self.Pros:
            i.tasks = []
        for i in self.Tasks:
            i.proindex = -1

        self.done = False
        # self.reward = 0
        #print(self.vm_time)
        obs = np.concatenate(([0], self.vm_time), 0)  # 返回当前任务的index值，VM_cost,0
        #print(obs)
        return obs

    def step(self, action):  # 推进一个时间步长
        # done = False
        sumCpu, sumRam, sumStorage = [0 for i in range(len(self.Pros))], [0 for i in range(len(self.Pros))], [0 for i in range(len(self.Pros))]
        canPut = self.canPutpro(self.Tasks[self.task], sumCpu, sumRam, sumStorage)   # 获取当前任务可以调度的VM集合
        # print(action,canPut)
        if action not in canPut:  # 若动作值不在集合中
            reward = -self.n_task
            obs = np.concatenate(([self.task], self.vm_time), 0)
            done = True
            return obs, reward, done

        self.Tasks[self.task].proindex = action
        self.Pros[action].tasks.append(self.task)

        self.ready_task.remove(self.task)  # 将task移除ready_task,并添加进已完成列表，
        self.released.append(self.task)

        self.set_action()   # 将后继任务中可以开始执行的任务添加进ready_task.

        reward = self.rewards(action)      # 返回奖励值
        obs = np.concatenate(([self.task], self.vm_time), 0)
        done = self.is_done()                    # 是否执行完毕
        #print(done)

        return obs, reward, done


    def set_action(self):
        # 将其后继任务中可以开始执行的任务添加进ready_task

        for i in self.Tasks[self.task].successors:          # 后继任务
            if i.index in self.ready_task:                  # 已经在ready——task  --多余
                continue
            count = 0
            for j in i.predecessors:
                if j.index not in self.released:            # 前置任务未完成
                    break
                count += 1
            if count == len(i.predecessors):     # 后继任务的前置任务都执行完毕
                self.ready_task.append(i.index)
        # print(self.ready_task)


    def time_reward(self, action):   # action--第几个VM 返回上一阶段的完成时间 和 当前任务的执行时间   --- done
        # flag -- 当前能放下
        strategy = []
        strategy.append(self.task)

        last_makespan = max(self.vm_time)   # 上一阶段的完成时间

        exec_time = self.Tasks[self.task].runtime #* (1 + self.Tasks[self.task].private_level * 0.2)   # 任务的执行时间
        self.Tasks[self.task].est = self.cal_est(action)

        if self.vm_time[action] >= self.Tasks[self.task].est:               # VM的可用时间 >= 任务的开始时间
            strategy.append(self.vm_time[action])
            self.Tasks[self.task].ast = self.vm_time[action]
            self.vm_time[action] += exec_time                                # VM的可用时间 += 任务的执行时间
            strategy.append(self.vm_time[action])
            self.Tasks[self.task].aft = self.vm_time[action]
        else:                                                                # VM的可用时间 < 任务的开始时间
            strategy.append(self.Tasks[self.task].est)
            self.Tasks[self.task].ast = self.Tasks[self.task].est
            self.vm_time[action] = self.Tasks[self.task].est + exec_time    # VM的可用时间 = 任务的开始时间 + 任务的执行时间
            strategy.append(self.vm_time[action])
            self.Tasks[self.task].aft = self.vm_time[action]

        self.task_exec.append(strategy)

        #print(self.is_done())
        if not self.is_done():       # 未结束运行
            self.task = random.choice(self.ready_task)
        return last_makespan, exec_time



    def rewards(self, action):
        rankservice = self.Pros[action].service_rank
        last_makespan, exec_time = self.time_reward(action)  # 上一阶段的完成时间 和 当前任务的执行时间
        inc_makespan = max(self.vm_time) - last_makespan  # 执行当前任务后的完成时间 - 上一阶段的完成时间
        inc_makespan = round(inc_makespan, 4)
        task = self.Tasks[self.task]
        if rankservice == -1:
            k = 0
            for i in self.Pros[action].service:
                if (task.cpu <= i.CPU) and (task.ram <= i.Ram) and (task.storage <= i.Storage):
                    self.Pros[action].service_rank = k
                    break
                k += 1

        cost = self.Pros[action].service[rankservice].Price * exec_time
        self.vm_cost[action] += cost
        if len(self.released) == self.n_task:
            return self.n_task * 2 - inc_makespan
        return self.n_task // 4 - inc_makespan



    def is_done(self):
        cost = sum(self.vm_cost)
        if cost > self.budget:
            return  True
        if len(self.released) == self.n_task:    # 如果以完成任务列表长度 == tasks的长度，即所有任务均已完成
            return True
        return False

    def cal_est(self, action):
        task = self.Tasks[self.task]
        est = 0
        for i in task.predecessors:
            if i.proindex == action:
                time = i.aft
            else:
                time = i.aft + self.transfer1(i.index, self.task)
            if time > est:
                est = time
        return est


    def canPutpro(self, stack, sumCpu, sumRam, sumStorage):    # 是否可以调度当前任务
        canPut = []
        for i in range(len(self.Pros)):
            for j in range(len(self.Tasks)):   # 遍历所有在 i 上的任务
                if self.Tasks[j].proindex == i:
                    # print('yes',self.tasks[j].index)
                    sumCpu[i] += self.Tasks[j].cpu
                    sumRam[i] += self.Tasks[j].ram
                    sumStorage[i] += self.Tasks[j].storage
            rankservice = self.Pros[i].service_rank
            if rankservice == -1:  # 未调度任务
                canPut.append(i)
            elif (sumCpu[i] + stack.cpu <= self.Pros[i].service[rankservice].CPU) and\
                    (sumRam[i] + stack.ram <= self.Pros[i].service[rankservice].Ram) and\
                    (sumStorage[i] + stack.storage <= self.Pros[i].service[rankservice].Storage):  # 已调度，是否能继续放下
                canPut.append(i)
        return canPut    # 返回可以调度当前任务的VM集合


    def transfer1(self, t1, t2):
        if self.Tasks[t1] == self.workflow.head_task or self.Tasks[t2] == self.workflow.head_task:
            return 0.0

        output = self.Tasks[t1].output_files  # 输入文件
        input = self.Tasks[t2].input_files  # 输出文件
        req_files = output.keys() & input.keys()  # 输出==输入，即为需要传输的数据量
        vol = sum([output[f].size for f in list(req_files)])
        return round(vol / 12500000, 5)  # 带宽 12500000
