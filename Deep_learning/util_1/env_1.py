import numpy as np
import math
import random
from Deep_learning.model_1.Workflow import Workflow

# 奖励值的设定

class Env:
    def __init__(self, wf, Tasks, Pros, budget, scheduling_order):
        self.wf = wf                                    # 工作流--wf
        self.Tasks = Tasks                              # 任务--Tasks
        self.Pros = Pros                                # VM--Pros
        self.order = scheduling_order                   # 调度顺序
        sum = 0
        for i in Tasks:
            sum += i.runtime
        self.avg_time = sum // len(Tasks)
        #print(len(Pros))
        self.budget = budget
        self.cost = 0                                   # 开销
        self.n_vm = len(Pros)                           # vm数量
        self.n_actions = len(Pros)         # 动作为 所有任务 与 VM 之间的映射关系，
        self.n_features = 1 + len(Pros)     # task_type and vm_state  观测特征数  --not sure
        self.n_task = len(Tasks)                        # 任务数
        self.dim_state = self.n_task

        self.now_time = None                            # 当前运行的时间
        self.workflow = None                            # 工作流
        self.task = None                                # 当前的任务
        self.vm_time = None                             # VM可用时间
        # self.vm_cost = None                             # VM的开销
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
        # self.vm_cost = np.zeros(self.n_vm)                # VM_cost
        self.released = []                                # 已完成列表
        # self.start_time = np.zeros(self.n_task)           # 任务可用开始执行的时间
        self.scheduling_order = [i for i in self.order]  # 调度顺序
        self.task = self.scheduling_order[0]  # 当前的任务设为--头结点,取的是 index值
        self.task_exec = []                               # 任务的 index ， 开始执行时间， 执行完毕的时间
        # self.ready_task = []                              # 将头结点添加进  ready_task


        for i in self.Pros:
            i.tasks = []   # 存放的任务数量
            i.service_rank = -1   # vm的配置
            i.endtime = []   # [[end0, end1],[end1, end2], [end3, end4]]
            i.surplus = []   # 每个end区间的剩余资源量  [cpu,ram,storage]
        for i in self.Tasks:
            i.proindex = -1
            i.est = 0
            i.ast = 0
            i.aft = 0

        self.done = False
        # self.reward = 0
        #print(self.vm_time)
        obs = np.concatenate(([0], self.vm_time), 0)  # 返回当前任务的index值，VM_cost,0
        #print(obs)
        return obs

    def step(self, action):  # 推进一个时间步长
        self.Tasks[self.task].proindex = action     # 将当前任务的处理器设置为action
        self.Pros[action].tasks.append(self.task)    # 将任务添加到action 对应的处理器的任务列表中
        self.scheduling_order.remove(self.task)  # 将task移除调度顺序列表,并添加进已完成列表，
        self.released.append(self.task)    # 任务调度完毕

        # self.set_action()   # 将后继任务中可以开始执行的任务添加进ready_task.

        reward = self.rewards(action)      # 返回奖励值
        obs = np.concatenate(([self.task], self.vm_time), 0)       
        done = self.is_done()                    # 是否执行完毕
        #print(done)

        return obs, reward, done


    def set_action(self):   # 将其后继任务中可以开始执行的任务添加进ready_task
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


    def time_reward(self, action):   # action--第几个VM 返回上一阶段的完成时间 和 当前任务的执行时间
        strategy = []
        strategy.append(self.task)   # 当前的任务

        last_makespan = max(self.vm_time)   # 上一阶段的完成时间
        vm_time = self.vm_time[action]     # 上一阶段VM的时间

        exec_time = self.Tasks[self.task].runtime   # 任务的执行时间
        self.Tasks[self.task].est = self.cal_est(action)    # 任务可以执行的时间

        start_time = self.cal_start(action, self.Tasks[self.task].est)     # 返回当前任务可以开始的时间
        # print('start_time:',start_time)

        strategy.append(start_time)
        self.Tasks[self.task].ast = start_time
        end_time = start_time + exec_time
        strategy.append(end_time)
        # print('end_time:',end_time)
        self.Tasks[self.task].aft = end_time

        n = len(self.Pros[action].tasks)
        rankservice = self.Pros[action].service_rank
        cpu_sur = self.Pros[action].service[rankservice].CPU - self.Tasks[self.task].cpu
        ram_sur = self.Pros[action].service[rankservice].Ram - self.Tasks[self.task].ram
        storage_sur = self.Pros[action].service[rankservice].Storage - self.Tasks[self.task].storage

        if n == 1:
            self.Pros[action].endtime.append([start_time, end_time])
            self.vm_time[action] = end_time
            self.Pros[action].surplus.append([cpu_sur, ram_sur, storage_sur])
        else:
            end0 = self.Pros[action].endtime[n - 2][0]
            end1 = self.Pros[action].endtime[n - 2][1]
            if start_time >= end1:             # 大于等于上一任务的结束时间 [start,end]
                self.Pros[action].endtime.append([start_time, end_time])
                self.vm_time[action] = end_time
                self.Pros[action].surplus.append([cpu_sur, ram_sur, storage_sur])
            if start_time >= end0 and start_time < end1:
                if end_time > end1:
                    self.Pros[action].endtime.append([end1, end_time])
                    self.vm_time[action] = end_time
                    self.Pros[action].surplus.append([cpu_sur, ram_sur, storage_sur])
                else:
                    self.Pros[action].endtime.append([end_time, end1])
                    # self.vm_time[action] =
                    sur = self.Pros[action].surplus[n - 2]
                    self.Pros[action].surplus.append(sur)

        self.task_exec.append(strategy)

        if not self.is_done():       # 未结束运行
            # runtime = [self.Tasks[i].runtime for i in self.ready_task]
            # index = runtime.index(max(runtime))
            # self.task = self.ready_task[index]    # 选择执行时间最长的任务
            self.task = self.scheduling_order[0]
        return last_makespan, self.vm_time[action] - vm_time

    # 返回任务开始执行的时间
    def cal_start(self, action, est):
        if len(self.Pros[action].tasks) == 1:    # 当前只有 self.task 被调度到action中
            return est
        n = len(self.Pros[action].tasks)

        end_time = self.Pros[action].endtime[n-2]  # 前一阶段的结束时间
        surplu = self.Pros[action].surplus[n-2]    # 前一阶段的剩余资源量
        if est < end_time[0] and self.satisfy(surplu):    # 最早开始时间 小于end 0 且满足资源需求
            return end_time[0]     # 返回 end 0
        if est >= end_time[0] and est < end_time[1] and self.satisfy(surplu):  # end 0  <= est <  end 1 且满足资源需求
            return est             # 返回 est
        if est < end_time[1] and not self.satisfy(surplu):     # 最早开始时间 < end 1  且不满足资源需求
            return end_time[1]     # 返回 end 1
        return est           # 返回 est

    # 判断当下vm的剩余资源量是否满足任务的需求量
    def satisfy(self, surplu):
        stack = self.Tasks[self.task]
        if (stack.cpu <= surplu[0]) and (stack.ram <= surplu[1]) and (stack.storage <= surplu[2]):
            return True
        return False

    def rewards(self, action):
        # rankservice = self.Pros[action].service_rank   # 获取当前vm的配置级别
        task = self.Tasks[self.task]         # 当前的任务
        if self.canPutpro(task, action):   # 根据当前的资源需求来更新VM的配置
            k = 0
            for i in self.Pros[action].service:        #找到第一个资源充足的service
                if (task.cpu <= i.CPU) and (task.ram <= i.Ram) and (task.storage <= i.Storage):
                    self.Pros[action].service_rank = k
                    break
                k += 1
        if self.Pros[action].private_level < task.private_level:
            self.Pros[action].private_level = task.private_level
        #奖励函数的设置
        last_makespan, vm_add_time = self.time_reward(action)  # 上一阶段的完成时间 和 当前vm增加的时间
        inc_makespan = max(self.vm_time) - last_makespan  # 执行当前任务后的完成时间 - 上一阶段的完成时间
        inc_makespan = round(inc_makespan, 4)

        self.cost = self.cal_cost()
        if self.cost > self.budget:
            return -100000

        return task.runtime - inc_makespan


    def is_done(self):
        if len(self.released) == self.n_task:    # 如果以完成任务列表长度 == tasks的长度，即所有任务均已完成
            return True
        return False


    def cal_cost(self):
        cost = 0
        for i in self.Pros:
            rankservice = i.service_rank
            if len(i.tasks) > 0:
                cost += i.service[rankservice].Price * max(self.vm_time)
        return cost


    # 返回任务可以执行的时间
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

    # 是否资源充足
    def canPutpro(self, stack, action):
        rankservice = self.Pros[action].service_rank
        if rankservice == -1:  # 未进行VM配置
            return True
        elif (stack.cpu > self.Pros[action].service[rankservice].CPU) or (stack.ram > self.Pros[action].service[rankservice].Ram) or\
                (stack.storage > self.Pros[action].service[rankservice].Storage):   # 不能满足当前任务的需求 更新VM的配置
            return True
        return False


    def transfer1(self, t1, t2):
        if self.Tasks[t1] == self.workflow.head_task or self.Tasks[t2] == self.workflow.head_task:
            return 0.0

        output = self.Tasks[t1].output_files  # 输入文件
        input = self.Tasks[t2].input_files  # 输出文件
        req_files = output.keys() & input.keys()  # 输出==输入，即为需要传输的数据量
        vol = sum([output[f].size for f in list(req_files)])
        return round(vol / 12500000, 5)  # 带宽 12500000
