import numpy as np
import math
import random
from Deep_learning.model_1.Workflow import Workflow

# 奖励值的设定

class Env:
    def __init__(self, wf, Tasks, Pros, n_agent):
        self.wf = wf                                    # 工作流--wf
        self.Tasks = Tasks                              # 任务--Tasks
        self.Pros = Pros                                # VM--Pros
        self.n_agent = n_agent
        #print(len(Pros))
        #self.budget = budget
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

        obs = []
        for i in range(self.n_agent):
            obs.append(self.observation(self.task,i))

        return obs

    def step(self, action):  # 推进一个时间步长
        # done = False
        canPut = self.canPutpro(self.Tasks[self.task])   # 获取当前任务可以调度的VM集合
        # print(action,canPut)
        if action not in canPut:  # 若动作值不在集合中
            reward = -10000
            obs = []
            for i in range(self.n_agent):
                obs.append(self.observation(self.task, i))
            done = True
            return obs, reward, done

        self.Tasks[self.task].proindex = action
        self.Pros[action].tasks.append(self.task)

        self.ready_task.remove(self.task)  # 将task移除ready_task,并添加进已完成列表，
        self.released.append(self.task)

        self.set_action()   # 将后继任务中可以开始执行的任务添加进ready_task.

        obs = []
        reward = []
        for i in range(self.n_agent):
            reward.append(self.rewards(action, i))
            obs.append(self.observation(self.task, i))


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



    def time_reward(self, action):  # action--第几个VM 返回上一阶段的完成时间 和 当前任务的执行时间
        strategy = []
        strategy.append(self.task)  # 当前的任务

        last_makespan = max(self.vm_time)  # 上一阶段的完成时间
        vm_time = self.vm_time[action]  # 上一阶段VM的时间

        exec_time = self.Tasks[self.task].runtime  # 任务的执行时间
        self.Tasks[self.task].est = self.cal_est(action)  # 任务可以执行的时间

        start_time = self.cal_start(action, self.Tasks[self.task].est)  # 返回当前任务可以开始的时间
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
            if start_time >= end1:  # 大于等于上一任务的结束时间 [start,end]
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

        return last_makespan, self.vm_time[action] - vm_time

    # 返回任务开始执行的时间
    def cal_start(self, action, est):
        if len(self.Pros[action].tasks) == 1:  # 当前只有 self.task 被调度到action中
            return est
        n = len(self.Pros[action].tasks)

        end_time = self.Pros[action].endtime[n - 2]  # 前一阶段的结束时间
        surplu = self.Pros[action].surplus[n - 2]  # 前一阶段的剩余资源量
        if est < end_time[0] and self.satisfy(surplu):  # 最早开始时间 小于end 0 且满足资源需求
            return end_time[0]  # 返回 end 0
        if est >= end_time[0] and est < end_time[1] and self.satisfy(surplu):  # end 0  <= est <  end 1 且满足资源需求
            return est  # 返回 est
        if est < end_time[1] and not self.satisfy(surplu):  # 最早开始时间 < end 1  且不满足资源需求
            return end_time[1]  # 返回 end 1
        return est  # 返回 est

    # 判断当下vm的剩余资源量是否满足任务的需求量
    def satisfy(self, surplu):
        stack = self.Tasks[self.task]
        if (stack.cpu <= surplu[0]) and (stack.ram <= surplu[1]) and (stack.storage <= surplu[2]):
            return True
        return False



    def rewards(self, action, flag):

        task = self.Tasks[self.task]
        rankservice = self.Pros[action].service_rank

        if flag ==0:
            if rankservice == -1:
                k = 0
                for i in self.Pros[action].service:
                    if (task.cpu <= i.CPU) and (task.ram <= i.Ram) and (task.storage <= i.Storage):
                        self.Pros[action].service_rank = k
                        break
                    k += 1
            last_makespan, exec_time = self.time_reward(action)  # 上一阶段的完成时间 和 当前任务的执行时间
            inc_makespan = max(self.vm_time) - last_makespan  # 执行当前任务后的完成时间 - 上一阶段的完成时间
            inc_makespan = round(inc_makespan, 4)
            return task.runtime - inc_makespan

        else:
            worst_cost = 2.6
            now_cost = self.Pros[action].service[rankservice].Price
            self.vm_cost[action] += task.runtime * now_cost
            if not self.is_done():  # 未结束运行
                self.task = random.choice(self.ready_task)

            return task.runtime * (worst_cost-now_cost)



    def cal_cost(self, pro):
        rankservice = self.Pros[pro].service_rank
        length = len(self.Pros[pro].tasks)
        if length == 0:
            return 0
        time_start = self.Pros[pro].tasks[0].ast
        time_end = 0
        for i in self.Pros[pro].tasks:
            if i.aft > time_end:
                time_end = i.aft
        return self.Pros[pro].service[rankservice].Price * math.ceil((time_end - time_start)/3600)




    def observation(self, task, flag):   # 返回 all agent state: 任务类型 + VM_time +   ;  任务类型 + VM_cost +  ; ---  done
        if flag == 0:
            return np.concatenate(([task], self.vm_time), 0)   # np.concatenate 拼接数组
        else:
            return np.concatenate(([task], self.vm_cost), 0)


    def is_done(self):
        # cost = sum(self.vm_cost)
        # if cost > self.budget:
        #     return  True
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


    def canPutpro(self, stack):    # 是否可以调度当前任务
        canPut = []
        for i in range(len(self.Pros)):
            rankservice = self.Pros[i].service_rank
            if rankservice == -1:  # 未调度任务
                canPut.append(i)
            elif (stack.cpu <= self.Pros[i].service[rankservice].CPU) and\
                    (stack.ram <= self.Pros[i].service[rankservice].Ram) and\
                    (stack.storage <= self.Pros[i].service[rankservice].Storage):  # 是否能放下
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
