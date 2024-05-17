from DAGscheduling.model.Task import Task
from DAGscheduling.model.Processor import Processor

import statistics as stats
import numpy as np
import logging


class BHEFTAlgo(object):

    def __init__(self, wf, Task1, processors):            
        self.head = wf.head_task
        self.processors = processors
        self.tasks = Task1


        last = [i for i in Task1 if len(i.children) == 0] # 出口任务   排序阶段
        #先计算出口任务的ranku 再算其他任务
        for i in last:
            self.ranku(self.tasks.index(i))

        for i in range(len(Task1)-1, -1, -1):
            if Task1[i] not in last:
                self.ranku(i)

        for i in Task1:
            print(i.ranku)

        sorted_task = Task1
        sorted_task.sort(key=lambda t: t.ranku, reverse=True)  # 按ranku的值排序
        
        self.head.ast = 0
        self.head.aft = 0

        self.noslots = 0
        for i in range(len(Task1)):
            #self.make_schedule(i)
            self.noslots += 1


    def make_schedule(self,task_i):
            SAB = self.cal_SAB(task_i)
            CTB = self.cal_CTB(task_i)




    def cal_SAB(self,t):
        sum_cost = 0
        sum_costk = 0
        for i in range(len(self.processors)):
            sum_cost += self.cal_p_cost(i,0)
            sum_costk += self.cal_p_cost(i,1)


    def cal_p_cost(self, i, flag):  # flag: 0-头结点  1-其他
        comput_cost = 0
        rank = -1
        #for i in




    def ranku(self,t):
        #print('---',self.tasks[t])
        if len(self.tasks[t].children) == 0:
            self.tasks[t].ranku = self.tasks[t].avg_comp_cost

        else:
            for j in self.tasks[t].successors:
                if j.ranku == None:
                    self.ranku(self.tasks.index(j))

            seq = [self.transfer(t, self.tasks.index(j), -1) + j.ranku for j in self.tasks[t].successors]
            self.tasks[t].ranku = round(self.tasks[t].avg_comp_cost + max(seq),3)



    def transfer(self, t1, t2, p):
        if self.tasks[t1] == self.head:
            return 0.0
        input = self.tasks[t1].input_files
        output = self.tasks[t2].output_files
        req_files = input.keys() & output.keys()
        vol = sum([input[f].size for f in list(req_files)])
        if p == -1:
            um = 0.0
            for i in self.processors:
                um += vol / i.bandwith
            return round(um / len(self.processors), 3)
        return round(vol / self.processors[p].bandwith, 3)