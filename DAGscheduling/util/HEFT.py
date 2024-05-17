from DAGscheduling.model.Task import Task
from DAGscheduling.model.Processor import Processor

import statistics as stats
import numpy as np
import logging

class HEFTAlgo(object):

    def __init__(self, wf, Task1, processors):
        self.head = wf.head_task
        self.processors = processors
        self.tasks = Task1


        last = [i for i in Task1 if len(i.children) == 0] # 出口任务
        #先计算出口任务的ranku 再算其他任务
        for i in last:
            self.ranku(self.tasks.index(i))

        for i in range(len(Task1)-1, -1, -1):
            if Task1[i] not in last:
                self.ranku(i)

        que = [self.head]  #头节点加入que中
        self.head.ast = 0
        self.head.aft = 0


        while len(que) != 0 :
            #print(que[0].ranku,que[0])
            self.schedule(self.tasks.index(que[0]))      #首先调度头结点
            #print(1,que[0].successors)
            for i in que[0].successors:  #将后置任务加入到que中
                if i not in que:
                    que.append(i)
            que.remove(que[0])
            que.sort(key=lambda t:t.ranku,reverse=True) # 按ranku的值排序


        makespan = self.makespan()  # 完成时间
        print('HEFTAlgo:', makespan)

        for i in self.processors:
            print(i)



    def ranku(self,t):
        #print('---',self.tasks[t])
        if self.tasks[t] == self.head:                                  
            self.tasks[t].ranku = 0.0
            return
        if len(self.tasks[t].children) == 0:
            self.tasks[t].ranku = self.tasks[t].avg_comp_cost    #如果是出口任务 结果为平均计算时间
 
        else:
            for j in self.tasks[t].successors:
                if j.ranku == None:
                    self.ranku(self.tasks.index(j))
                #print(j)
                #print(j.ranku)
            #print()

            seq = [self.transfer(t, self.tasks.index(j), -1) + j.ranku for j in self.tasks[t].successors]
            #print(t,seq)
            self.tasks[t].ranku = round(self.tasks[t].avg_comp_cost + max(seq),3)
        #print(self.tasks[t].ranku)

    def transfer(self, t1, t2, p):                                        #计算两个任务之间的数据传输时间
        if self.tasks[t1] == self.head:
            return 0.0
        output = self.tasks[t1].output_files
        input = self.tasks[t2].input_files
        req_files = input.keys() & output.keys()
        vol = sum([int(output[f].size) for f in list(req_files)])
        if p == -1:
            um = 0.0
            for i in self.processors:
                um += vol / i.bandwith
            #print(t1,t2,'trans:',round(um / len(self.processors), 3))
            return round(um / len(self.processors), 3)
        print(t1,t2,'transfer:',round(vol / self.processors[p].bandwith, 3))
        return round(vol / self.processors[p].bandwith, 3)


    def est(self,t,p):                         #计算当前任务的最早开始时间
        if self.tasks[t] == self.head:
            return 0.0
        for m in self.tasks[t].predecessors:   # 先判断所有前置任务是否被完成
            if m.aft is None:
                print(m)
                self.schedule(m)
            #print(self.tasks[t].predecessors)
        seq = []
        for m in self.tasks[t].predecessors:
            if p == m.processor:
                seq.append(m.aft)
            else:
                seq.append(self.transfer(self.tasks.index(m),t, p) + m.aft)

        #print('seq:', seq)
        read_time = max(seq)
        #print(read_time)
        #print(type(max([read_time, self.processors[p].avail])))
        return max([read_time, self.processors[p].avail])

    def eft(self, t, p):  # 完成时间 + est
        if self.tasks[t] == self.head:
            return 0.0 + self.est(t, p)
        return round(self.tasks[t].runtime / self.processors[p].cpu, 3)  + self.est(t, p)

    def makespan(self):                       #计算当前工作流的完成时间
        seq = [t.aft for t in self.tasks]
        #print(seq)
        return max(seq)

    def assign(self,i, p):
        self.processors[p].tasks.append(i)
        self.tasks[i].processor = p
        self.tasks[i].ast = self.est(i,p)
        self.tasks[i].aft = self.eft(i,p)
        self.processors[p].avail = self.tasks[i].aft

    def schedule(self, t):    #  每个处理器上的最早完成时间----取最小----分配        按照最早完成时间=运算时间+最早开始时间进行分配
        seq = [self.eft(t, p.id) for p in self.processors]
        p = seq.index(min(seq))
        #print('seq:',seq,p)
        if self.tasks[t].processor is None:
            self.assign(t, p)
