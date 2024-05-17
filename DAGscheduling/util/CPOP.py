from DAGscheduling.model.Task import Task
from DAGscheduling.model.Processor import Processor
from queue import PriorityQueue
import statistics as stats
import numpy as np
import logging

class CPOPAlgo(object):

    def __init__(self, wf, Task1, processors):
        self.head = wf.head_task
        self.processors = processors
        self.tasks = Task1

        self.head.avg_comp_cost = 0.0
        self.head.runtime = 0.0
        #print(self.head)

        last = [i for i in Task1 if len(i.children) == 0]  # 出口任务
        # 先计算出口任务的ranku 再算其他任务
        for i in last:
            self.ranku(self.tasks.index(i))


        for i in range(len(Task1) - 1, -1, -1):
            if Task1[i] not in last:
                self.ranku(i)

        for i in range(len(self.tasks)):
                self.rankd(i)

        for i in range(len(self.tasks)):
            self.tasks[i].priority = round(self.tasks[i].rankd + self.tasks[i].ranku,3)
            #print(self.tasks[i].id, self.tasks[i].ranku, self.tasks[i].rankd, self.tasks[i].priority)

        #print(queue)
        #queue = sorted(queue, key=lambda x:x[1], reverse=True)
        #print(queue)
        CP = self.head.priority
        #print(CP)
        set_up = {self.tasks[0]}
        for i in self.tasks:
            if i.priority == CP:
                set_up.add(i)


        PCP = [0] * len(processors)
        #print(PCP)
        for t in set_up:
            for i in range(len(processors)):
                #print(PCP[i])
                PCP[i] += round(t.runtime / processors[i].cpu, 3)
        #print('PCP',PCP)
        cp_processor = PCP.index(min(PCP))
        #print('cp_processor',cp_processor)


        self.head.ast = 0
        self.head.aft = 0
        que = [self.head]
        #print(queue)
        order = []

        while len(que) != 0:
            task = que[0]
            #print('sche',task.id)
            order.append(self.tasks.index(task))

            if task in set_up:
                self.assign(self.tasks.index(task), cp_processor)
            else:
                seq = [self.eft(self.tasks.index(task), p) for p in range(len(processors))]
                p = seq.index(min(seq))
                self.assign(self.tasks.index(task), p)
            for s in task.successors:
                if None not in [p.processor for p in s.predecessors] and s not in que:
                    #print(s.id, s.priority)
                    que.append(s)
            que.remove(que[0])
            que.sort(key=lambda t: t.priority, reverse=True)


        #print('CPOPAlgo:',order)

        #
        #for i in schedul_que:
            #print(i)
            #self.schedule(i)
        makespan = self.makespan()
        print('CPOPAlgo:',makespan)
        #for i in range(10):
            #print(self.tasks[i])
        for i in range(3):
            print(self.processors[i])


    def ranku(self, t):
        if len(self.tasks[t].children) == 0:
            self.tasks[t].ranku = self.tasks[t].avg_comp_cost
        else:
            # for j in self.tasks[t].successors:
            # print(j)
            # print(j.ranku)
            # print()
            if self.head in self.tasks[t].successors:
                seq = []
            seq = [self.transfer(t, self.tasks.index(j), -1) + j.ranku for j in self.tasks[t].successors]
            # print(t,seq)
            self.tasks[t].ranku = round(self.tasks[t].avg_comp_cost + max(seq), 3)
        # print(self.tasks[t].ranku)


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


    def rankd(self,t):
        if self.tasks[t] == self.head or self.head in self.tasks[t].predecessors:
            self.tasks[t].rankd = 0.0
        else:
            seq = [(j.rankd + j.avg_comp_cost + self.transfer(self.tasks.index(j), t, -1)) for j in self.tasks[t].predecessors]
            self.tasks[t].rankd = max(seq)

    def est(self, t, p):
        if self.tasks[t] == self.head:
            return 0.0

        seq = []
        for m in self.tasks[t].predecessors:
            if p == m.processor:
                seq.append(m.aft)
            else:
                seq.append(self.transfer(self.tasks.index(m), t, p) + m.aft)

        #print('seq:', seq)
        read_time = max(seq)
        #print(read_time)
        #print(type(max([read_time, self.processors[p].avail])))
        return max([read_time, self.processors[p].avail])

    def eft(self, t, p):
        if self.tasks[t] == self.head:
            return 0.0 + self.est(t, p)
        return round(self.tasks[t].runtime / self.processors[p].cpu, 3) + self.est(t, p)

    def makespan(self):
        seq = [t.aft for t in self.tasks]
        #print(seq)
        return max(seq)

    def assign(self,i, p):
        self.processors[p].tasks.append(i)
        self.tasks[i].processor = p
        self.tasks[i].ast = self.est(i, p)
        self.tasks[i].aft = self.eft(i, p)
        self.processors[p].avail = self.tasks[i].aft
