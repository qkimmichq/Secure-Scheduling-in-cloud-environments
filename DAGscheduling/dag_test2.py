import sys
import logging
import os
# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DAGscheduling.model.Task import Task
from DAGscheduling.model.Processor import Processor
from DAGscheduling.model.Workflow import Workflow
from DAGscheduling.util.HEFT import HEFTAlgo
from DAGscheduling.util.CPOP import CPOPAlgo

from DAGscheduling.resources.dax.DAX_parser import DAXParser
from DAGscheduling.resources.dax.DAX_parser import read_workflow

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dax_name = ['Montage_25', 'Montage_50', 'Montage_100', 'Montage_1000']
    dax_filepath = "./resources/dax/{}.xml".format(dax_name)
    Wf = [read_workflow("./resources/dax/{}.xml".format(i), i) for i in dax_name]

    #print(Wf)
    processors = []  # 初始化处理器
    processors.append(Processor(0))
    processors.append(Processor(1, type='m1_medium'))
    processors.append(Processor(2, type='m1_large'))

    #x = [25,50,100,1000]
    y = []

    for wf in Wf:

        id_task = dict()
        for i in wf.tasks:
            id_task[i.id] = i

        for i in wf.tasks:
            l1,l2 = list(i.parents),list(i.children)
            l1.sort(key=lambda t:t.id)
            l2.sort(key=lambda t:t.id)
            i.predecessors = l1   #前置任务 按id排序
            i.successors = l2   #后置任务 按id排序

        l3 = list(wf.head_task.children)
        l3.sort(key=lambda t: t.id)
        wf.head_task.successors = l3   # 设置头节点的后置任务

        Task_1 = list(wf.tasks)
        Task_1.sort(key=lambda t:t.id)  # 按id排序
        #print(Task_1)

        for i in Task_1:
            sum = 0.0
            for j in processors:
                a = round(i.runtime / j.cpu, 3)
                i.comp_cost.append(a)
                sum += a
            i.avg_comp_cost = round(sum / 3, 3)   #计算任务在处理器上的平均完成时间
            #print(i.comp_cost,i.avg_comp_cost)
            #print(i)
        #print(Task_1[0])

        Task_1.insert(0, wf.head_task)
    #for i in Task_1:
        #print(i)

    #HEFT = HEFTAlgo(wf, Task_1, processors)
        print('-------')
        CPOP = CPOPAlgo(wf, Task_1, processors)
        y.append(CPOP.makespan())

    print(y)
    plt.plot(dax_name, y)
    plt.show()



