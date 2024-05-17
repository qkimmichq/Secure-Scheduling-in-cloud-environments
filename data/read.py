import xml.etree.ElementTree as ET   #  读取XMl文件

from GRCP.model.Task import Task
from GRCP.model.Task import File
from GRCP.model.Workflow import Workflow
from GRCP.resources import dax

import numpy as np


def read_workflow(dax_filepath, wf_name):
    parser = DAXParser()
    wf = parser.parseXml(dax_filepath, wf_name)
    return wf


class DAXParser:
    def __init__(self):
        pass

    def readFiles(self, job, task):
        files = job.findall('./{http://pegasus.isi.edu/schema/DAX}uses')

        def buildFile(file):
            return File(file.attrib['file'], file.attrib['size'])

        output_files = {fl.name: fl for fl in [buildFile(file) for file in files if file.attrib['link'] == "output"]}
        input_files = {fl.name: fl for fl in [buildFile(file) for file in files if file.attrib['link'] == "input"]}
        task.output_files = output_files
        task.input_files = input_files


    def parseXml(self, filepath, wf_name):
        tree = ET.parse(filepath)
        root = tree.getroot()
        jobs = root.findall('./{http://pegasus.isi.edu/schema/DAX}job')  # findall用于指定在哪一级标签下开始遍历 job标签
        children = root.findall('./{http://pegasus.isi.edu/schema/DAX}child')  # child标签
        id2Task = dict()
        for job in jobs:
            ## build task
            id = job.attrib['id']
            task = Task(id)
            task.runtime = float(job.attrib['runtime'])
            task.times = float(job.attrib['runtime'])
            task.cpu = float(job.attrib['cpu'])
            task.ram = float(job.attrib['ram'])
            task.storage = float(job.attrib['storage'])
            self.readFiles(job, task)  #获取 job标签下 file 信息
            id2Task[task.id] = task

        for child in children:
            id = child.attrib['ref']
            parents = [id2Task[prt.attrib['ref']] for prt in
                       child.findall('./{http://pegasus.isi.edu/schema/DAX}parent')]
            child = id2Task[id]
            child.parents.update(parents)  #更新父节点
            for parent in parents:
                parent.children.add(child)  #更新子节点

        heads = [task for (name, task) in id2Task.items() if len(task.parents) == 0]


        common_head = Task("000", is_head=True)  # 设置唯一的入节点
        for head in heads:
            head.parents = set([common_head])
        common_head.children = heads


        wf = Workflow(wf_name, common_head)
        return wf


if __name__ == '__main__':
    dax_name = 'conf_100'
    dax_filepath = "{}.xml".format(dax_name)
    wf = read_workflow(dax_filepath, dax_name)
    # print(wf)
    #for i in wf.head_task.children:
        #print(i)
        #for j in i.children:
            #print(j)
        #print()
    #print('')
    cpu = []
    ram = []
    storage = []
    for i in wf.tasks:
        cpu.append(i.cpu)
        ram.append(i.ram)
        storage.append(i.storage)

        #print(i)

    # print('cpu:',cpu)
    # print('ram:',ram)
    # print('storage:',storage)
    #
    # print(max(cpu),min(cpu),np.mean(cpu))
    # print(max(ram),min(ram),np.mean(ram))
    # print(max(storage),min(storage),np.mean(storage))

    #print(wf.head_task.children)

    # import scipy.stats as stats
    # print(stats.shapiro(cpu))
    # print(stats.kstest(cpu,'norm'))
    # print(stats.normaltest(cpu))
    #
    # print(stats.shapiro(ram))
    # print(stats.kstest(ram, 'norm'))
    # print(stats.normaltest(ram))
    #
    # print(stats.shapiro(storage))
    # print(stats.kstest(storage, 'norm'))
    # print(stats.normaltest(storage))


    # cpu_0 = np.random.randint(low=1.0, high=11.0, size=997)
    # cpu_1 = [i for i in cpu_0]
    # print(cpu_1)
    #
    # ram_0 = np.random.uniform(low=0.5, high=14.5, size=997)
    # ram_1 = [round(i,2) for i in ram_0]
    # print(ram_1)
    #
    # storage_0 = np.random.uniform(low=1, high=150, size=997)
    # storage_1 = [round(i,2) for i in storage_0]
    # print(storage_1)

    cpu__1 = [9, 5, 9, 2, 7, 2, 2, 6, 10, 8, 2, 7, 5, 1, 10, 2, 2, 9, 5, 10, 4, 8, 3, 7, 9]
    ram__1 = [10.65, 8.41, 8.79, 8.63, 11.33, 3.15, 5.65, 6.83, 1.17, 6.93, 8.57, 9.36, 6.99, 5.59, 1.12, 13.14, 6.89, 10.71, 8.56, 10.97, 5.18, 10.05, 0.8, 12.56, 5.55]
    storage__1 = [18.01, 135.91, 135.01, 26.55, 18.31, 38.16, 43.39, 111.7, 139.07, 33.51, 96.7, 79.57, 64.08, 49.11, 107.72, 77.25, 138.12, 85.99, 88.47, 92.87, 123.35, 18.71, 65.92, 44.33, 119.74]



    a = np.random.randint(low=0, high=3,size=60)
    b = [i for i in a]
    print(b)

    # for i in range(100):
    #     a = np.random.randint(low=0, high=3)
    #     print(a)