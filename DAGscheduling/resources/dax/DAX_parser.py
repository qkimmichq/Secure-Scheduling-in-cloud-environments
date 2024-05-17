import xml.etree.ElementTree as ET   #  读取XMl文件

from DAGscheduling.model.Task import Task
from DAGscheduling.model.Task import File
from DAGscheduling.model.Workflow import Workflow
from DAGscheduling.resources import dax


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
        common_head.runtime = 0.0
        common_head.avg_comp_cost = 0.0
        for head in heads:
            head.parents = set([common_head])
        common_head.children = heads


        wf = Workflow(wf_name, common_head)
        return wf


if __name__ == '__main__':
    dax_name = 'CyberShake_30'
    dax_filepath = "{}.xml".format(dax_name)
    wf = read_workflow(dax_filepath, dax_name)
    print(wf)
    #for i in wf.head_task.children:
        #print(i)
        #for j in i.children:
            #print(j)
        #print()
    #print('')
    for i in wf.tasks:
        print(i)

    print('')
    for i in wf.id_to_task:
        print(i)
    print(len(wf.id_to_task))
    #print(wf.head_task.children)
