class Task:
    def __init__(self,num, is_head=False):         #任务类     定义了任务的基础信息与返回的方法
        self.index = None
        self.id = num
        self.proindex = -1
        self.bestindex = -1
        self.container = None                                         
        self.ast = None
        self.aft = None
        self.private_level = None
        self.est = None
        self.lst = None

        self.ranku = None
        self.rankd = None
        self.priority = None                                  #优先级
        self.comp_cost = []
        self.avg_comp_cost = None
        self.successors = []                                  #后继任务
        self.predecessors = []                                #前驱任务

        self.times = None   #

        self.cpu = None  #cpu
        self.ram = None  #内存
        self.storage = None #磁盘容量

        self.parents = set()  # set of parents tasks
        self.children = set()  # set of children tasks
        self.runtime = None  # flops for calculating
        self.input_files = None
        self.output_files = None
        self.is_head = is_head
 
    def __str__(self):                     #该方法用于返回一个字符串，描述对象的状态。
        return str(" index: {}, TASK id: {}, parents: {}, children: {}, runtime: {}, successors: {}, predecessors: {}".format(
            self.index, self.id, self.parents, self.children, self.runtime, self.successors, self.predecessors
        ))

class File:
    def __init__(self, name, size):
        self.name = name
        self.size = size