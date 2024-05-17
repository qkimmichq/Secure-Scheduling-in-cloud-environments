class Task:
    def __init__(self,num, is_head=False):
        self.id = num
        self.index = None
        self.processor = None
        self.container = None
        self.ast = None
        self.aft = None
        self.est = []
        self.eft = []
        self.ranku = None
        self.rankd = None
        self.priority = None
        self.comp_cost = []
        self.avg_comp_cost = None
        self.successors = []
        self.predecessors = []

        self.cpu = None  #cpu
        self.ram = None  #内存
        self.storage = None #磁盘容量

        self.parents = set()  # set of parents tasks
        self.children = set()  # set of children tasks
        self.runtime = None  # flops for calculating
        self.input_files = None
        self.output_files = None
        self.is_head = is_head

    def __str__(self):
        return str(" TASK id: {}, parents: {}, children: {}, runtime: {}, successors: {}, predecessors: {}".format(
            self.id, self.parents, self.children, self.runtime, self.successors, self.predecessors
        ))

class File:
    def __init__(self, name, size):
        self.name = name
        self.size = size