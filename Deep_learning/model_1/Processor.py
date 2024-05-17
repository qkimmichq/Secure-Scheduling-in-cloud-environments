import random

instance = {'m1_small':[1.7,39321600,0.06],'m1_medium':[3.75,85196800,0.12],'m3_medium':[3.75,85196800,0.113],
            'm1_large':[7.5,85196800,0.24],'m3_large':[7.5,85196800,0.225],'m1_xlarge':[15,131072000,0.48],
            'm3_xlarge':[15,131072000,0.45],'m3_2xlarge':[30,131072000,0.9]}
#print(instance['m1_large'])
instance1 = {
    'm4_large':[6.5,2,8,0.1],'m5d_large':[8,2,8,0.113],'m4_xlarge':[13,4,16,0.2],
    'm5d_xlarge':[16,4,16,0.226],'m4_2xlarge':[26,8,32,0.4],'m5d_2xlarge':[31,8,31,0.452],
    'm4_4xlarge':[53.5,16,64,0.8],'m5d_4xlarge':[60,16,64,0.904]
}
instance2 = {  # CPU--ECU--RAM--Storage--Prices
    'r3_large':[2,6.5,15,32,0.166],'r3_xlarge':[4,13,30.5,80,0.333],'r3_2xlarge':[8,26,61,160,0.665],
    'r3_4xlarge':[16,52,122,320,1.33],'r3_8xlarge':[32,104,244,640,2.66]
}
class Processor:                                       #似乎是在定义处理的虚拟机
    def __init__(self,num):
        self.id = num
        self.tasks = []
        self.endtime = []    # [[end0, end1],[end1, end2], [end3, end4]]
        self.surplus = []    # 每个end区间的剩余资源量  [cpu,ram,storage]
        self.avail = 0
        self.container = []
        self.private_level = 0
        self.service = []
        self.service_rank = -1
        self.service_flag = 0

        #计算能力和资源   离散单位为0.5ECU
        # self.ECU = instance2[type][1]
        #cpu:
        self.cpu = None
        #带宽：
        self.bandwith = 12500000
        #内存
        self.ram = None
        #磁盘容量
        self.storage = None
        self.price = None

    def __str__(self):
        return str(
            " Processor id: {}, tasks: {}, avail: {}, cpu: {}, bandwith: {}, price: {}".format(
                self.id, self.tasks, self.avail, self.cpu, self.bandwith, self.price
            ))

class Service():
    def __init__(self, num):
        self.index = num
        self.CPU = None
        self.Ram = None
        self.Storage = None
        self.Price = None