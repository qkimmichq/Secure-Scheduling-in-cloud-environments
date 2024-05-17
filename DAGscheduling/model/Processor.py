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
class Processor:
    def __init__(self,num,type='m1_small'):
        self.id = num
        self.tasks = []
        self.avail = 0
        self.container = []

        #计算能力和资源   离散单位为0.5ECU
        #self.ECU = instance1[type][0]
        #cpu:
        self.cpu = instance[type][0]
        #带宽：
        self.bandwith = instance[type][1]
        #内存
        #self.ram =
        #磁盘容量
        #self.storage =
        self.price = instance[type][2]


    def __str__(self):
        return str(
            " Processor id: {}, tasks: {}, avail: {}, cpu: {}, bandwith: {}, price: {}".format(
                self.id, self.tasks, self.avail, self.cpu, self.bandwith, self.price
            ))

#a = random.randint(100,512)
#print(a)