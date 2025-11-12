class Fragment:  # 按时间划分的片段，不是圈次
    def __init__(self):
        self.id = 0
        self.Arc_list = []
        self.front_may_conf = []
        self.rear_may_conf = []
        self.max_et = 0
        self.currentSol = None
        self.newSol = None
        self.bestSol = None
