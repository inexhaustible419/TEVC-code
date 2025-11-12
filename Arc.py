class Arc:
    def __init__(self):
        self.id = 0
        self.satellite = 0
        self.satellite_name = ''
        self.ground = 0
        self.ground_name = ''
        self.circle = 0
        self.link_st = 0
        self.ctrl_st = 0
        self.trace_st = 0
        self.link_et = 0
        self.ctrl_et = 0
        self.trace_et = 0
        self.priority = 0
        self.f_id = 0
        self.link_time = 0
        self.confArc_list = []
        self.f_confArc_list = []
        self.conf = 0
        self.lt_divide_conf = float('inf')   # 建链时长/冲突度
