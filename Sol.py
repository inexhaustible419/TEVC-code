class Sol:
    '''
    该类代表一个解（Solution），用于存储任务调度的相关信息。
    构造函数 __init__
    功能：初始化解决方案对象，准备存储解的相关数据。
    成员变量：
    Arc_list_id：包含当前解所选择的弧段（链路）ID的列表。
    link_time：当前解的总建链时长，表示所有选中弧段的时间总和。
    change_times：表示解中发生了多少次馈电切换（可能用于计算电力或其他转换约束）。
    link_num：当前解中所选择的链路数量，即弧段的数量。
    c_list_id：包含发生馈电切换的弧段的列表。每个元素也是一个列表，表示与第一个弧段有馈电切换关系的其他弧段。
    ground_lt：字典，用于记录每个地面站的链路时长，可以用于分析地面站的负载。
    average_rate：当前解的均衡性，表示任务调度中的均衡程度（可能与地面站或卫星的负载均衡性有关）。
    '''
    def __init__(self):
        self.Arc_list_id = []
        self.link_time = 0
        self.change_times = 0
        self.link_num = 0
        self.c_list_id = []  # 存在馈电切换的弧段，列表中的每一个元素也是列表，每个列表中的其他弧段都与第一个弧段存在馈电切换
        self.ground_lt = {}
        self.average_rate = 0    # 均衡性


class BestSol:
    """
    best_sol_ls：最优解所包含的弧段列表。
    sum_lt：最优解的总链路时长，即该解所选择的所有弧段的总建链时长。
    sum_ct：最优解的总馈电切换次数，记录了馈电切换的总数。
    sum_ln：最优解的链路数量，即所选弧段的数量。
    """
    def __init__(self):
        self.best_sol_ls = []
        self.sum_lt = 0
        self.sum_ct = 0
        self.sum_ln = 0