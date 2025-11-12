from Sol import Sol
import multiprocessing as mp
from Config import Config
class Model:
    """
    Model 类定义了卫星-地面站链路调度问题的基本模型，包含了所需的链路信息、片段划分及其对应的参数。其主要作用是组织和管理与调度相关的数据结构，为后续的优化算法提供支持。
    1. __init__ 构造函数
    功能：初始化 Model 类的实例，并设置模型所需的基本参数。
    成员变量：
        fragment_list：用于存储不同时间片段的数据。
        Arc_list：存储所有的链路（Arc 对象），每个链路代表地面站和卫星之间的通信链路。
        fragment_span：定义了时间片段的划分。每个片段对应着一段时间范围。例如 [0, 43200, 86400, ...] 表示从第 0 秒到 12 小时、第 12 小时到 24 小时等。
        fragment_min_st_max_et：存储每个片段的最早开始时间和最晚结束时间。用于检测片段之间是否可能存在冲突。
        satellite_change_time：卫星切换时间，表示卫星在任务切换时所需的时间，单位为秒（150秒）。
        satellite_trans_time：卫星之间的转换时间，单位为秒（300秒）。
        ground_trans_time：地面站之间的转换时间，单位为秒（340秒）。
    """
    def __init__(self):
        self.fragment_list = []
        self.Arc_list = []

        # # 自动根据CPU核心数确定分组粒度
        # cpu_count = mp.cpu_count()
        # total_time = 259200  # 3天的总秒数
        # # 粗略估计每个核心处理2-3个分组
        # # target_fragments = min(12, cpu_count)  # 至少18个分组，或更多
        # target_fragments = Config('args.yaml', 'yaml').get("fragments")
        # # 计算时间间隔
        # interval = total_time // target_fragments
        # print("CPU核心数：" + str(cpu_count) + "，分组数：" + str(target_fragments) + "，时间间隔：" + str(interval))
        # # 创建时间分组点
        # self.fragment_span = [i * interval for i in range(target_fragments + 1)]
        self.target_fragments = Config('args.yaml', 'yaml').get("fragments")
        # self.fragment_span =[0, 43200, 86400, 129600, 172800, 216000, 259200]#[0,259200]#
        self.fragment_min_st_max_et = []  # 存储每个片段的最晚结束弧段的结束时间，便于找出片段之间可能产生冲突的弧段
        self.satellite_change_time = 150
        self.satellite_trans_time = 300
        self.ground_trans_time = 340
        self.last_fragment_bestsole=Sol()