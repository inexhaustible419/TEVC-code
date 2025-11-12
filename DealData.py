from Arc import Arc
from Fragment import Fragment
import time
import copy
import csv


class DealData:
    def __init__(self, mode):
        self.model = mode

    def read_file(self, demand_file):
        """
        功能：从CSV文件中读取需求数据，构建 Arc 对象并添加到模型中的 Arc_list。
        步骤说明：
        文件读取：使用 csv.DictReader 读取 CSV 文件，该文件包含卫星-地面站链路的信息。
        地面站和卫星索引化：通过地面站和卫星的标识符，为每个链路分配一个唯一的地面站编号和卫星编号。
        构造 Arc 对象：根据读取的数据构造 Arc 对象，填充链路开始和结束时间、跟踪时间等信息，并将其加入到模型的 Arc_list 中。
        关联论文：
        该部分与论文中提到的 数据输入与模型构建 部分相符。它处理了卫星和地面站的连接数据，为后续的调度算法提供输入。
        """
        t1 = time.time()
        ground_list = []
        satellite_list = []

        with open(demand_file, 'r', encoding='gbk') as f:
            demand_reader = csv.DictReader(f)
            i = 0
            for row in demand_reader:
                arc = Arc()
                arc.id = i
                i += 1
                if row['地面站标识'] not in ground_list:
                    arc.ground = len(ground_list)
                    ground_list.append(row['地面站标识'])
                else:
                    arc.ground = ground_list.index(row['地面站标识'])
                if row['卫星标识'] not in satellite_list:
                    arc.satellite = len(satellite_list)
                    satellite_list.append(row['卫星标识'])
                else:
                    arc.satellite = satellite_list.index(row['卫星标识'])
                arc.ground_name = row['地面站标识']
                arc.satellite_name = row['卫星标识']
                arc.link_st = int(row['开始建链时间'])
                arc.link_et = int(row['结束建链时间'])
                arc.trace_st = int(row['开始跟踪时间'])
                arc.trace_et = int(row['结束跟踪时间'])
                arc.link_time = int(row['结束建链时间']) - int(row['开始建链时间'])
                self.model.Arc_list.append(arc)
        t2 = time.time()
        print('read_time', t2 - t1)

    def find_conf_arc(self):
        """
        功能：识别并计算链路之间的冲突关系。此函数根据地面站和卫星的任务时间，判断哪些链路存在冲突，并计算冲突度。
        步骤说明：
        链路排序：根据链路的开始时间对链路进行排序。
        冲突检测：通过遍历所有链路，判断同一地面站或同一卫星的链路是否存在时间上的重叠或冲突。
        冲突判断依据包括：地面站转换时间、卫星切换时间、链路重叠等。
        计算冲突度：对于每个链路，计算与其他链路冲突的总时间（冲突度），并计算链路的 link_time/conflict度（即时间占比）以评估其冲突情况。
        关联论文：
        该函数与论文中提到的 冲突检测和冲突度计算 部分一致，帮助识别链路之间的冲突，从而为后续的调度优化提供基础数据。
        """
        t1 = time.time()
        sort_arc = sorted(copy.copy(self.model.Arc_list), key=lambda x: x.link_st)
        for i in range(len(sort_arc)):
            a1 = sort_arc[i]
            for j in range(i+1, len(sort_arc)):
                a2 = sort_arc[j]
                # 同一地面站
                if a1.ground == a2.ground:
                    # 相邻跟踪弧段不满足地面站转换时间则冲突（地面站约束）
                    if a2.link_st < a1.link_et + self.model.ground_trans_time:
                        a1.confArc_list.append(a2.id)
                        a2.confArc_list.append(a1.id)
                        if a2.f_id == a1.f_id:
                            a1.f_confArc_list.append(a2.id)
                            a2.f_confArc_list.append(a1.id)
                    else:
                        break
            for j in range(i + 1, len(sort_arc)):
                a2 = sort_arc[j]
                # 同一卫星
                if a1.satellite == a2.satellite:
                    # 若同一卫星两个弧段重叠则冲突
                    if a2.link_st == a1.link_st:
                        a1.confArc_list.append(a2.id)
                        a2.confArc_list.append(a1.id)
                        if a2.f_id == a1.f_id:
                            a1.f_confArc_list.append(a2.id)
                            a2.f_confArc_list.append(a1.id)
                    elif a2.link_st < a1.link_et:
                        # 若两个弧段有包含关系则冲突
                        if a2.link_et <= a1.link_et:
                            a1.confArc_list.append(a2.id)
                            a2.confArc_list.append(a1.id)
                            if a2.f_id == a1.f_id:
                                a1.f_confArc_list.append(a2.id)
                                a2.f_confArc_list.append(a1.id)
                        # 若弧段重叠不满足馈电切换时间则冲突（卫星约束）
                        elif a1.link_et-a2.link_st < self.model.satellite_change_time:
                            a1.confArc_list.append(a2.id)
                            a2.confArc_list.append(a1.id)
                            if a2.f_id == a1.f_id:
                                a1.f_confArc_list.append(a2.id)
                                a2.f_confArc_list.append(a1.id)
                    # 若建链弧段不重叠，两个弧段之间的转换时间小于天线转换时间则冲突（卫星约束）
                    elif a2.link_st-a1.link_et < self.model.satellite_trans_time:
                        a1.confArc_list.append(a2.id)
                        a2.confArc_list.append(a1.id)
                        if a2.f_id == a1.f_id:
                            a1.f_confArc_list.append(a2.id)
                            a2.f_confArc_list.append(a1.id)
                    else:
                        break
            a1.confArc_list = list(set(a1.confArc_list))
            a1.conf = sum(self.model.Arc_list[arc_id].link_time for arc_id in a1.confArc_list)  # 冲突度按照冲突弧段的总建链时长定义
            a1.lt_divide_conf = a1.link_time / a1.conf if a1.conf != 0 else float('inf')  # 建链时长/冲突度的值计算
        t2 = time.time()
        print('find conf', t2 - t1)

    # def split_fragment(self):
    #     """
    #     功能：将链路数据按照指定的时间片段进行分割，每个片段对应一组链路，并为其计算最早开始时间和最晚结束时间。
    #
    #     步骤说明：
    #     时间片段划分：通过 self.model.fragment_span 定义的时间段，将链路分配到不同的片段中。每个片段包含一定时间范围内的链路。
    #     更新片段数据：为每个片段记录最早的开始时间和最晚的结束时间，并考虑最大转换时间（如卫星和地面站之间的转换时间）。
    #     片段列表：将生成的片段添加到 self.model.fragment_list 中。
    #
    #     关联论文：
    #     该部分对应论文中的 任务划分，将链路分配到不同的片段内，便于后续的调度和优化。
    #     """
    #     sort_arc = sorted(copy.copy(self.model.Arc_list), key=lambda x: x.link_st)
    #     max_trans = max(self.model.satellite_trans_time, self.model.ground_trans_time)
    #     for f_id in range(len(self.model.fragment_span) - 1):
    #         fragment = Fragment()
    #         for i in range(len(sort_arc)):
    #             arc = sort_arc[i]
    #             if self.model.fragment_span[f_id] <= arc.link_st < self.model.fragment_span[f_id + 1]:
    #                 fragment.Arc_list.append(arc.id)
    #                 arc.f_id = f_id
    #                 if i == len(sort_arc) - 1:
    #                     fragment.max_et = max(self.model.Arc_list[fragment.Arc_list[i]].link_et
    #                                           for i in range(len(fragment.Arc_list)))
    #                     fragment.min_st = min(self.model.Arc_list[fragment.Arc_list[i]].link_st
    #                                           for i in range(len(fragment.Arc_list))) - max_trans
    #                     self.model.fragment_min_st_max_et.append([fragment.min_st, fragment.max_et])
    #                     self.model.fragment_list.append(fragment)
    #             elif arc.link_st >= self.model.fragment_span[f_id + 1]:
    #                 fragment.max_et = max(self.model.Arc_list[fragment.Arc_list[i]].link_et
    #                                       for i in range(len(fragment.Arc_list))) + max_trans
    #                 fragment.min_st = min(self.model.Arc_list[fragment.Arc_list[i]].link_st
    #                                       for i in range(len(fragment.Arc_list))) - max_trans
    #                 self.model.fragment_min_st_max_et.append([fragment.min_st, fragment.max_et])
    #                 self.model.fragment_list.append(fragment)
    #                 break
    #             else:
    #                 continue

    def split_fragment(self):
        """按弧段数量均匀分组的新实现"""
        t1 = time.time()

        # 1. 按开始时间排序所有弧段
        sort_arc = sorted(copy.copy(self.model.Arc_list), key=lambda x: x.link_st)
        total_arcs = len(sort_arc)
        max_trans = max(self.model.satellite_trans_time, self.model.ground_trans_time)

        # 2. 计算每个片段应包含的弧段数量
        target_fragments = self.model.target_fragments
        arcs_per_fragment = total_arcs // target_fragments
        remainder = total_arcs % target_fragments

        # 3. 创建片段并分配弧段
        self.model.fragment_list = []
        self.model.fragment_min_st_max_et = []

        start_index = 0
        for frag_id in range(target_fragments):
            fragment = Fragment()
            fragment.id = frag_id

            # 计算本片段应包含的弧段数量
            end_index = start_index + arcs_per_fragment
            if frag_id < remainder:  # 前几个片段多分一个弧段
                end_index += 1

            # 分配弧段
            for i in range(start_index, end_index):
                arc = sort_arc[i]
                arc.f_id = frag_id
                fragment.Arc_list.append(arc.id)

            # 计算片段时间范围
            if fragment.Arc_list:
                fragment.min_st = min(self.model.Arc_list[arc_id].link_st
                                      for arc_id in fragment.Arc_list) - max_trans
                fragment.max_et = max(self.model.Arc_list[arc_id].link_et
                                      for arc_id in fragment.Arc_list) + max_trans
            else:
                fragment.min_st = 0
                fragment.max_et = 0

            self.model.fragment_list.append(fragment)
            self.model.fragment_min_st_max_et.append([fragment.min_st, fragment.max_et])

            start_index = end_index

        t2 = time.time()
        print(f'Fragment split time: {t2 - t1:.2f}s, Fragments: {len(self.model.fragment_list)}')
        print(f'Arc distribution:')
        for i, frag in enumerate(self.model.fragment_list):
            print(f'  Fragment {i}: {len(frag.Arc_list)} arcs')

    def run(self, demand_file):
        self.read_file(demand_file)
        self.split_fragment()
        self.find_conf_arc()
