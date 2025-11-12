import copy
import numpy as np
import pandas as pd
import time
import csv


class Check:
    def __init__(self, model, ls):
        self.model = model
        self.arc_list = ls

    def c_find_conf(self, arc_list):
        sort_arc = sorted(copy.copy(arc_list), key=lambda x: self.model.Arc_list[x].link_st)
        num = 0
        for i in range(len(sort_arc)):
            a1 = self.model.Arc_list[sort_arc[i]]
            for j in range(i+1, len(sort_arc)):
                a2 = self.model.Arc_list[sort_arc[j]]
                # 同一地面站
                if a1.ground == a2.ground:
                    # 若相邻不重叠跟踪弧段不满足地面站转换时间则冲突（地面站约束）
                    if a2.link_st < a1.link_et + 340:
                        num = 1
                        print('error!!!!', a1.id, a2.id, 1)
                        # break
            for j in range(i + 1, len(sort_arc)):
                a2 = self.model.Arc_list[sort_arc[j]]
                # 同一卫星
                if a1.satellite == a2.satellite:
                    # 若同一卫星两个弧段建链时长相同则冲突：
                    if a2.link_st == a1.link_st:
                        num = 1
                        print('error!!!!', a1.id, a2.id, 2)
                        # break
                    elif a2.link_st < a1.link_et:
                        # 若两个弧段有包含关系，去掉被包含的弧段??
                        if a2.link_et <= a1.link_et:
                            num = 1
                            print('error!!!!', a1.id, a2.id, 3)
                            # break
                        # 若弧段重叠不满足馈电切换时间则冲突（卫星约束）
                        elif a1.link_et-a2.link_st < 150:
                            num = 1
                            print('error!!!!', a1.id, a2.id, 4)
                            # break
                    # 若建链弧段不重叠，两个弧段之间的转换时间小于天线转换时间则冲突（卫星约束）
                    elif a2.link_st-a1.link_et < 300:
                        num = 1
                        print('error!!!!', a1.id, a2.id, 5)
                        # break
            if num == 1:
                print('error!!!!')
                # break
        if num == 0:
            print("congratulations!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def check_result(self):
        arc_list = self.arc_list
        self.c_find_conf(arc_list)
        all = [arc.id for arc in self.model.Arc_list]
        for i in arc_list:
            arc = self.model.Arc_list[i]
            all.remove(arc.id)
            for c_arc in self.model.Arc_list[arc.id].confArc_list:
                if c_arc in all:
                    all.remove(c_arc)
        if all == []:
            print('no problem')
            return 1
        else:
            print(all)


