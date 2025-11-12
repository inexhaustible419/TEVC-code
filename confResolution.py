import copy
from Sol import BestSol

class confRes:
    def __init__(self, model, fraglink):
        self.model = model
        self.fraglink = fraglink

    def start(self):
        """
        功能:该函数的目标是从给定的弧段列表 fraglink 中选出一个不包含冲突的解，并返回一个满足冲突消解约束的弧段列表。

        主要步骤：
        排序操作：首先根据弧段的建链开始时间对弧段进行升序排序。link_st 是弧段的建链开始时间。
        冲突检查与消解：
        创建 rest_arc 作为备份，表示待处理的弧段集合。
        遍历 sort_arc 中的弧段，对于每个弧段，首先将它加入到结果列表 ls 中。
        然后，检查该弧段的冲突弧段（confArc_list）并将它们从 rest_arc 中移除，保证 ls 中的弧段不包含冲突的弧段。
        返回结果：
        创建一个 BestSol 类实例，并将冲突消解后的弧段列表（ls）保存到 best_sol_ls 中。
        计算总的链路时长 sum_lt 和链路数量 sum_ln。
        最终返回无冲突的弧段列表 ls。

        """
        ls = []
        sort_arc = sorted(self.fraglink, key=lambda x: self.model.Arc_list[x].link_st)
        rest_arc = copy.copy(sort_arc)
        for arc in sort_arc:
            if arc in rest_arc:
                ls.append(arc)
                for conf_arc_id in self.model.Arc_list[arc].confArc_list:
                    if conf_arc_id in rest_arc:
                        rest_arc.remove(conf_arc_id)
        sol = BestSol()
        sol.best_sol_ls = ls
        for i in sol.best_sol_ls:
            sol.sum_lt += self.model.Arc_list[i].link_time
        sol.sum_ln = len(sol.best_sol_ls)
        # op = Output(self.model, sol, 'ip')
        # op.output()
        return ls
