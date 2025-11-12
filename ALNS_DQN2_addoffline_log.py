# 基于 ALNS_DQN.py 的最小化修改
# 主要变化：
# 1. 确保 VAE 状态表示在片段级别正确工作
# 2. 修复维度问题
# 3. 增强训练连续性
from collections import deque

from Sol import Sol, BestSol
import numpy as np
import random
import time
import copy
from IP import IP

from tqdm import tqdm
from Config import Config
config = Config('args.yaml', 'yaml')

# from state_encoder import EnhancedStateEncoder

class ALNS:
    def __init__(self, model, elite_sol, dqn_agent,
                 offline_mode=False,
                 metrics_logger=None,
                 blackboard=None):
        self.model = model
        self.best_sol = None
        self.elite_sol = elite_sol
        self.dqn_agent = dqn_agent
        from state_encoder import create_encoder
        self.state_encoder = create_encoder(self.model)
        # self.state_encoder=EnhancedStateEncoder()

        # 【新增】offline模式标记
        self.offline_mode = offline_mode
        # 【新增】记录器
        self.metrics_logger = metrics_logger
        self.reward_components = {
            "link_time": 0.0,
            "improvement": 0.0,
            "structure": 0.0,  # 即使在calculate_reward中被注释，也最好在此处定义
            "boundary": 0.0,
            "raw_link_time_change": 0.0,
            "raw_conflict_improvement": 0.0
        }
        
        # 【保持原有参数不变】
        self.epochs = config.get('epochs')
        self.q = config.get('q')
        self.phi = config.get('phi')

        self.p = 0.8
        self.d = 0.03
        self.n = 5
        self.d_weight = np.ones(5) / 5
        self.r_weight = np.ones(4) / 4
        self.d_select = np.zeros(5)
        self.r_select = np.zeros(4)
        self.d_score = np.zeros(5)
        self.r_score = np.zeros(4)
        self.history_lt = []
        self.tabu_destroy = {}
        self.tabu_insert = {}
        self.r1 = 30
        self.r2 = 20
        self.r3 = 5
        self.r4 = 1
        self.b_lt = 0

        # 【新增代码】保存blackboard的引用，并定义边界阈值
        self.blackboard = blackboard
        self.boundary_threshold = 3600  # 1小时，可调整

        # 【新增】片段级别的VAE管理
        self.fragment_vae = None
        self.current_fragment_id = None

        # 【新增】：用于条件VAE的数据收集
        self.previous_best_objective = 0

        # 【新增】缓存优化
        self._boundary_arcs_cache = {}  # 缓存边界弧段
        self._boundary_weights_cache = {}  # 缓存边界权重
        self._constraint_violations_cache = {}  # 缓存约束违反检查

        # 【新增代码】为新状态特征添加追踪器
        self.stagnation_counter = 0
        # maxlen=50 表示我们只关心最近50步的改善情况，这个值可以根据需要调整
        self.recent_improvements = deque(maxlen=50)

    # 【新增代码】在ALNS类中新增此方法
    def _update_blackboard(self, solution):
        """识别solution中的边界弧段，并更新到全局信息板"""
        if self.blackboard is None or solution is None:
            return

        boundary_arcs = []
        f_num = self.f_num
        frag_start_time = self.model.fragment_min_st_max_et[f_num][0]
        frag_end_time = self.model.fragment_min_st_max_et[f_num][1]

        for arc_id in solution.Arc_list_id:
            arc = self.model.Arc_list[arc_id]
            if (arc.link_st < frag_start_time + self.boundary_threshold) or \
                    (arc.link_et > frag_end_time - self.boundary_threshold):
                # 为了节约内存，我们只存储ID、开始和结束时间
                boundary_arcs.append(
                    {'id': arc.id, 'st': arc.link_st, 'et': arc.link_et, 'g': arc.ground, 's': arc.satellite})

        # 原子化更新
        self.blackboard[f_num] = boundary_arcs

        # 【保持所有原有的摧毁和修复操作不变】
    def const_tabu_dic(self):
        for i in range(len(self.model.Arc_list)):
            self.tabu_destroy[i] = 0
            self.tabu_insert[i] = 0

    def cal_obj(self, sol):
        for i in range(len(sol.Arc_list_id)):
            arc1 = sol.Arc_list_id[i]
            sol.link_time += self.model.Arc_list[arc1].link_time
            sol.link_num += 1
        return sol

    def initial_sol(self, fragment, init):
        sol = Sol()
        if init == 0:
            sort_arc = sorted([arc for arc in fragment.Arc_list], key=lambda x: self.model.Arc_list[x].link_time,
                              reverse=True)
        elif init == 1:
            sort_arc = sorted([arc for arc in fragment.Arc_list], key=lambda x: self.model.Arc_list[x].lt_divide_conf,
                              reverse=True)
        elif init == 2:
            sort_arc = sorted([arc for arc in fragment.Arc_list], key=lambda x: self.model.Arc_list[x].link_st)
        else:
            sort_arc = sorted([arc for arc in fragment.Arc_list], key=lambda x: self.model.Arc_list[x].conf)
        rest_arc = copy.copy(sort_arc)
        for arc in sort_arc:
            if arc in rest_arc:
                sol.Arc_list_id.append(arc)
                for conf_arc_id in self.model.Arc_list[arc].f_confArc_list:
                    if conf_arc_id in rest_arc:
                        rest_arc.remove(conf_arc_id)
        sol = self.cal_obj(sol)
        return sol

    # 【保持所有原有的摧毁和修复操作】
    def random_destroy(self, fragment, num):
        not_tabu_ls = [arc for arc in fragment.currentSol.Arc_list_id if self.tabu_destroy[arc] == 0]
        if len(not_tabu_ls) < num:
            not_tabu_ls = copy.copy(fragment.currentSol.Arc_list_id)
        sort_arc = random.sample(range(len(not_tabu_ls)), num)
        remove_list = [not_tabu_ls[i] for i in sort_arc]
        return remove_list

    def max_station_destroy(self, fragment, num):
        station_split = self.split_station(copy.copy(fragment.currentSol.Arc_list_id))
        sort_station_ln = list(station_split.keys())
        not_tabu_ls = [arc for arc in fragment.currentSol.Arc_list_id if self.tabu_destroy[arc] == 0]
        if len(not_tabu_ls) < num:
            not_tabu_ls = copy.copy(fragment.currentSol.Arc_list_id)
        s_index = sorted(not_tabu_ls, key=lambda x: sort_station_ln.index(self.model.Arc_list[x].ground), reverse=True)
        for i in range(len(s_index)):
            arc1 = s_index[i]
            for j in range(i):
                arc2 = s_index[j]
                if self.model.Arc_list[arc1].ground == self.model.Arc_list[arc2].ground:
                    if self.model.Arc_list[arc1].link_time < self.model.Arc_list[arc2].link_time:
                        s_index[i], s_index[j] = s_index[j], s_index[i]
        remove_list = s_index[: num]
        return remove_list

    def lt_destroy(self, fragment, num):
        r_lt = []
        not_tabu_ls = [arc for arc in fragment.currentSol.Arc_list_id if self.tabu_destroy[arc] == 0]
        if len(not_tabu_ls) < num:
            not_tabu_ls = copy.copy(fragment.currentSol.Arc_list_id)
        for arc in not_tabu_ls:
            r = random.random()
            r_lt.append(self.model.Arc_list[arc].link_time * (1 + r))
        s_index = sorted(range(len(not_tabu_ls)), key=lambda x: r_lt[x])
        remove_list = []
        for i in s_index[: num]:
            remove_list.append(not_tabu_ls[i])
        return remove_list

    def conf_destroy(self, fragment, num):
        not_tabu_ls = [arc for arc in fragment.currentSol.Arc_list_id if self.tabu_destroy[arc] == 0]
        if len(not_tabu_ls) < num:
            not_tabu_ls = copy.copy(fragment.currentSol.Arc_list_id)
        s_index = sorted(not_tabu_ls, key=lambda x: self.model.Arc_list[x].conf, reverse=True)
        remove_list = s_index[: num]
        return remove_list

    def cont_time_destroy(self, fragment, num):
        remove_list = []
        not_tabu_ls = [arc for arc in fragment.currentSol.Arc_list_id if self.tabu_destroy[arc] == 0]
        if len(not_tabu_ls) < num:
            not_tabu_ls = copy.copy(fragment.currentSol.Arc_list_id)
        if len(not_tabu_ls) == 0:
            print(1)
        not_tabu_ls.sort(key=lambda x: self.model.Arc_list[x].link_st)
        s = random.randint(0, len(not_tabu_ls) - 1)
        if s < len(not_tabu_ls) - 1 - num:
            for i in range(num):
                index = i + s
                remove_list.append(not_tabu_ls[index])
        else:
            for i in range(num):
                index = s - i
                remove_list.append(not_tabu_ls[index])
        return remove_list

    # 【保持所有原有的插入操作】
    def insert_process(self, arc_list, i_arc_list, lt):
        conf_list_id = []
        insert_arc = []
        for i_arc in i_arc_list:
            if i_arc not in conf_list_id and i_arc not in arc_list:
                choose = {'i': [i_arc]}
                conf_dic = {'i': []}
                conf_dic['i'].extend(self.model.Arc_list[i_arc].f_confArc_list)
                i_conf = [arc for arc in self.model.Arc_list[i_arc].f_confArc_list
                          if arc not in conf_list_id and arc in i_arc_list]
                i_conf_list = sorted(i_conf, key=lambda x: i_arc_list.index(x))
                for c in range(len(i_conf_list)):
                    arc = i_conf_list[c]
                    kk = 0
                    for k, v in conf_dic.items():
                        if arc not in v:
                            choose[k].append(arc)
                            conf_dic[k].extend(self.model.Arc_list[arc].f_confArc_list)
                            kk = 1
                            break
                        else:
                            kk = 0
                    if kk == 0:
                        choose[c] = [arc]
                        conf_dic[c] = []
                        conf_dic[c].extend(self.model.Arc_list[arc].f_confArc_list)
                t = {}
                for k, v in choose.items():
                    t[k] = sum([self.model.Arc_list[i].link_time for i in v])
                index = max(t, key=lambda x: t[x])
                if len(choose[index]) > 1:
                    insert_arc += choose[index]
                    arc_list += choose[index]
                    conf_list_id += conf_dic[index]
                    lt += t[index]
                else:
                    insert_arc.append(i_arc)
                    arc_list.append(i_arc)
                    conf_list_id += self.model.Arc_list[i_arc].f_confArc_list
                    lt += self.model.Arc_list[i_arc].link_time
            else:
                continue
        return arc_list, insert_arc, lt

    def lt_insert(self, d_arc_list_id, rest_arc_list_id, lt):
        not_tabu_insert = [arc for arc in rest_arc_list_id if self.tabu_insert[arc] == 0]
        s_arc_list = sorted(not_tabu_insert, key=lambda x: self.model.Arc_list[x].link_time, reverse=True)
        arc_list, insert_arc, lt = self.insert_process(d_arc_list_id, s_arc_list, lt)
        return arc_list, insert_arc, lt

    def lt_dev_con_insert(self, d_arc_list_id, rest_arc_list, lt):
        insert_conf_dic = {}
        not_tabu_insert = [arc for arc in rest_arc_list if self.tabu_insert[arc] == 0]
        sort_arc = sorted(not_tabu_insert, key=lambda x: self.model.Arc_list[x].link_st)
        for i in range(len(sort_arc)):
            a1 = sort_arc[i]
            if a1 not in insert_conf_dic.keys():
                insert_conf_dic[a1] = []
            for j in range(i + 1, len(sort_arc)):
                a2 = sort_arc[j]
                if self.model.Arc_list[a1].ground == self.model.Arc_list[a2].ground:
                    if self.model.Arc_list[a2].link_st < self.model.Arc_list[a1].link_et + self.model.ground_trans_time:
                        insert_conf_dic[a1].append(a2)
                        if a2 not in insert_conf_dic.keys():
                            insert_conf_dic[a2] = [a1]
                        else:
                            insert_conf_dic[a2].append(a1)
                    else:
                        break
            for j in range(i + 1, len(sort_arc)):
                a2 = sort_arc[j]
                if self.model.Arc_list[a1].satellite == self.model.Arc_list[a2].satellite:
                    if self.model.Arc_list[a2].link_st == self.model.Arc_list[a1].link_st:
                        insert_conf_dic[a1].append(a2)
                        if a2 not in insert_conf_dic.keys():
                            insert_conf_dic[a2] = [a1]
                        else:
                            insert_conf_dic[a2].append(a1)
                    elif self.model.Arc_list[a2].link_st < self.model.Arc_list[a1].link_et:
                        if self.model.Arc_list[a2].link_et <= self.model.Arc_list[a1].link_et:
                            insert_conf_dic[a1].append(a2)
                            if a2 not in insert_conf_dic.keys():
                                insert_conf_dic[a2] = [a1]
                            else:
                                insert_conf_dic[a2].append(a1)
                        elif self.model.Arc_list[a1].link_et - self.model.Arc_list[a2].link_st < self.model.satellite_change_time:
                            insert_conf_dic[a1].append(a2)
                            if a2 not in insert_conf_dic.keys():
                                insert_conf_dic[a2] = [a1]
                            else:
                                insert_conf_dic[a2].append(a1)
                    elif self.model.Arc_list[a2].link_st - self.model.Arc_list[a1].link_et < self.model.satellite_trans_time:
                        insert_conf_dic[a1].append(a2)
                        if a2 not in insert_conf_dic.keys():
                            insert_conf_dic[a2] = [a1]
                        else:
                            insert_conf_dic[a2].append(a1)
                    else:
                        break
            insert_conf_dic[a1] = list(set(self.model.Arc_list[a1].f_confArc_list))
        conf_dic = {k: sum([self.model.Arc_list[arc].link_time for arc in v]) for k, v in insert_conf_dic.items()}
        insert_ls = sorted(insert_conf_dic.items(),
                           key=lambda x: self.model.Arc_list[x[0]].link_time/conf_dic[x[0]] if conf_dic[x[0]] != 0
                           else float('inf'), reverse=True)
        insert_dic = dict(insert_ls)
        s_arc_list = list(insert_dic.keys())
        arc_list, insert_arc, lt = self.insert_process(d_arc_list_id, s_arc_list, lt)
        return arc_list, insert_arc, lt

    def split_station(self, arc_list):
        station_split = {}
        for i in arc_list:
            g_id = self.model.Arc_list[i].ground
            if g_id not in station_split.keys():
                station_split[g_id] = [i]
            else:
                station_split[g_id].append(i)
        station_ls = sorted(station_split.items(), key=lambda x: sum([self.model.Arc_list[j].link_time for j in x[1]]))
        station_split = dict(station_ls)
        for v in station_split.values():
            v.sort(key=lambda x: self.model.Arc_list[x].link_time)
        return station_split

    def min_station_insert(self, d_arc_list_id, rest_arc_list_id, lt):
        station_split = self.split_station(d_arc_list_id)
        not_tabu_insert = [arc for arc in rest_arc_list_id if self.tabu_insert[arc] == 0]
        i_station_split = self.split_station(not_tabu_insert)
        conf_list_id = []
        for g_id in i_station_split.keys():
            if g_id not in station_split.keys():
                s_arc = sorted(i_station_split[g_id], key=lambda x: self.model.Arc_list[x].link_time, reverse=True)
                k = 0
                for arc in s_arc:
                    if arc not in conf_list_id:
                        station_split[g_id] = [arc]
                        d_arc_list_id.append(arc)
                        not_tabu_insert.remove(arc)
                        k += 1
                        conf_list_id.extend(self.model.Arc_list[arc].f_confArc_list)
                        break
                if k == 0:
                    for arc in s_arc:
                        not_tabu_insert.remove(arc)
        sort_station_ln = list(station_split.keys())
        s_arc_list = sorted(copy.copy(not_tabu_insert),
                            key=lambda x: sort_station_ln.index(self.model.Arc_list[x].ground))
        for i in range(len(s_arc_list)):
            arc1 = s_arc_list[i]
            for j in range(i):
                arc2 = s_arc_list[j]
                if self.model.Arc_list[arc1].ground == self.model.Arc_list[arc2].ground:
                    if self.model.Arc_list[arc1].link_time > self.model.Arc_list[arc2].link_time:
                        s_arc_list[i], s_arc_list[j] = s_arc_list[j], s_arc_list[i]
        arc_list, insert_arc, lt = self.insert_process(d_arc_list_id, s_arc_list, lt)
        return arc_list, insert_arc, lt

    def early_time_insert(self, d_arc_list_id, rest_arc_list_id, lt):
        not_tabu_insert = [arc for arc in rest_arc_list_id if self.tabu_insert[arc] == 0]
        s_arc_list = sorted(not_tabu_insert, key=lambda x: self.model.Arc_list[x].link_st)
        arc_list, insert_arc, lt = self.insert_process(d_arc_list_id, s_arc_list, lt)
        return arc_list, insert_arc, lt

    def ip_insert(self, d_arc_list_id, rest_arc_list_id, lt):
        not_tabu_insert = [arc for arc in rest_arc_list_id if self.tabu_insert[arc] == 0]
        ip = IP(self.model, rest_arc_list_id)
        insert_arc = ip.const_ip_model()
        for arc in insert_arc:
            lt += self.model.Arc_list[arc].link_time
        arc_list = d_arc_list_id + insert_arc
        return arc_list, insert_arc, lt

    ###############################################################################################边界
    def boundary_preserving_destroy(self, fragment, num):
        """边界保护破坏：优先破坏内部弧段，保护边界区域"""
        not_tabu_ls = [arc for arc in fragment.currentSol.Arc_list_id if self.tabu_destroy[arc] == 0]
        if len(not_tabu_ls) < num:
            not_tabu_ls = copy.copy(fragment.currentSol.Arc_list_id)

        # 计算边界权重并按权重升序排序（权重低的先破坏）
        arc_weights = []
        for arc_id in not_tabu_ls:
            boundary_weight = self._calculate_boundary_weight(arc_id)
            arc_weights.append((arc_id, boundary_weight))

        # 按边界权重升序排序，优先移除内部弧段
        sorted_arcs = sorted(arc_weights, key=lambda x: x[1])
        remove_list = [arc_id for arc_id, _ in sorted_arcs[:num]]
        return remove_list
    def boundary_conflict_cleanup_destroy(self, fragment, num):
        """边界冲突清理破坏：专门清理已知的边界冲突弧段"""
        not_tabu_ls = [arc for arc in fragment.currentSol.Arc_list_id if self.tabu_destroy[arc] == 0]
        if len(not_tabu_ls) < num:
            not_tabu_ls = copy.copy(fragment.currentSol.Arc_list_id)

        # 识别边界冲突弧段
        conflict_arcs = []
        normal_arcs = []

        for arc_id in not_tabu_ls:
            if self._has_boundary_conflicts(arc_id):
                conflict_severity = self._calculate_conflict_severity(arc_id)
                conflict_arcs.append((arc_id, conflict_severity))
            else:
                normal_arcs.append(arc_id)

        # 按冲突严重度降序排序
        conflict_arcs.sort(key=lambda x: x[1], reverse=True)

        # 优先移除冲突弧段，不足时补充普通弧段
        remove_list = [arc_id for arc_id, _ in conflict_arcs[:num]]
        if len(remove_list) < num:
            remaining = num - len(remove_list)
            remove_list.extend(random.sample(normal_arcs, min(remaining, len(normal_arcs))))

        return remove_list
    def boundary_buffer_destroy(self, fragment, num):
        """边界缓冲破坏：在边界附近创造时间缓冲区"""
        not_tabu_ls = [arc for arc in fragment.currentSol.Arc_list_id if self.tabu_destroy[arc] == 0]
        if len(not_tabu_ls) < num:
            not_tabu_ls = copy.copy(fragment.currentSol.Arc_list_id)

        # 识别边界临近弧段（前后30分钟内）
        boundary_arcs = []
        inner_arcs = []

        for arc_id in not_tabu_ls:
            if self._is_near_boundary(arc_id, buffer_time=1800):  # 30分钟
                boundary_arcs.append(arc_id)
            else:
                inner_arcs.append(arc_id)

        # 优先移除边界临近弧段，为边界创造缓冲
        remove_list = []
        if len(boundary_arcs) >= num:
            remove_list = random.sample(boundary_arcs, num)
        else:
            remove_list.extend(boundary_arcs)
            remaining = num - len(boundary_arcs)
            if remaining > 0 and inner_arcs:
                remove_list.extend(random.sample(inner_arcs, min(remaining, len(inner_arcs))))

        return remove_list
    def boundary_coordinated_insert(self, d_arc_list_id, rest_arc_list_id, lt):
        """边界协调插入：插入时主动考虑跨片段协调"""
        not_tabu_insert = [arc for arc in rest_arc_list_id if self.tabu_insert[arc] == 0]

        # 按边界友好程度排序
        boundary_friendly_arcs = []
        for arc_id in not_tabu_insert:
            friendliness_score = self._calculate_boundary_friendliness(arc_id)
            boundary_friendly_arcs.append((arc_id, friendliness_score))

        # 按友好度降序排序
        s_arc_list = [arc_id for arc_id, _ in sorted(boundary_friendly_arcs, key=lambda x: x[1], reverse=True)]
        arc_list, insert_arc, lt = self.insert_process(d_arc_list_id, s_arc_list, lt)
        return arc_list, insert_arc, lt
    def boundary_risk_minimizing_insert(self, d_arc_list_id, rest_arc_list_id, lt):
        """边界风险最小插入：以最小化边界风险为目标插入"""
        not_tabu_insert = [arc for arc in rest_arc_list_id if self.tabu_insert[arc] == 0]

        # 计算插入风险
        risk_arcs = []
        for arc_id in not_tabu_insert:
            risk_increment = self._calculate_insertion_risk(arc_id, d_arc_list_id)
            risk_arcs.append((arc_id, risk_increment))

        # 按风险增量升序排序
        s_arc_list = [arc_id for arc_id, _ in sorted(risk_arcs, key=lambda x: x[1])]
        arc_list, insert_arc, lt = self.insert_process(d_arc_list_id, s_arc_list, lt)
        return arc_list, insert_arc, lt
    def boundary_time_window_insert(self, d_arc_list_id, rest_arc_list_id, lt):
        """边界时间窗口插入：基于边界时间约束的智能插入"""
        not_tabu_insert = [arc for arc in rest_arc_list_id if self.tabu_insert[arc] == 0]

        # 筛选满足边界时间约束的弧段
        safe_arcs = []
        for arc_id in not_tabu_insert:
            if self._satisfies_boundary_constraints(arc_id):
                safe_arcs.append(arc_id)

        # 如果没有安全弧段，回退到所有可用弧段
        if not safe_arcs:
            safe_arcs = not_tabu_insert

        # 按链路时间降序排序
        s_arc_list = sorted(safe_arcs, key=lambda x: self.model.Arc_list[x].link_time, reverse=True)
        arc_list, insert_arc, lt = self.insert_process(d_arc_list_id, s_arc_list, lt)
        return arc_list, insert_arc, lt

    # def _calculate_boundary_weight(self, arc_id):
    #     """计算弧段的边界权重"""
    #     arc = self.model.Arc_list[arc_id]
    #     f_id = arc.f_id
    #     weight = 0.0
    #
    #     # 检查与前一片段的边界距离
    #     if f_id > 0:
    #         prev_boundary = self.model.fragment_min_st_max_et[f_id - 1][1]
    #         time_to_prev = arc.link_st - prev_boundary
    #         if time_to_prev < 1800:  # 30分钟内
    #             weight += (1800 - time_to_prev) / 1800
    #
    #     # 检查与后一片段的边界距离
    #     if f_id < len(self.model.fragment_min_st_max_et) - 1:
    #         next_boundary = self.model.fragment_min_st_max_et[f_id + 1][0]
    #         time_to_next = next_boundary - arc.link_et
    #         if time_to_next < 1800:
    #             weight += (1800 - time_to_next) / 1800
    #
    #     return min(weight, 1.0)
    def _has_boundary_conflicts(self, arc_id):
        """检查弧段是否有边界冲突"""
        current_f_num = self.model.Arc_list[arc_id].f_id

        # 检查与前一片段的冲突
        if current_f_num > 0:
            prev_boundary_arcs = self._get_boundary_arcs(current_f_num - 1)
            for boundary_arc_id in prev_boundary_arcs:
                if self._check_constraint_violation(arc_id, boundary_arc_id):
                    return True

        # 检查与后一片段的冲突
        if current_f_num < len(self.model.fragment_min_st_max_et) - 1:
            next_boundary_arcs = self._get_boundary_arcs(current_f_num + 1)
            for boundary_arc_id in next_boundary_arcs:
                if self._check_constraint_violation(arc_id, boundary_arc_id):
                    return True

        return False
    def _calculate_conflict_severity(self, arc_id):
        """计算边界冲突严重度"""
        severity = 0
        current_f_num = self.model.Arc_list[arc_id].f_id

        # 统计冲突数量和严重程度
        if current_f_num > 0:
            prev_boundary_arcs = self._get_boundary_arcs(current_f_num - 1)
            for boundary_arc_id in prev_boundary_arcs:
                if self._check_constraint_violation(arc_id, boundary_arc_id):
                    severity += self.model.Arc_list[arc_id].link_time

        if current_f_num < len(self.model.fragment_min_st_max_et) - 1:
            next_boundary_arcs = self._get_boundary_arcs(current_f_num + 1)
            for boundary_arc_id in next_boundary_arcs:
                if self._check_constraint_violation(arc_id, boundary_arc_id):
                    severity += self.model.Arc_list[arc_id].link_time

        return severity
    def _is_near_boundary(self, arc_id, buffer_time=1800):
        """检查弧段是否临近边界"""
        return self._calculate_boundary_weight(arc_id) > 0
    def _calculate_boundary_friendliness(self, arc_id):
        """计算边界友好度"""
        # 边界权重越低越友好
        boundary_weight = self._calculate_boundary_weight(arc_id)
        friendliness = 1.0 - boundary_weight

        # 加入链路时间因子
        link_time_factor = self.model.Arc_list[arc_id].link_time / 10000
        return friendliness + link_time_factor
    def _calculate_insertion_risk(self, arc_id, current_solution):
        """计算插入弧段的边界风险增量"""
        # 简化：直接使用边界权重作为风险
        return self._calculate_boundary_weight(arc_id)
    def _satisfies_boundary_constraints(self, arc_id):
        """检查弧段是否满足边界约束"""
        return not self._has_boundary_conflicts(arc_id)
    ##################################################################################################边界结束

    def do_destroy(self, fragment, destroy_id):
        num = int(len(fragment.currentSol.Arc_list_id) * self.d)
        remove_list = []
        if destroy_id == 0:
            remove_list = self.random_destroy(fragment, num)
        elif destroy_id == 1:
            remove_list = self.max_station_destroy(fragment, num)
        elif destroy_id == 2:
            remove_list = self.lt_destroy(fragment, num)
        elif destroy_id == 3:
            remove_list = self.conf_destroy(fragment, num)
        elif destroy_id == 4:
            remove_list = self.cont_time_destroy(fragment, num)
        # #########边界
        elif destroy_id == 5:
            remove_list = self.boundary_preserving_destroy(fragment, num)
        elif destroy_id == 6:
            remove_list = self.boundary_conflict_cleanup_destroy(fragment, num)
        elif destroy_id == 7:
            remove_list = self.boundary_buffer_destroy(fragment, num)

        d_arclist_id = copy.copy(fragment.currentSol.Arc_list_id)
        remove_list = list(set(remove_list))
        lt = fragment.currentSol.link_time
        for arc in remove_list:
            d_arclist_id.remove(arc)
            lt -= self.model.Arc_list[arc].link_time
        return d_arclist_id, remove_list, lt

    def do_repair(self, fragment, repair_id, d_arc_list_id, remove_list, lt):
        new_sol = Sol()
        rest_arc_list_id = copy.copy(fragment.Arc_list)
        remove_list_id = []
        for arc in d_arc_list_id:
            remove_list_id.extend(self.model.Arc_list[arc].f_confArc_list)
            remove_list_id.append(arc)
        if len(remove_list) > 1:
            uninsert_ls = random.sample(remove_list, 1)
            for arc in uninsert_ls:
                remove_list_id.append(arc)
        remove_list_id = list(set(remove_list_id))
        insert_arc = []
        for r_arc in remove_list_id:
            if r_arc not in rest_arc_list_id:
                print(1)
            else:
                rest_arc_list_id.remove(r_arc)
        if repair_id == 0:
            new_sol.Arc_list_id, insert_arc, lt = self.lt_insert(d_arc_list_id, rest_arc_list_id, lt)
        elif repair_id == 1:
            new_sol.Arc_list_id, insert_arc, lt = self.lt_dev_con_insert(d_arc_list_id, rest_arc_list_id, lt)
        elif repair_id == 2:
            new_sol.Arc_list_id, insert_arc, lt = self.min_station_insert(d_arc_list_id, rest_arc_list_id, lt)
        elif repair_id == 3:
            new_sol.Arc_list_id, insert_arc, lt = self.early_time_insert(d_arc_list_id, rest_arc_list_id, lt)
        elif repair_id == 4:
            new_sol.Arc_list_id, insert_arc, lt = self.ip_insert(d_arc_list_id, rest_arc_list_id, lt)
        ##################边界
        elif repair_id == 5:
            new_sol.Arc_list_id, insert_arc, lt = self.boundary_coordinated_insert(d_arc_list_id, rest_arc_list_id, lt)
        elif repair_id == 6:
            new_sol.Arc_list_id, insert_arc, lt = self.boundary_risk_minimizing_insert(d_arc_list_id, rest_arc_list_id, lt)
        elif repair_id == 7:
            new_sol.Arc_list_id, insert_arc, lt = self.boundary_time_window_insert(d_arc_list_id, rest_arc_list_id, lt)

        rest_arc_list = copy.copy(fragment.Arc_list)
        after_remove_list = []
        for arc in new_sol.Arc_list_id:
            after_remove_list.extend(self.model.Arc_list[arc].f_confArc_list)
            after_remove_list.append(arc)
        after_remove_list = list(set(after_remove_list))
        for r_arc in after_remove_list:
            rest_arc_list.remove(r_arc)
        if len(rest_arc_list) == 0:
            new_sol.link_time = lt
            new_sol.link_num = len(new_sol.Arc_list_id)
            return new_sol, insert_arc
        else:
            tabu_insert_arc = sorted(rest_arc_list, key=lambda x: self.model.Arc_list[x].link_time, reverse=True)
            new_sol.Arc_list_id, insert_arc1, lt1 = self.insert_process(new_sol.Arc_list_id, tabu_insert_arc, lt)
            new_sol.link_time = lt1
            new_sol.link_num = len(new_sol.Arc_list_id)
            return new_sol, insert_arc + insert_arc1

    def reset_score(self):
        self.d_select = np.zeros(5)
        self.d_score = np.zeros(5)
        self.r_select = np.zeros(4)
        self.r_score = np.zeros(4)

    def update_weight(self):
        d_score_sum = sum(self.d_score[i] for i in range(self.d_score.shape[0]))
        r_score_sum = sum(self.r_score[i] for i in range(self.r_score.shape[0]))
        for i in range(self.d_weight.shape[0]):
            self.d_weight[i] = (1 - self.p) * self.d_weight[i] + self.p * self.d_score[i] / d_score_sum
        for i in range(self.r_weight.shape[0]):
            self.r_weight[i] = (1 - self.p) * self.r_weight[i] + self.p * self.r_score[i] / r_score_sum

    def sol_may_conf(self, f_num):
        mayConf = []
        for i in self.best_sol.Arc_list_id:
            arc = self.model.Arc_list[i]
            if f_num != 0 and arc.link_st < self.model.fragment_min_st_max_et[f_num - 1][1]:
                mayConf.append(i)
            elif f_num != len(self.model.fragment_min_st_max_et) - 1 and arc.link_et > self.model.fragment_min_st_max_et[f_num + 1][0]:
                mayConf.append(i)
        return mayConf

    def _get_boundary_arcs(self, f_num, boundary_size=20):
        """获取片段边界附近的弧段 - 优化缓存版本"""
        if f_num in self._boundary_arcs_cache:
            return self._boundary_arcs_cache[f_num]

        if f_num < 0 or f_num >= len(self.model.fragment_list):
            return []

        fragment_arcs = self.model.fragment_list[f_num].Arc_list
        if not fragment_arcs:
            return []

        sorted_arcs = sorted(fragment_arcs, key=lambda x: self.model.Arc_list[x].link_st)
        boundary_arcs = list(set(sorted_arcs[:boundary_size] + sorted_arcs[-boundary_size:]))

        # 缓存结果
        self._boundary_arcs_cache[f_num] = boundary_arcs
        return boundary_arcs

    def _check_constraint_violation(self, arc1_id, arc2_id):
        """检查两个弧段是否违反约束 - 优化缓存版本"""
        # 使用有序的key避免重复计算
        key = (min(arc1_id, arc2_id), max(arc1_id, arc2_id))
        if key in self._constraint_violations_cache:
            return self._constraint_violations_cache[key]

        a1 = self.model.Arc_list[arc1_id]
        a2 = self.model.Arc_list[arc2_id]
        violation = False

        # 地面站约束 (340s)
        if a1.ground == a2.ground:
            if abs(a2.link_st - a1.link_et) < 340 or abs(a1.link_st - a2.link_et) < 340:
                violation = True

        # 卫星约束
        if not violation and a1.satellite == a2.satellite:
            if a2.link_st == a1.link_st:  # 同时开始
                violation = True
            elif min(a1.link_et, a2.link_et) > max(a1.link_st, a2.link_st):  # 有重叠
                violation = True
            elif abs(a2.link_st - a1.link_et) < 300 or abs(a1.link_st - a2.link_et) < 300:  # 转换时间不足
                violation = True

        # 缓存结果
        self._constraint_violations_cache[key] = violation
        return violation

    def _calculate_boundary_weight(self, arc_id):
        """计算弧段的边界权重 - 优化缓存版本"""
        if arc_id in self._boundary_weights_cache:
            return self._boundary_weights_cache[arc_id]

        arc = self.model.Arc_list[arc_id]
        f_id = arc.f_id
        weight = 0.0

        # 检查与前一片段的边界距离
        if f_id > 0:
            prev_boundary = self.model.fragment_min_st_max_et[f_id - 1][1]
            time_to_prev = arc.link_st - prev_boundary
            if time_to_prev < 1800:  # 30分钟内
                weight += (1800 - time_to_prev) / 1800

        # 检查与后一片段的边界距离
        if f_id < len(self.model.fragment_min_st_max_et) - 1:
            next_boundary = self.model.fragment_min_st_max_et[f_id + 1][0]
            time_to_next = next_boundary - arc.link_et
            if time_to_next < 1800:
                weight += (1800 - time_to_next) / 1800

        weight = min(weight, 1.0)
        # 缓存结果
        self._boundary_weights_cache[arc_id] = weight
        return weight

    def _calculate_boundary_info_batch(self, sol, current_f_num):
        """批量计算边界相关信息，避免重复计算"""
        external_conflicts = 0
        boundary_risk_sum = 0

        if not sol.Arc_list_id:
            return external_conflicts, 0.0

        # 获取相邻片段的边界弧段（缓存）
        prev_boundary_arcs = self._get_boundary_arcs(current_f_num - 1) if current_f_num > 0 else []
        next_boundary_arcs = self._get_boundary_arcs(current_f_num + 1) if current_f_num < len(
            self.model.fragment_min_st_max_et) - 1 else []

        for arc_id in sol.Arc_list_id:
            # 计算边界权重
            boundary_weight = self._calculate_boundary_weight(arc_id)
            boundary_risk_sum += boundary_weight

            # 检查边界冲突
            for boundary_arc_id in prev_boundary_arcs:
                if self._check_constraint_violation(arc_id, boundary_arc_id):
                    external_conflicts += 1
                    break  # 每个弧段只计算一次冲突

            if external_conflicts == 0:  # 如果前面没有冲突，检查后面
                for boundary_arc_id in next_boundary_arcs:
                    if self._check_constraint_violation(arc_id, boundary_arc_id):
                        external_conflicts += 1
                        break

        boundary_risk_density = boundary_risk_sum / len(sol.Arc_list_id)
        return external_conflicts, boundary_risk_density

    def get_state_from_solution(self, sol, f_num=None):
        """
        构建全新的、更高效的13维状态向量。
        该函数基于四大支柱逻辑：质量、结构、动态、协同。
        """
        # --- 支柱1 & 2: 质量与结构评估 ---
        link_time = sol.link_time
        link_num = sol.link_num
        current_f_num = f_num if f_num is not None else getattr(self, 'f_num', 0)

        if not sol.Arc_list_id:
            return [0.0] * 13  # 如果解为空，返回13维的零向量

        fragment = self.model.fragment_list[current_f_num]
        density = len(sol.Arc_list_id) / len(fragment.Arc_list) if fragment and len(fragment.Arc_list) > 0 else 0.0
        avg_link_time = link_time / link_num if link_num > 0 else 0.0

        ground_stations = {}
        total_conflicts = 0
        for arc_id in sol.Arc_list_id:
            arc = self.model.Arc_list[arc_id]
            gs = arc.ground
            if gs not in ground_stations:
                ground_stations[gs] = 0
            ground_stations[gs] += 1
            total_conflicts += len([c for c in arc.f_confArc_list if c in sol.Arc_list_id])

        gs_count = len(ground_stations)
        gs_link_counts = list(ground_stations.values())
        gs_mean_count = sum(gs_link_counts) / len(gs_link_counts) if gs_link_counts else 0.0
        gs_count_std = np.std(gs_link_counts) if len(gs_link_counts) > 1 else 0.0
        gs_cv = gs_count_std / gs_mean_count if gs_mean_count > 0 else 0.0
        gs_balance = 1.0 / (1.0 + gs_cv)

        conflict_ratio = total_conflicts / (link_num * 2) if link_num > 0 else 0.0

        timestamps = [self.model.Arc_list[arc_id].link_st for arc_id in sol.Arc_list_id]
        end_timestamps = [self.model.Arc_list[arc_id].link_et for arc_id in sol.Arc_list_id]
        min_time, max_time = min(timestamps), max(end_timestamps)
        time_span = max_time - min_time if min_time < max_time else 0.0
        time_density = link_time / time_span if time_span > 0 else 0.0

        time_uniformity = 0.0
        if time_span > 0:
            num_bins = 10
            bin_counts, _ = np.histogram([(st + et) / 2 for st, et in zip(timestamps, end_timestamps)],
                                         bins=num_bins, range=(min_time, max_time))
            time_mean, time_std = np.mean(bin_counts), np.std(bin_counts)
            time_uniformity = 1.0 / (1.0 + time_std / time_mean) if time_mean > 0 else 0.0

        # --- 支柱3: 搜索动态感知 ---
        stagnation = min(self.stagnation_counter / 200.0, 1.0)  # 用200步作为停滞归一化的参考最大值
        improvement_rate = sum(self.recent_improvements) / len(
            self.recent_improvements) if self.recent_improvements else 0.0

        # --- 支柱4: 全局协同意识 ---
        fragment_position = current_f_num / max(1, len(self.model.fragment_list) - 1)
        external_conflicts, _ = self._calculate_boundary_info_batch(sol, current_f_num)
        external_conflict_ratio = external_conflicts / link_num if link_num > 0 else 0.0

        boundary_density = 0.0
        boundary_arc_count = 0
        boundary_threshold = 1800  # 30分钟
        frag_start_time = self.model.fragment_min_st_max_et[current_f_num][0]
        frag_end_time = self.model.fragment_min_st_max_et[current_f_num][1]
        for arc_id in sol.Arc_list_id:
            arc = self.model.Arc_list[arc_id]
            if (arc.link_st < frag_start_time + boundary_threshold) or (
                    arc.link_et > frag_end_time - boundary_threshold):
                boundary_arc_count += 1
        boundary_density = boundary_arc_count / link_num if link_num > 0 else 0.0

        # 【新增代码】从信息板读取邻居信息，计算实时边界冲突
        left_neighbor_conflict = 0
        right_neighbor_conflict = 0
        current_f_num = f_num if f_num is not None else getattr(self, 'f_num', 0)
        if self.blackboard is not None and sol.Arc_list_id:
            # 检查左邻居
            if current_f_num > 0 and self.blackboard[current_f_num - 1]:
                neighbor_arcs = self.blackboard[current_f_num - 1]
                for arc_id in sol.Arc_list_id:
                    for n_arc in neighbor_arcs:
                        if self._check_constraint_violation(arc_id, n_arc['id']):
                            left_neighbor_conflict += 1
                            break  # 每个本地弧段只计算一次
            # 检查右邻居
            if current_f_num < len(self.model.fragment_list) - 1 and self.blackboard[current_f_num + 1]:
                neighbor_arcs = self.blackboard[current_f_num + 1]
                for arc_id in sol.Arc_list_id:
                    for n_arc in neighbor_arcs:
                        if self._check_constraint_violation(arc_id, n_arc['id']):
                            right_neighbor_conflict += 1
                            break
        # 将冲突数归一化，例如除以解中弧段数量
        norm_left_conflict = left_neighbor_conflict / len(sol.Arc_list_id) if sol.Arc_list_id else 0.0
        norm_right_conflict = right_neighbor_conflict / len(sol.Arc_list_id) if sol.Arc_list_id else 0.0

        # --- 构建最终的13维状态向量 (带注释) ---
        state = [
            link_time / 1e6,  # 0. 质量: 主要目标 (归一化)
            density,  # 1. 结构: 解的密度
            avg_link_time / 10000,  # 2. 结构: 效率 (归一化)
            gs_count / 100,  # 3. 结构: 资源使用量 (归一化)
            gs_balance,  # 4. 结构: 资源均衡性
            conflict_ratio,  # 5. 结构: 内部健康度
            time_density / 100,  # 6. 结构: 时间密度 (归一化)
            time_uniformity,  # 7. 结构: 时间均衡性
            fragment_position,  # 8. 协同: 自身定位
            external_conflict_ratio,  # 9. 协同: 被动协同 (事后)
            boundary_density,  # 10. 协同: 主动协同 (事前)
            stagnation,  # 11. 动态: 停滞感知
            improvement_rate,  # 12. 动态: 近期表现
            # 【新增最后2维】
            norm_left_conflict,  # 13. 协同: 与左邻居实时冲突度
            norm_right_conflict  # 14. 协同: 与右邻居实时冲突度
        ]

        return state

    def count_conf(self, f_num, sol):
        """优化版本的冲突计算 - 复用边界信息"""
        mayConf = []
        mayConf_sum_lt = 0

        if not sol.Arc_list_id:
            return mayConf, 0, 0

        # 复用缓存的边界弧段
        prev_boundary_arcs = self._get_boundary_arcs(f_num - 1) if f_num > 0 else []
        next_boundary_arcs = self._get_boundary_arcs(f_num + 1) if f_num < len(
            self.model.fragment_min_st_max_et) - 1 else []

        for arc_id in sol.Arc_list_id:
            has_conflict = False

            # 检查与前一片段的冲突
            for boundary_arc_id in prev_boundary_arcs:
                if self._check_constraint_violation(arc_id, boundary_arc_id):
                    has_conflict = True
                    break

            # 检查与后一片段的冲突
            if not has_conflict:
                for boundary_arc_id in next_boundary_arcs:
                    if self._check_constraint_violation(arc_id, boundary_arc_id):
                        has_conflict = True
                        break

            if has_conflict:
                mayConf.append(arc_id)
                mayConf_sum_lt += self.model.Arc_list[arc_id].link_time

        return mayConf, len(mayConf), mayConf_sum_lt

    def _calculate_solution_boundary_risk(self, solution):
        """优化版本的边界风险计算 - 复用已计算的权重"""
        if not solution.Arc_list_id:
            return 0

        total_boundary_weight = sum(
            self._calculate_boundary_weight(arc_id)
            for arc_id in solution.Arc_list_id
        )
        return total_boundary_weight / len(solution.Arc_list_id)

    # def calculate_reward(self, prev_solution, new_solution, confn, conft):
    #     """优化版本的奖励计算"""
    #     # 链接时间改进
    #     link_time_diff = new_solution.link_time - prev_solution.link_time
    #
    #     # print("冲突",confn,conft)
    #     #
    #     # # 冲突惩罚
    #     # conflict_penalty = confn * 0.55 + conft * 0.00015
    #
    #     # 归一化处理
    #     reward1_min, reward1_max = -2000, 500
    #     reward2_min, reward2_max = 0, 50
    #     reward3_min, reward3_max = 0, 3000
    #
    #     link_time_diff_norm = max(min(link_time_diff, reward1_max), reward1_min)
    #     confn_norm = max(min(confn, reward2_max), reward2_min)
    #     conft_norm = max(min(conft, reward3_max), reward3_min)
    #
    #     reward1_normalized = (link_time_diff_norm - reward1_min) / (reward1_max - reward1_min)
    #     reward2_normalized = (confn_norm - reward2_min) / (reward2_max - reward2_min)
    #     reward3_normalized = (conft_norm - reward3_min) / (reward3_max - reward3_min)
    #
    #     reward1_normalized = 2 * reward1_normalized - 1
    #
    #     # 权重
    #     weight1 = config.get('weight1') * 1.1
    #     weight2 = config.get('weight2')
    #     weight3 = config.get('weight3')
    #
    #     # 多样性和效率奖励
    #     solution_diversity = len(set(new_solution.Arc_list_id) - set(prev_solution.Arc_list_id)) / max(1,
    #                                                                                                    len(new_solution.Arc_list_id))
    #     diversity_weight = 10
    #
    #     efficiency = new_solution.link_time / max(1, len(new_solution.Arc_list_id))
    #     prev_efficiency = prev_solution.link_time / max(1, len(prev_solution.Arc_list_id))
    #     efficiency_improvement = (efficiency - prev_efficiency) / max(1, prev_efficiency)
    #     efficiency_weight = 100
    #
    #     # 【优化】边界风险惩罚 - 复用缓存计算
    #     prev_boundary_risk = self._calculate_solution_boundary_risk(prev_solution)
    #     new_boundary_risk = self._calculate_solution_boundary_risk(new_solution)
    #     boundary_risk_penalty = (new_boundary_risk - prev_boundary_risk) * 1000
    #
    #     # print('+:',reward1_normalized,'-:',reward2_normalized,'-:',reward3_normalized,'+:',solution_diversity,'+:',efficiency_improvement,'-:',boundary_risk_penalty)
    #
    #     # 计算总奖励
    #     total_reward = (weight1 * reward1_normalized -
    #                     weight2 * reward2_normalized -
    #                     weight3 * reward3_normalized +
    #                     diversity_weight * solution_diversity +
    #                     efficiency_weight * efficiency_improvement -
    #                     boundary_risk_penalty)
    #
    #     self.reward_components={
    #             "link_time": weight1 * reward1_normalized,
    #             "conflict": -weight2 * reward2_normalized - weight3 * reward3_normalized,
    #             "diversity": diversity_weight * solution_diversity,
    #             "efficiency": efficiency_weight * efficiency_improvement,
    #             "boundary": -boundary_risk_penalty
    #         }
    #
    #     return total_reward

    # ALNS_DQN2_addoffline_log.py
    # 【注意：这是最终的完整替换版本】
    def calculate_reward(self, prev_solution, new_solution, confn, conft):
        """
        一个稳定、均衡且具备协同引导的全新奖励函数。
        (优雅实现版：将边界冲突计算内联)
        """
        # --- 1. 定义各分量的权重 ---
        W_LINK_TIME = 10  # 主要目标
        # W_IMPROVEMENT = 1.5  # 对改善内部和外部冲突的综合奖励
        # W_STRUCTURE = 0.3  # 对改善解结构的奖励（效率和多样性）
        # W_BOUNDARY = 0

        # --- 2. 计算主要目标奖励 (Link Time) ---
        link_time_change = new_solution.link_time - prev_solution.link_time
        scaled_link_time_change = link_time_change / 500.0
        link_time_reward = np.clip(scaled_link_time_change, -1.0, 1.0)

        # # --- 3. 计算冲突改善奖励 (内部冲突 + 边界协调) ---
        # # 3.1 内部冲突改善
        # # 注意: 此处 prev_internal_conflicts 需您在ALNS.run循环中记录并传入
        # # 为确保代码能直接运行，我们先假设一个获取方式，您可能需要调整
        # prev_internal_conflicts = getattr(prev_solution, 'internal_conflicts', 0)  # 假设冲突数已计算并存在解对象中
        # new_internal_conflicts = confn
        # internal_conflict_improvement = prev_internal_conflicts - new_internal_conflicts
        # # 3.2 外部边界冲突改善 (内联计算)
        # prev_boundary_conflicts = 0
        # if self.blackboard is not None and prev_solution and prev_solution.Arc_list_id:
        #     f_num = self.f_num
        #     # 检查左邻居
        #     if f_num > 0 and self.blackboard.get(f_num - 1):
        #         for arc_id in prev_solution.Arc_list_id:
        #             for n_arc in self.blackboard[f_num - 1]:
        #                 if self._check_constraint_violation(arc_id, n_arc['id']):
        #                     prev_boundary_conflicts += 1
        #                     break
        #     # 检查右邻居
        #     if f_num < len(self.model.fragment_list) - 1 and self.blackboard.get(f_num + 1):
        #         for arc_id in prev_solution.Arc_list_id:
        #             for n_arc in self.blackboard[f_num + 1]:
        #                 if self._check_constraint_violation(arc_id, n_arc['id']):
        #                     prev_boundary_conflicts += 1
        #                     break
        # new_boundary_conflicts = 0
        # if self.blackboard is not None and new_solution and new_solution.Arc_list_id:
        #     f_num = self.f_num
        #     # 检查左邻居
        #     if f_num > 0 and self.blackboard.get(f_num - 1):
        #         for arc_id in new_solution.Arc_list_id:
        #             for n_arc in self.blackboard[f_num - 1]:
        #                 if self._check_constraint_violation(arc_id, n_arc['id']):
        #                     new_boundary_conflicts += 1
        #                     break
        #     # 检查右邻居
        #     if f_num < len(self.model.fragment_list) - 1 and self.blackboard.get(f_num + 1):
        #         for arc_id in new_solution.Arc_list_id:
        #             for n_arc in self.blackboard[f_num + 1]:
        #                 if self._check_constraint_violation(arc_id, n_arc['id']):
        #                     new_boundary_conflicts += 1
        #                     break
        # boundary_conflict_improvement = prev_boundary_conflicts - new_boundary_conflicts
        # # 3.3 综合冲突改善
        # total_improvement = internal_conflict_improvement + boundary_conflict_improvement
        # scaled_improvement = total_improvement / 5.0  # 用5个冲突作为典型变化值进行缩放
        # improvement_reward = np.clip(scaled_improvement, -2.0, 2.0)
        #
        # #【优化】边界风险惩罚 - 复用缓存计算
        # prev_boundary_risk = self._calculate_solution_boundary_risk(prev_solution)
        # new_boundary_risk = self._calculate_solution_boundary_risk(new_solution)
        # boundary_risk_penalty = -(new_boundary_risk - prev_boundary_risk) * 1000

        # # --- 4. 计算解结构奖励 (效率 + 多样性) ---
        # prev_efficiency = prev_solution.link_time / max(1, len(prev_solution.Arc_list_id))
        # new_efficiency = new_solution.link_time / max(1, len(new_solution.Arc_list_id))
        # efficiency_change = new_efficiency - prev_efficiency
        # scaled_efficiency_change = efficiency_change / 100.0
        #
        # diversity = len(set(new_solution.Arc_list_id) - set(prev_solution.Arc_list_id)) / max(1,
        #                                                                                       len(new_solution.Arc_list_id))
        #
        # structure_reward = np.clip(scaled_efficiency_change + diversity, -0.5, 0.5)

        # --- 5. 计算总奖励 ---
        total_reward = (W_LINK_TIME * link_time_reward
                        # + W_IMPROVEMENT * improvement_reward
                        # + W_STRUCTURE * structure_reward
                        # + W_BOUNDARY*boundary_risk_penalty
                        )

        # 用于日志记录
        self.reward_components = {
            "link_time": W_LINK_TIME * link_time_reward,
            # "improvement": W_IMPROVEMENT * improvement_reward,
            # "structure": W_STRUCTURE * structure_reward,
            # "boundary": -boundary_risk_penalty,
            "raw_link_time_change": link_time_change,
            # "raw_conflict_improvement": total_improvement
        }

        return total_reward

    def clear_caches(self):
        """清除缓存（在片段切换时调用）"""
        self._constraint_violations_cache.clear()
        # 边界弧段和权重缓存在整个运行期间保持有效

    # 【保持原有的主要运行逻辑，但调整以适配新架构】
    def run(self, t1, start_time, init, f_num, sol_ls):
        """
        保持原有运行逻辑，但增强对片段级别训练的支持
        """
        # Store fragment number for state representation
        self.f_num = f_num

        # 【保持原有初始化逻辑】
        # destroy_list = [i for i in range(len(self.d_select))] * len(self.r_select)
        # repair_list = [i for i in range(len(self.r_select))] * len(self.d_select)
        # random.shuffle(destroy_list)
        # random.shuffle(repair_list)

        self.const_tabu_dic()

        # 【新增代码】在每次运行开始时，重置追踪器
        self.stagnation_counter = 0
        self.recent_improvements.clear()

        stop = 0
        fragment = self.model.fragment_list[f_num]
        
        if len(sol_ls) == 0:
            fragment.currentSol = self.initial_sol(fragment, init)
        else:
            if init > len(sol_ls):
                print(1)
            fragment.currentSol = copy.copy(sol_ls[init])

        fragment.bestSol = copy.copy(fragment.currentSol)
        fragment.bestSol.Arc_list_id.sort(key=lambda x: self.model.Arc_list[x].link_st)
        self.elite_sol[fragment.bestSol.link_time] = fragment.bestSol
        self.history_lt.append(fragment.bestSol.link_time)
        it = 0
        
        # 根据片段位置调整初始温度
        # t = 110 if (self.f_num == 0 or self.f_num == len(self.model.fragment_min_st_max_et) - 1) else 100

        # fire_reward = 0
        # need_fire = 0

        # 【提示】在offline模式下的运行信息
        if self.offline_mode:
            print(f"Fragment {f_num}: Running in OFFLINE mode - no training will occur")
        
        # 【保持原有的训练循环逻辑】
        # for ep in tqdm(range(self.epochs), desc="Training Epoch Progress"):
        for ep in range(self.epochs):
            for i in range(self.q):
                # 获取当前解的状态
                state = self.get_state_from_solution(fragment.currentSol, f_num)
                state2 = self.state_encoder.get_enhanced_features(fragment.currentSol)
                state.extend(state2)
                t0 = time.time()
                fire_reward = 0
                need_fire = 0

                # # 使用DQN选择摧毁和修复算子
                # destroy_id, repair_id = self.dqn_agent.act(state)
                # 计算温度（随搜索进度衰减）
                progress = ep / self.epochs
                temperature = 2.0 * (1 - progress) + 0.1  # 从2.0衰减到0.1
                # 使用Softmax
                destroy_id, repair_id = self.dqn_agent.act(state, temperature)
                
                # 执行摧毁操作
                d_arclist, remove_list, lt = self.do_destroy(fragment, destroy_id)
                # 执行修复操作
                fragment.newSol, insert_list = self.do_repair(fragment, repair_id, d_arclist, remove_list, lt)

                conf, confn, conft = self.count_conf(f_num, sol=fragment.newSol)

                improvement = fragment.newSol.link_time - fragment.currentSol.link_time
                is_accepted = improvement > 0
                is_best_improved = False

                # 【新增代码】更新追踪器的核心逻辑
                self.recent_improvements.append(1 if is_accepted else 0)

                # 【修改】只在非offline模式下进行训练
                if not self.offline_mode:
                    # 计算奖励
                    reward = self.calculate_reward(fragment.currentSol, fragment.newSol, confn, conft)
                    done = False

                # # ✅
                # improvement=fragment.newSol.link_time - fragment.currentSol.link_time
                # is_accepted = improvement > 0
                # is_best_improved = False

                # 【保持原有的接受准则和更新逻辑】
                if improvement > 0:

                    fragment.currentSol = copy.copy(fragment.newSol)
                    # 【新增】：添加改进的解到条件VAE
                    if hasattr(self.dqn_agent, 'vae_manager') and self.dqn_agent.vae_manager:
                        self.dqn_agent.vae_manager.add_elite_solution_with_context(
                            solution_arcs=fragment.newSol.Arc_list_id,
                            objective_value=fragment.newSol.link_time,
                            previous_best=fragment.currentSol.link_time
                        )

                    if fragment.newSol.link_time > fragment.bestSol.link_time:

                        self.previous_best_objective = fragment.bestSol.link_time

                        it = 0
                        # ✅
                        is_best_improved = True
                        fragment.bestSol = copy.copy(fragment.newSol)
                        fragment.bestSol.Arc_list_id.sort(key=lambda x: self.model.Arc_list[x].link_st)
                        if len(self.elite_sol) < self.n:
                            self.elite_sol[fragment.bestSol.link_time] = fragment.bestSol
                        else:
                            ls = sorted(self.elite_sol)
                            del self.elite_sol[ls[0]]
                            self.elite_sol[fragment.bestSol.link_time] = fragment.bestSol
                        self.d = 0.005

                        # 【新增代码】当找到新的最优解时，更新信息板
                        self._update_blackboard(fragment.bestSol)

                    else:
                        it -= 1
                        self.d = 0.01
                else:
                    self.d = 0.03
                    it += 1

                if is_best_improved:
                    self.stagnation_counter = 0  # 如果最优解更新，停滞计数器清零
                else:
                    self.stagnation_counter += 1 # 否则，计数器加一

                # 【保持原有的早停和模拟退火逻辑】
                max_it = 100 if (self.f_num == 0 or self.f_num == len(self.model.fragment_min_st_max_et) - 1) else 20
                if it > max_it:
                    stop = 1
                    break

                fire_threshold = 12 if (self.f_num == 0 or self.f_num == len(self.model.fragment_min_st_max_et) - 1) else 10
                if it > fire_threshold:
                    self.d = 0.4
                    need_fire = 1
                    d_arclist, remove_list, lt = self.do_destroy(fragment, random.randint(0, 4))
                    fragment.newSol, insert_list = self.do_repair(fragment, 4, d_arclist, remove_list, lt)
                    fireSol = fragment.newSol
                    fire_reward = fireSol.link_time - fragment.currentSol.link_time
                    if fragment.newSol.link_time > fragment.currentSol.link_time:
                        fragment.currentSol = copy.copy(fragment.newSol)
                        if fragment.newSol.link_time > fragment.bestSol.link_time:
                            it = 0
                            fragment.bestSol = copy.copy(fragment.newSol)
                            fragment.bestSol.Arc_list_id.sort(key=lambda x: self.model.Arc_list[x].link_st)
                            if len(self.elite_sol) < self.n:
                                self.elite_sol[fragment.bestSol.link_time] = fragment.bestSol
                            else:
                                ls = sorted(self.elite_sol)
                                del self.elite_sol[ls[0]]
                                self.elite_sol[fragment.bestSol.link_time] = fragment.bestSol
                            self.d = 0.005
                        else:
                            it += 1
                            self.d = 0.01
                            
                self.history_lt.append(fragment.bestSol.link_time)

                # ✅ 【核心】记录本步数据（一次调用，全部搞定）
                if self.metrics_logger:
                    # 获取DQN指标
                    dqn_metrics = self.dqn_agent.get_metrics_for_logging()

                    # 计算解的多样性
                    diversity = len(set(fragment.newSol.Arc_list_id) - set(fragment.currentSol.Arc_list_id)) / max(1,
                                                                                                                   len(fragment.newSol.Arc_list_id))

                    # ✅ 一行记录，自动处理所有细节
                    self.metrics_logger.log_step(
                        # 基本信息
                        iteration=0,  # 由外部传入
                        it_index=0,  # 由外部传入
                        epoch=ep,
                        inner_iteration=i,

                        # 算子选择
                        destroy_id=destroy_id,
                        repair_id=repair_id,
                        is_exploration=random.random() < self.dqn_agent.epsilon,  # 近似

                        # 目标值
                        current_obj=fragment.currentSol.link_time,
                        new_obj=fragment.newSol.link_time,
                        best_obj=fragment.bestSol.link_time,
                        improvement=improvement,

                        # 状态
                        state=state,

                        # 训练状态
                        epsilon=self.dqn_agent.epsilon,
                        reward=reward,
                        is_accepted=is_accepted,
                        is_best_improved=is_best_improved,
                        loss=self.dqn_agent.loss_log,
                        td_error=getattr(self.dqn_agent, 'last_td_error', None),

                        # DQN指标（如果有）
                        q_metrics=dqn_metrics['q_metrics'] if dqn_metrics else None,
                        operator_values=dqn_metrics['operator_values'] if dqn_metrics else None,

                        # 解的规模
                        solution_sizes={
                            'current': len(fragment.currentSol.Arc_list_id),
                            'new': len(fragment.newSol.Arc_list_id),
                            'best': len(fragment.bestSol.Arc_list_id),
                            'diversity': diversity
                        },

                        # 冲突
                        conflicts={
                            'internal': len(conf),
                            'external_count': confn,
                            'external_time': conft,
                            'boundary_risk': self._calculate_solution_boundary_risk(fragment.newSol) if hasattr(self,
                                                                                                                '_calculate_solution_boundary_risk') else 0.0
                        },

                        reward_components=self.reward_components,

                        # 时间
                        step_time=time.time() - t0
                    )

                # 【修改】只在非offline模式下进行训练
                if not self.offline_mode:
                    next_state = self.get_state_from_solution(fragment.newSol, f_num)
                    next_state2 = self.state_encoder.get_enhanced_features(fragment.newSol)
                    next_state.extend(next_state2)

                    # 存储经验并训练DQN模型
                    action_id = destroy_id * self.dqn_agent.n_repair_actions + repair_id
                    self.dqn_agent.remember(state, action_id, reward, next_state, done)
                    self.dqn_agent.train_local()

                    # 更新本地操作权重
                    if i % 5 == 0:
                        self.dqn_agent.update_operator()

            # # 根据片段位置调整退火温度
            # if self.f_num == 0 or self.f_num == len(self.model.fragment_min_st_max_et) - 1:
            #     t = max(self.phi * t * 1.1, 25)
            # else:
            #     t = max(self.phi * t, 20)

            # elapsed_time = time.time() - start_time
            # elapsed_time_all = time.time() - t1
            if stop == 1:
                break
            if time.time() - t1 > config.get('set_time'):
                print(f"Time limit reached, stopping for fragment {f_num}")
                break
        # print(f"FRAGMENT {f_num} ALNS_DQN DONE USING {elapsed_time_all}!")
                
        # self.dqn_agent.plot_metrics(f_num)
        self.b_lt = fragment.bestSol.link_time
        self.best_sol = fragment.bestSol
        self.model.last_fragment_bestsole = fragment.bestSol
        may_conf = self.sol_may_conf(f_num)

        sol = BestSol()
        sol.best_sol_ls = fragment.bestSol.Arc_list_id
        for arc in sol.best_sol_ls:
            sol.sum_lt += self.model.Arc_list[arc].link_time
        sol.sum_ln = len(sol.best_sol_ls)

        return self.elite_sol, self.best_sol, init, f_num, may_conf