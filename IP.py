import cplex
import re
from Sol import BestSol

#同IP_CON.PY
class IP:
    def __init__(self, model, arc_list):
        self.model = model
        self.arc_list = arc_list

    def split_station(self):
        station_split = {}
        for i in self.arc_list:
            arc = self.model.Arc_list[i]
            g_id = arc.ground
            if g_id not in station_split.keys():
                station_split[g_id] = [arc.id]
            else:
                station_split[g_id].append(arc.id)
        station_split = dict(sorted(station_split.items()))

        satellite_split = {}
        for i in self.arc_list:
            arc = self.model.Arc_list[i]
            s_id = arc.satellite
            if s_id not in satellite_split.keys():
                satellite_split[s_id] = [arc.id]
            else:
                satellite_split[s_id].append(arc.id)
        satellite_split = dict(sorted(satellite_split.items()))

        return station_split, satellite_split

    def const_coefficient(self):  # 系数
        station_split, satellite_split = self.split_station()
        x, y, m, n, p, q, a, b = [], [], [], [], [], [], [], []
        st_i = 0
        s_i = 0
        for value in station_split.values():  # 形如{0：[],1:[],2:[],...}
            x.append([])  # 存储决策变量x对应弧段的建链时长
            m.append([])  # 存储决策变量x对应弧段的建链开始时间
            p.append([])  # 存储决策变量x对应弧段的建链结束时间
            a.append([])  # 存储决策变量x对应弧段id
            for j in range(len(value)):
                x[st_i].append(self.model.Arc_list[value[j]].link_time)
                m[st_i].append(self.model.Arc_list[value[j]].link_st)
                p[st_i].append(self.model.Arc_list[value[j]].link_et)
                a[st_i].append(value[j])
            st_i += 1
        for value in satellite_split.values():  # {0：[],1:[],2:[],...}
            y.append([])  # 存储决策变量y对应弧段的建链时长
            n.append([])  # 存储决策变量y对应弧段的建链开始时间
            q.append([])  # 存储决策变量y对应弧段的建链结束时间
            b.append([])  # 存储决策变量y对应弧段id
            for j in range(len(value)):
                y[s_i].append(self.model.Arc_list[value[j]].link_time)
                n[s_i].append(self.model.Arc_list[value[j]].link_st)
                q[s_i].append(self.model.Arc_list[value[j]].link_et)
                b[s_i].append(value[j])
            s_i += 1
        return x, y, m, n, p, q, a, b
    @staticmethod
    def output(ip_model, results):
        result = []
        # print(ip_model.solution.get_objective_value())
        # print(ip_model.linear_constraints.get_num())
        for i in range(len(results)):
            id, value = results[i][0], results[i][1]
            if value > 0.9:
                result.append(id)
        return result

    def const_ip_model(self):
        M = 1000000
        station_split, satellite_split = self.split_station()
        ip_model = cplex.Cplex()
        ip_model.parameters.output.clonelog.set(0)  # 关闭克隆日志
        ip_model.set_log_stream(None)  # 关闭求解过程日志
        ip_model.set_results_stream(None)  # 关闭结果输出
        ip_model.set_warning_stream(None)  # 关闭警告信息
        ip_model.parameters.timelimit.set(30)
        tx, ty, sx, sy, ex, ey, ax, ay = self.const_coefficient()
        x = []
        y = []
        st_i = 0
        s_i = 0
        for value in station_split.values():
            x.append([])
            for p in range(len(value)):
                x[st_i].append(ip_model.variables.add(obj=[tx[st_i][p]], lb=[0], ub=[1], types=['B'], names=[f"x_{st_i}_{p}"]))
            st_i += 1
        # print('finish add decision x')
        for value in satellite_split.values():
            y.append([])
            for q in range(len(value)):
                y[s_i].append(ip_model.variables.add(obj=[ty[s_i][q]], lb=[0], ub=[1], types=['B'], names=[f"y_{s_i}_{q}"]))
            s_i += 1
        # print('finish add decision y')
        # 地面站约束添加
        g_rows = []
        g_rhs = []
        g_names = []
        for i in range(len(x)):
            for p in range(len(x[i])):
                for o in range(len(x[i])):
                    if p != o:
                        if sx[i][o] >= sx[i][p]:
                            g_rows.append([[f"x_{i}_{p}", f"x_{i}_{o}"], [M, M]])
                            g_rhs.append(sx[i][o] - ex[i][p] + 2 * M - 340.0)
                            g_names.append(f"station_constrain_{i}_{p}_{o}")
        g_senses = 'L' * len(g_rows)
        ip_model.linear_constraints.add(lin_expr=g_rows, senses=g_senses, rhs=g_rhs, names=g_names)
        # print('finish station')
        # 卫星约束添加
        s_rows1 = []
        s_rhs1 = []
        s_names1 = []
        s_rows2 = []
        s_rhs2 = []
        s_names2 = []
        s_rows3 = []
        s_rhs3 = []
        s_names3 = []
        for j in range(len(y)):
            for q in range(len(y[j])):
                for r in range(len(y[j])):
                    if q != r:
                        if sy[j][q] < sy[j][r] < ey[j][q] < ey[j][r]:
                            # 卫星切换约束
                            s_rows1.append([[f"y_{j}_{q}", f"y_{j}_{r}"], [M, M]])
                            s_rhs1.append(ey[j][q] - sy[j][r] + 2 * M - 150.0)
                            s_names1.append(f"station_constrain_{j}_{q}_{r}")
                        if sy[j][r] >= ey[j][q]:
                            # 卫星转换约束
                            s_rows2.append([[f"y_{j}_{q}", f"y_{j}_{r}"], [M, M]])
                            s_rhs2.append(sy[j][r] - ey[j][q] + 2 * M - 300.0)
                            s_names2.append(f"station_constrain_{j}_{q}_{r}")
                        if sy[j][r] >= sy[j][q] and ey[j][q] >= ey[j][r]:
                            # 重叠弧段约束
                            s_rows3.append([[f"y_{j}_{q}", f"y_{j}_{r}"], [M, 1 + M]])
                            s_rhs3.append(2 * M)
                            s_names3.append(f"station_constrain_{j}_{q}_{r}")
        s_senses1 = 'L' * len(s_rows1)
        ip_model.linear_constraints.add(lin_expr=s_rows1, senses=s_senses1, rhs=s_rhs1, names=s_names1)
        # print('finish s_change_constrain_add')
        s_senses2 = 'L' * len(s_rows2)
        ip_model.linear_constraints.add(lin_expr=s_rows2, senses=s_senses2, rhs=s_rhs2, names=s_names2)
        # print('finish s_trans_constrain_add')
        s_senses3 = 'L' * len(s_rows3)
        ip_model.linear_constraints.add(lin_expr=s_rows3, senses=s_senses3, rhs=s_rhs3, names=s_names3)
        # print('finish s_overlap_constrain_add')
        g_s_rows = []
        g_s_rhs = []
        g_s_names = []
        # 等式约束，将两类决策变量联系起来
        for i in range(len(ax)):
            for p in range(len(ax[i])):
                a = ax[i][p]
                choose = 0
                for j in range(len(ay)):
                    for q in range(len(ay[j])):
                        b = ay[j][q]
                        if b == a:
                            g_s_rows.append([[f"y_{j}_{q}", f"x_{i}_{p}"], [-1, 1]])
                            g_s_rhs.append(0)
                            g_s_names.append(f"link_constrain_{j}_{q}_{i}_{p}")
                            choose = 1
                            break
                    if choose == 1:
                        break
        g_s_senses = 'E' * len(g_s_rows)
        ip_model.linear_constraints.add(lin_expr=g_s_rows, senses=g_s_senses, rhs=g_s_rhs, names=g_s_names)
        ip_model.objective.set_sense(ip_model.objective.sense.maximize)
        # print('finish build model')
        ip_model.solve()

        # print("Solution num = ", ip_model.variables.get_num())
        # print("Constrain num = ", ip_model.linear_constraints.get_num())
        # print("Solution values = ", ip_model.solution.get_values())
        # print("Objective value = ", ip_model.solution.get_objective_value() / 2)

        var_names = ip_model.variables.get_names()
        var_values = ip_model.solution.get_values()
        results = []
        num = ip_model.variables.get_num()
        i = 0
        a = 0
        for var_name, var_value in zip(var_names, var_values):
            ls = re.findall(r'\d+', var_name)
            # print(f"Variable {var_name}: Value = {var_value}")
            if i < num / 2:
                results.append([ax[eval(ls[0])][eval(ls[1])], var_value])
                i += 1
                if var_value != 0.0:
                    a += 1
            else:
                break
        # print(a)
        result = IP.output(ip_model, results)

        sol = BestSol()
        sol.best_sol_ls = result
        for i in sol.best_sol_ls:
            sol.sum_lt += self.model.Arc_list[i].link_time
        sol.sum_ln = len(sol.best_sol_ls)
        # op = Output(self.model, sol, 'ip')
        # op.output()
        return result