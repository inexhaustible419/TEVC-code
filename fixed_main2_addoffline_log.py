# 改进的修正主函数 - 恢复连续训练特性
# 主要改进：
# 1. 移除主函数中的内层it_num循环，让片段在内部独立连续训练
# 2. 恢复DQN的连续训练能力（不重复创建）
# 3. 使用修正后的可序列化模型

from multiprocessing import Manager # 【新增这行】

import copy
import time
import logging
import multiprocessing as mp
import torch

from Model import Model
from DealData import DealData
from ALNS_DQN2_addoffline_log import ALNS
import Sol
from IP import IP

from confResolution import confRes

from collections import OrderedDict

# 导入修正后的可序列化模型类
from serializable_model import (
    create_serializable_model,
    test_serializable_model
)

from FP_Growth import FpGrowth
def use_fp(f_dict, mode):
    # 建立 ID 到索引的映射
    id_to_idx = {arc.id: idx for idx, arc in enumerate(mode.Arc_list)}

    def safe_get_arc(arc_id):
        """安全获取弧段对象"""
        if arc_id in id_to_idx:
            return mode.Arc_list[id_to_idx[arc_id]]
        return None

    def split_station_simple(mode, arc_list):
        station_split = {}
        for arc_id in arc_list:
            arc = safe_get_arc(arc_id)
            if arc is None:
                continue
            g_id = arc.ground
            if g_id not in station_split:
                station_split[g_id] = [arc_id]
            else:
                station_split[g_id].append(arc_id)

        # 简单排序，避免复杂逻辑
        station_ls = sorted(station_split.items(),
                            key=lambda x: sum([safe_get_arc(j).link_time if safe_get_arc(j) else 0 for j in x[1]]))
        station_split = dict(station_ls)
        for v in station_split.values():
            v.sort(key=lambda x: safe_get_arc(x).link_time if safe_get_arc(x) else 0)

        return station_split

    # 原有逻辑，但使用安全访问
    elite_sol = {}
    for k in f_dict:
        dic = f_dict[k]
        if dic:
            ls = list(dic.keys())
            ls.sort(reverse=True)
            key_ls = ls[:10]
            k_dic = {i: dic[i] for i in key_ls}
            elite_sol.update(k_dic)

    if not elite_sol:
        return []

    elites_g_split = []
    for sol in elite_sol.values():
        if hasattr(sol, 'Arc_list_id') and sol.Arc_list_id:
            result = split_station_simple(mode, sol.Arc_list_id)
            if result:
                elites_g_split.append(result)

    if not elites_g_split:
        return []

    # 简化的FP-Growth调用
    from FP_Growth import FpGrowth
    fp = FpGrowth(elites_g_split, 4, mode)
    freq_ls = fp.start_dm()

    # 简化的冲突处理
    freq = []
    conf = []
    for ls in freq_ls:
        for i in ls:
            for j in ls:
                arc_i = safe_get_arc(i)
                if arc_i and j in arc_i.confArc_list:
                    conf.extend([i, j])
        freq.extend(ls)

    freq.sort()
    for i in freq:
        for j in freq:
            arc_i = safe_get_arc(i)
            if arc_i and j in arc_i.confArc_list:
                conf.extend([i, j])

    for arc in list(set(conf)):
        if arc in freq:
            freq.remove(arc)

    return freq


def elite_cons_sol(sol, mode, fp_ls, i):
    """
    适配序列化模型的精英解构造函数

    Args:
        sol: 原始解对象
        mode: 序列化模型对象
        fp_ls: FP-Growth挖掘出的频繁模式弧段列表
        i: 片段索引

    Returns:
        new_sol: 构造的新解
    """
    import copy
    import Sol

    # 建立 ID 到索引的映射（如果还没有的话）
    if not hasattr(mode, '_id_to_idx_cache'):
        mode._id_to_idx_cache = {arc.id: idx for idx, arc in enumerate(mode.Arc_list)}

    def safe_get_arc(arc_id):
        """安全获取弧段对象"""
        if arc_id in mode._id_to_idx_cache:
            return mode.Arc_list[mode._id_to_idx_cache[arc_id]]
        return None

    def safe_get_link_time(arc_id):
        """安全获取弧段链路时间"""
        arc = safe_get_arc(arc_id)
        return arc.link_time if arc else 0

    def safe_get_conf_list(arc_id):
        """安全获取弧段冲突列表"""
        arc = safe_get_arc(arc_id)
        return arc.confArc_list if arc else []

    # 1. 复制原解的弧段ID列表
    sol_insert = copy.copy(sol.Arc_list_id)

    # 2. 创建新解并添加FP-Growth挖掘出的弧段
    new_sol = Sol.Sol()
    new_sol.Arc_list_id.extend(fp_ls)
    new_sol.link_time = sum([safe_get_link_time(arc) for arc in fp_ls])

    # 3. 收集FP弧段的所有冲突弧段（包括自己）
    fp_conf = []
    for arc in fp_ls:
        conf_list = safe_get_conf_list(arc)
        fp_conf.extend(conf_list + [arc])
    fp_conf = list(set(fp_conf))

    # 4. 从原解中移除与FP弧段冲突的弧段
    for d_arc in fp_conf:
        if d_arc in sol_insert:
            sol_insert.remove(d_arc)

    # 5. 将剩余的非冲突弧段添加到新解中
    new_sol.Arc_list_id.extend(sol_insert)
    new_sol.link_time += sum([safe_get_link_time(arc) for arc in sol_insert])

    # 6. 找到当前片段中还可以插入的弧段
    insert_arc = copy.copy(mode.fragment_list[i].Arc_list)
    remove_arc = []

    # 收集所有已选弧段的冲突弧段（包括自己）
    for arc in new_sol.Arc_list_id:
        conf_list = safe_get_conf_list(arc)
        remove_arc.extend(conf_list + [arc])
    remove_arc = list(set(remove_arc))

    # 从可插入弧段中移除冲突弧段
    for d_arc in remove_arc:
        if d_arc in insert_arc:
            insert_arc.remove(d_arc)

    # 7. 如果还有可插入的弧段，使用IP求解最优插入
    if insert_arc:
        from IP import IP  # 假设IP类存在
        ip = IP(mode, insert_arc)
        optimal_insert_arc = ip.const_ip_model()
        if optimal_insert_arc:
            new_sol.Arc_list_id.extend(optimal_insert_arc)
            new_sol.link_time += sum([safe_get_link_time(arc) for arc in optimal_insert_arc])
        # 如果IP求解失败，可以选择简单的贪心策略
        # 按链路时间从大到小排序，选择不冲突的弧段
        sorted_arcs = sorted(insert_arc,
                             key=lambda x: safe_get_link_time(x),
                             reverse=True)
        current_arcs = set(new_sol.Arc_list_id)
        for arc_id in sorted_arcs:
            # 检查是否与已选弧段冲突
            conf_list = safe_get_conf_list(arc_id)
            if not any(conf_arc in current_arcs for conf_arc in conf_list):
                new_sol.Arc_list_id.append(arc_id)
                new_sol.link_time += safe_get_link_time(arc_id)
                current_arcs.add(arc_id)

    # 8. 设置解的链路数量
    new_sol.link_num = len(new_sol.Arc_list_id)

    return new_sol

def IP_cons_sol(mode, fp_ls, i):
    """
    适配序列化模型的IP构造解函数

    Args:
        mode: 序列化模型对象
        fp_ls: FP-Growth挖掘出的频繁模式弧段列表
        i: 片段索引

    Returns:
        sol: 构造的新解
    """
    import copy
    import Sol

    # 建立 ID 到索引的映射（如果还没有的话）
    if not hasattr(mode, '_id_to_idx_cache'):
        mode._id_to_idx_cache = {arc.id: idx for idx, arc in enumerate(mode.Arc_list)}

    def safe_get_arc(arc_id):
        """安全获取弧段对象"""
        if arc_id in mode._id_to_idx_cache:
            return mode.Arc_list[mode._id_to_idx_cache[arc_id]]
        return None

    def safe_get_link_time(arc_id):
        """安全获取弧段链路时间"""
        arc = safe_get_arc(arc_id)
        return arc.link_time if arc else 0

    def safe_get_conf_list(arc_id):
        """安全获取弧段冲突列表"""
        arc = safe_get_arc(arc_id)
        return arc.confArc_list if arc else []

    # 1. 创建新解并添加FP-Growth挖掘出的弧段
    sol = Sol.Sol()
    sol.Arc_list_id.extend(fp_ls)
    sol.link_time = sum([safe_get_link_time(arc) for arc in fp_ls])

    # 2. 找到当前片段中还可以插入的弧段
    insert_arc = copy.copy(mode.fragment_list[i].Arc_list)
    remove_arc = []

    # 3. 收集FP弧段的所有冲突弧段（包括自己）
    for arc in fp_ls:
        conf_list = safe_get_conf_list(arc)
        remove_arc.extend(conf_list + [arc])
    remove_arc = list(set(remove_arc))

    # 4. 从可插入弧段中移除冲突弧段
    for d_arc in remove_arc:
        if d_arc in insert_arc:
            insert_arc.remove(d_arc)

    # 5. 使用IP求解剩余弧段的最优插入
    if insert_arc:  # 检查是否有可插入的弧段
        try:
            ip = IP(mode, insert_arc)
            optimal_insert_arc = ip.const_ip_model()
            if optimal_insert_arc:
                sol.Arc_list_id.extend(optimal_insert_arc)
                sol.link_time += sum([safe_get_link_time(arc) for arc in optimal_insert_arc])
        except Exception as e:
            print(f"IP solving failed: {str(e)}, using greedy fallback")
            # 如果IP求解失败，使用简单的贪心策略作为备选
            sorted_arcs = sorted(insert_arc,
                                 key=lambda x: safe_get_link_time(x),
                                 reverse=True)
            current_arcs = set(sol.Arc_list_id)
            for arc_id in sorted_arcs:
                # 检查是否与已选弧段冲突
                conf_list = safe_get_conf_list(arc_id)
                if not any(conf_arc in current_arcs for conf_arc in conf_list):
                    sol.Arc_list_id.append(arc_id)
                    sol.link_time += safe_get_link_time(arc_id)
                    current_arcs.add(arc_id)

    # 6. 设置解的链路数量
    sol.link_num = len(sol.Arc_list_id)

    return sol


def use_fp_global(f_dict, mode):
    def split_station(mode,arc_list):
        station_split = {}
        for i in arc_list:
            g_id = mode.Arc_list[i].ground
            if g_id not in station_split.keys():
                station_split[g_id] = [i]
            else:
                station_split[g_id].append(i)
        station_ls = sorted(station_split.items(), key=lambda x: sum([mode.Arc_list[j].link_time for j in x[1]]))
        station_split = dict(station_ls)
        for v in station_split.values():
            v.sort(key=lambda x: mode.Arc_list[x].link_time)
        return station_split

    elite_sol = {}
    # 合并不同初始解产生的精英解，elite_sol = {lt1: sol1, lt2: sol2}
    for k in f_dict:
        dic = f_dict[k]
        ls = list(dic.keys())
        ls.sort(reverse=True)
        key_ls = ls[:3]
        # key_ls = random.sample(ls, 3)
        k_dic = {i: dic[i] for i in key_ls}
        elite_sol.update(k_dic)
    # print('elite_sol----------', sorted(list(elite_sol.keys())))
    elites_g_split = []  # 将精英解按地面站划分
    # al = ALNS(mode, {})
    for sol in elite_sol.values():
        elites_g_split.append(split_station(mode,sol.Arc_list_id))
        # 形如[{地面站1：[id1,id2,...], 地面站2：[id1,id2,...]},{地面站1：[id1,id2,...], 地面站2：[id1,id2,...]}]
    # print(len(elites_g_split), elites_g_split)
    t1 = time.time()
    # 进行数据挖掘
    fp = FpGrowth(elites_g_split, 4, mode)
    freq_ls = fp.start_dm()
    t2 = time.time()
    # print('fp-------------------------', t2 - t1)
    freq = []
    conf = []  # 记录挖掘出的弧段中冲突的数量
    for ls in freq_ls:
        for i in ls:
            for j in ls:
                if j in mode.Arc_list[i].confArc_list:
                    conf.extend([i, j])
                    # print(111, i, j, '**************')
        freq.extend(ls)
    freq.sort()
    for i in freq:
        for j in freq:
            if j in mode.Arc_list[i].confArc_list:
                conf.extend([i, j])
                # print(1222, i, j, '**************')
    for arc in list(set(conf)):
        freq.remove(arc)
    return freq


def elite_cons_sol_global(sol, mode, fp_ls, i):
    sol_insert = copy.copy(sol.Arc_list_id)
    new_sol = Sol.Sol()
    new_sol.Arc_list_id.extend(fp_ls)
    new_sol.link_time = sum([mode.Arc_list[arc].link_time for arc in fp_ls])
    fp_conf = []
    for arc in fp_ls:
        fp_conf.extend(mode.Arc_list[arc].confArc_list + [arc])
    fp_conf = list(set(fp_conf))
    for d_arc in fp_conf:
        if d_arc in sol_insert:
            sol_insert.remove(d_arc)
    new_sol.Arc_list_id.extend(sol_insert)
    new_sol.link_time += sum([mode.Arc_list[arc].link_time for arc in sol_insert])
    insert_arc = copy.copy(mode.fragment_list[i].Arc_list)
    remove_arc = []
    # 找到还可以插入的弧段, 使用cplex求出这些弧段中最优插入
    for arc in new_sol.Arc_list_id:
        remove_arc.extend(mode.Arc_list[arc].confArc_list + [arc])
    remove_arc = list(set(remove_arc))
    for d_arc in remove_arc:
        if d_arc in insert_arc:
            insert_arc.remove(d_arc)
    if insert_arc is not None:
        ip = IP(mode, insert_arc)
        insert_arc = ip.const_ip_model()
        new_sol.Arc_list_id.extend(insert_arc)
        new_sol.link_time += sum([mode.Arc_list[arc].link_time for arc in insert_arc])
    new_sol.link_num = len(new_sol.Arc_list_id)
    return new_sol

def improved_alns_worker_with_corrected_model(serializable_model, elite_sol, f_num, t1, start_time,limit_time, arc_count,
                                            it_num, initial_sol,iter, offline_mode=False,global_model_path=None, blackboard=None):
    """
    改进后的ALNS工作函数：结合序列化模型和连续训练
    关键特性：
    1. 使用修正后的可序列化模型
    2. 每个片段独立运行完整的it_num循环（连续训练）
    3. DQN不重复创建，保持训练连续性
    """
    # ✅ 1. 创建记录器（添加超参数）
    from metrics_logger import MetricsLogger
    import os
    from Config import Config
    config = Config('args.yaml', 'yaml')
    hyperparameters = {
        # DQN参数
        'learning_rate': config.get('learning_rate'),
        'gamma': config.get('gamma'),
        'epsilon_start': config.get('epsilon'),
        'epsilon_min': config.get('epsilon_min'),
        'epsilon_decay': config.get('epsilon_decay'),
        'batch_size': config.get('batch_size'),

        # ALNS搜索参数
        'iter_num': config.get('iter_num'),
        'current_iteration': iter,  # ✅ 记录当前是第几轮
        'it_num': config.get('it_num'),
        'epochs': config.get('epochs'),
        'q': config.get('q'),
        'd': config.get('d'),
        'p': config.get('p'),
        'phi': config.get('phi'),

        # 问题规模
        'total_arcs': len(serializable_model.Arc_list),
        'fragment_arcs': len(serializable_model.fragment_list[f_num].Arc_list),
        'n_fragments': len(serializable_model.fragment_list)
    }
    metrics_logger = MetricsLogger(
        fragment_id=f_num,
        iteration_id=iter,  # ✅ 传入当前迭代编号
        log_dir=os.path.join(config.get('log_dir'), 'metrics'),
        hyperparameters=hyperparameters
    )

    # try:
    print(f"Fragment {f_num} worker started with continuous training capability")
    print(f"  - Model has {len(serializable_model.Arc_list)} arcs (list structure)")
    print(f"  - Fragment {f_num} has {len(serializable_model.fragment_list[f_num].Arc_list)} arc IDs")
    print(f"  - Will run {it_num} continuous it_nums")

    # 验证Fragment中的arc_id是否有效
    fragment = serializable_model.fragment_list[f_num]
    max_arc_id = max(fragment.Arc_list) if fragment.Arc_list else -1
    if max_arc_id >= len(serializable_model.Arc_list):
        print(f"ERROR: Fragment {f_num} has invalid arc_id {max_arc_id}, max valid is {len(serializable_model.Arc_list)-1}")
        return None

    # 直接使用传入的可序列化模型
    mode = serializable_model

    # 创建持久化的DQN代理（关键：只创建一次，保持连续训练）
    agent = None
    from multi_dqn_agent_addoffline_log import ParallelDQNAgent, DuelingDQN

    device = torch.device("cuda")#torch.device("cpu") #if torch.cuda.is_available() else "cpu"

    # 1. 确定要加载的模型路径
    # 如果是第一次迭代且没有全局模型，则使用初始化的DuelingDQN
    # 否则，加载全局模型
    model_to_load_path = global_model_path
    if not os.path.exists(model_to_load_path):
        model_to_load_path = None  # 找不到全局模型，agent会自行初始化
        print(f"Fragment {f_num}: Global model not found. Will initialize a new model.")
    # # 2. 创建一个临时的DuelingDQN实例来加载权重
    global_model = DuelingDQN(8 * 8, 65).to(device)
    global_weights_for_prox = None  # 默认没有全局权重
    # if model_to_load_path:
    #     print(f"Fragment {f_num}: Loading global model from {model_to_load_path}")
    #     checkpoint = torch.load(model_to_load_path, map_location=device)
    #     global_model.load_state_dict(checkpoint['model_state'])
    #     # 提取全局模型权重，用于FedProx
    #     global_weights_for_prox = [p.data.clone() for p in global_model.parameters()]

    agent = ParallelDQNAgent(
        agent_id=f_num,
        n_destroy_actions=8,
        n_repair_actions=8,
        state_dim=65,#15,
        global_model=global_model,
        offline_mode = offline_mode,  # 【新增】传入offline模式
        metrics_logger=None, # 【新增】记录器
        mu=config.get('fedprox_mu')
    )
    # 4. 如果有全局权重，将其设置给agent
    if global_weights_for_prox:
        agent.set_global_weights(global_weights_for_prox)
        print(f"Fragment {f_num}: FedProx enabled with mu={agent.mu}")

    agent.set_arc_count(len(mode.Arc_list))
    # print(f"Fragment {f_num}: Persistent DQN agent created - will maintain training state across {it_num} it_nums")

    # 创建持久化的ALNS实例
    alns = ALNS(mode, elite_sol, agent, offline_mode=offline_mode,metrics_logger=metrics_logger, blackboard=blackboard)

    # 片段历史精英解
    fragment_elite_solutions = []
    all_results = []

    # 将Sol导入移到循环外
    import Sol
    # 【关键修改】每个片段独立运行完整的it_num循环
    # 这确保了DQN的连续性训练，而不是每次重新开始
    all_elite_sol = {f_num: {} for f_num in range(len(mode.fragment_list))}
    initial_sol_ls = initial_sol[f_num]
    for num in range(it_num):
        if num == it_num-1:
            LAHC = True
        else:
            LAHC = False
        # LAHC = False
        # print(f"Fragment {f_num} - Continuous it_num {num+1}/{it_num}")

        # 1. 基于累积的精英解生成更好的初始解
        if num > 0:
            initial_sol_ls.append(best_sol)
            # print("挖掘前执行的", num, ' ', iter)
        if num > 0 and num <= 2 : #and iter == 0
            # print("进行了挖掘",num,' ',iter)
            # print(f'------------------------------3RD STEP---------------------------------')
            # initial_sol_ls.append(best_sol)
            fp_ls = use_fp(all_elite_sol[f_num], mode)
            enhanced_sol=elite_cons_sol(best_sol, mode, fp_ls, f_num)
            if enhanced_sol.link_time > best_sol.link_time :
                initial_sol_ls.append(enhanced_sol)
                print(f"Fragment {f_num} it_num {num} - accepted improved solution: {enhanced_sol.link_time}>{best_sol.link_time}")
                if agent.epsilon < 0.5:
                    agent.epsilon = 0.5
            enhanced_sol = IP_cons_sol(mode, fp_ls, f_num)
            if enhanced_sol.link_time > best_sol.link_time :
                initial_sol_ls.append(enhanced_sol)
                print(f"Fragment {f_num} it_num {num} - accepted improved solution: {enhanced_sol.link_time}>{best_sol.link_time}")
                if agent.epsilon < 0.5:
                    agent.epsilon = 0.5

            # 按link_time降序排序，保留最优解
            initial_sol_ls = sorted(initial_sol_ls, key=lambda x: x.link_time, reverse=True)


        def smart_initial_selection():
            # 生成4种候选初始解并快速评估
            candidates = []
            for i in range(4):
                for init_type in [0, 1, 2, 3]:
                    candidate = alns.initial_sol(fragment, init_type)
                    if candidate and candidate.Arc_list_id:
                        # 简单评估：解的质量 = 目标值 + 弧段数量奖励
                        quality = candidate.link_time - len(candidate.Arc_list_id) * 10
                        candidates.append((candidate, quality, init_type))
                        print(
                            f"  - Init {init_type}: {candidate.link_time} link_time, {len(candidate.Arc_list_id)} arcs, quality: {quality}")

            if candidates:
                # 选择质量最高的候选解
                best_candidate = max(candidates, key=lambda x: x[1])
                selected, quality, selected_init = best_candidate
                print(f"  - Selected init method {selected_init} with quality {quality}")
            return selected
        # 【新增】智能初始解选择 - 只需要添加这个函数
        if num == 0 and iter == 0:
            # print(f'------------------------------1ST STEP---------------------------------')
            # 智能选择初始解（只在第一次迭代时使用）
            best_initial_sol = smart_initial_selection()
            # 第一次迭代：使用智能选择的初始解
            initial_sol_ls = [best_initial_sol]

        # 2. 运行ALNS算法（使用持续的agent，保持训练状态）
        # print(f'------------------------------2ND STEP---------------------------------')
        # print(f"Fragment {f_num}: Running ALNS with continuous DQN agent (it_num {num})")
        result = alns.run(t1, start_time, 0, f_num, initial_sol_ls,LAHC=LAHC)
        elite_sol_result, best_sol, init, f_num_result, may_conf = result
        all_elite_sol[f_num][init] = elite_sol_result

        # 3. 收集本片段的精英解（累积学习）
        current_elite_solutions = []
        for sol in elite_sol_result.values():
            if hasattr(sol, 'Arc_list_id') and sol.Arc_list_id:
                current_elite_solutions.append(sol.Arc_list_id)

        # 累积添加到片段精英解集合
        fragment_elite_solutions.extend(current_elite_solutions)

        # 存储当前迭代结果
        all_results.append({
            'iteration': num,
            'elite_sol': elite_sol_result,
            'best_sol': best_sol,
            'may_conf': may_conf,
            'elite_solutions': current_elite_solutions
        })

        print(f"Fragment {f_num} - it_num {num} completed, best: {best_sol.link_time if best_sol else 0}")
        # print(f"Fragment {f_num} - Cumulative elite solutions: {len(fragment_elite_solutions)}")

        if time.time() - t1 > limit_time:
            print(f"Time limit reached, stopping for fragment {f_num}")
            break

    # ✅ Worker结束时保存当前iter的数据（计算汇总）
    saved_file = metrics_logger.save(compute_summary=True)
    print(f"Fragment {f_num} Iter {iter}: Data saved to {saved_file}")

    if not offline_mode:
        checkpoint_path = agent.save_checkpoint()
        print(f"Fragment {f_num}: Training state saved to {checkpoint_path}")

    # 返回最终结果（最后一次迭代的结果 + 累积的精英解）
    if all_results:
        final_result = all_results[-1]
        # print(f"Fragment {f_num} - All {it_num} continuous it_nums completed successfully")
        print(f"Fragment {f_num} - Final best link_time: {final_result['best_sol'].link_time}")
        # print(f"Fragment {f_num} - Total accumulated elite solutions: {len(fragment_elite_solutions)}")

        return (final_result['elite_sol'], final_result['best_sol'],
               0, f_num, final_result['may_conf'], final_result['iteration']) #fragment_elite_solutions,
    else:
        print(f"Fragment {f_num} - No results generated")
        return (elite_sol, None, 0, f_num, [], fragment_elite_solutions)


def average_model_weights(model_paths):
    """
    读取多个模型文件的state_dict，计算它们的平均值。

    Args:
        model_paths (list): 包含模型文件路径的列表。

    Returns:
        OrderedDict: 平均后的state_dict。
    """
    if not model_paths:
        return None

    # 读取第一个模型作为基准
    base_state_dict = torch.load(model_paths[0])['model_state']
    avg_state_dict = OrderedDict()

    # 初始化平均字典
    for key in base_state_dict:
        avg_state_dict[key] = base_state_dict[key].clone().float()

    # 累加其他模型的权重
    for i in range(1, len(model_paths)):
        try:
            state_dict = torch.load(model_paths[i])['model_state']
            for key in base_state_dict:
                if key in state_dict:
                    avg_state_dict[key] += state_dict[key].float()
        except Exception as e:
            print(f"Warning: Could not load or process model {model_paths[i]}: {e}")
            # 如果某个模型加载失败，我们可以选择跳过它，或者让总数减一
            # 这里简单处理，让分母不变，相当于给了一个权重惩罚
            continue

    # 计算平均值
    num_models = len(model_paths)
    if num_models > 0:
        for key in avg_state_dict:
            avg_state_dict[key] = avg_state_dict[key] / num_models

    return avg_state_dict

def main(f):
    """
    改进的主函数：恢复连续训练特性
    关键改进：
    1. 移除内层it_num循环，让片段在内部独立连续训练
    2. 使用barrier同步各片段的迭代
    3. 保持原有的数据挖掘和冲突解决逻辑
    """
    demand_file_ls = ['data/'+config.get('data')+'.csv']
    rrrrrrrr = []
    tttttttt = []
    iiiii = 0
    iter_num = config.get('iter_num')  # 通常只需要1次外层迭代
    break_all = False
    limit_time=config.get('set_time')
    offline_mode=False

    # 设置日志
    log_dir = config.get('log_dir')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'improved_corrected_training_log.txt'),
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # logger = logging.getLogger('main_improved_corrected')

    # 配置多进程
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    print(f"Using Python multiprocessing with {mp.cpu_count()} available CPU cores", file=f)
    print(f"PyTorch version: {torch.__version__}", file=f)
    print(f"CUDA available: {torch.cuda.is_available()}", file=f)
        
    for file_num in range(len(demand_file_ls)):
        demand_file = demand_file_ls[file_num]
        print(f"Processing data file: {demand_file}", file=f)
        
        # 初始化原始模型
        print("Initializing original model...", file=f)
        mode = Model()
        deal_data = DealData(mode)
        deal_data.run(demand_file)
        it_num = config.get('it_num')

        # 创建并验证可序列化模型
        print("Creating improved corrected serializable model...", file=f)
        serializable_model = create_serializable_model(mode)
        if test_serializable_model(mode):
            print("✓ Serializable model validation passed!", file=f)
        else:
            print("✗ Serializable model validation failed!", file=f)
            # logger.error("Serializable model validation failed")
            continue
        arc_count = len(serializable_model.Arc_list)
        num_fragments = len(serializable_model.fragment_list)
        print(f"Improved corrected serializable model created successfully:", file=f)
        print(f"  - Arc count: {arc_count}", file=f)
        print(f"  - Fragment count: {num_fragments}", file=f)
        print(f"  - Will run {it_num} continuous it_nums per fragment", file=f)
        print(f"  - Will run {iter_num} continuous iter_nums per fragment", file=f)

        # 简化的外层结构（通常只需要1次迭代）
        lls = []
        t = []
        t1 = time.time()
        # --- 新增：定义全局模型的路径 ---
        #测试github上传。
        global_model_path = os.path.join(config.get('log_dir'), "trained_model", "global_model.pt")
        initial_sol = [[] for i in range(len(mode.fragment_list))]

        # 【新增代码】创建Manager和全局共享的blackboard
        manager = Manager()
        blackboard = manager.dict()
        # 初始化黑板，每个分片的值是一个空列表
        for i in range(len(mode.fragment_list)):
            blackboard[i] = []

        for iter in range(iter_num):
            # initial_sol = [[] for i in range(len(mode.fragment_list))]
            print(f'------------------------------Improved Continuous DQN-LNS start {iter}---------------------------------', file=f)
            print(f'------------------------------Improved Continuous DQN-LNS start {iter}---------------------------------')
            # 【移除内层it_num循环】直接启动持续训练的片段进程
            print("Starting continuous training with barrier synchronization...", file=f)

            max_processes = min(len(serializable_model.fragment_list), mp.cpu_count())
            result_tasks = []

            # try:
            with mp.Pool(processes=max_processes) as p:
                # 提交任务到池，每个片段将独立运行it_num次连续迭代
                for f_num in range(len(serializable_model.fragment_list)):
                    elite_sol = {}  # 每次重新开始
                    start_time = time.time()

                    print(f"Submitting continuous training task for fragment {f_num}", file=f)

                    result_tasks.append(p.apply_async(
                        improved_alns_worker_with_corrected_model,  # 改进后的工作函数
                        args=(serializable_model, elite_sol, f_num, t1, start_time,limit_time, arc_count,
                             it_num,initial_sol,iter,offline_mode,global_model_path,blackboard)  # it_num传递给worker，让其内部连续训练
                    ))

                # 收集结果
                result_ls = []
                for i, task in enumerate(result_tasks):
                    print(f"Waiting for continuous training result from fragment {i}...", file=f)
                    result = task.get(timeout=7200)  # 2小时超时，因为是连续训练
                    if result is not None:
                        result_ls.append(result)
                        print(f"✓ Successfully collected continuous training result from fragment {i}", file=f)
                    else:
                        print(f"⚠ Got None result from fragment {i}", file=f)
            
            # 处理结果
            fragment_sol = {f_num: {} for f_num in range(len(serializable_model.fragment_list))}
            may_conf_arc = {f_num: {} for f_num in range(len(serializable_model.fragment_list))}
            # # 分组外频繁模式挖掘
            # all_elite_sol = {f_num: {} for f_num in range(len(mode.fragment_list))}

            # # 收集所有进程的最终结果
            # successful_fragments = 0
            #
            # for result in result_ls:
            #     if result : #and len(result) >= 5
            #         elite_sol, best_sol, init, f_num, may_conf,num= result#[:5]
            #         fragment_sol[f_num][init] = best_sol
            #         may_conf_arc[f_num][init] = may_conf
            #         # 分组外频繁模式挖掘
            #         all_elite_sol[f_num][init] = elite_sol
            #
            #         successful_fragments += 1
            #         best_link_time = best_sol.link_time if best_sol else 0
            #         print(f"✓ Processed continuous training result by it_num {num+1} from fragment {f_num}: best_link_time = {best_link_time}", file=f)
            # print(f"Successfully processed {successful_fragments}/{num_fragments} fragments", file=f)
            #

            # 收集所有进程的最终结果
            successful_fragments = 0
            for result in result_ls:
                if result : #and len(result) >= 5
                    elite_sol, best_sol, init, f_num, may_conf,num= result
                    fragment_sol[f_num][init] = best_sol
                    may_conf_arc[f_num][init] = may_conf
                    # --- 关键修改：用上一轮的最优解替换掉旧的初始解列表 ---
                    # 这样能确保下一轮worker启动时，只使用这个最新的最优解
                    if best_sol:
                        initial_sol[f_num] = [best_sol]
                    successful_fragments += 1
                    best_link_time = best_sol.link_time if best_sol else 0
                    print(f"✓ Processed continuous training result by it_num {num} from fragment {f_num}: best_link_time = {best_link_time}", file=f)
            print(f"Successfully processed {successful_fragments}/{num_fragments} fragments", file=f)

            # 检查时间限制
            if time.time() - t1 > limit_time:
                break_all = True
                break
            if break_all:
                break


            # # --- 【核心替换】用联邦平均替换掉整个“分组外频繁模式挖掘”部分 ---
            # # 替换为下面的联邦平均逻辑：
            # print("开始全局模型联邦平均...", file=f)
            # # logger.info(f"Iteration {iter}: Starting global model averaging.")
            #
            # # 1. 收集所有子进程保存的模型路径
            # agent_model_paths = []
            # for f_num in range(len(serializable_model.fragment_list)):
            #     path = os.path.join(config.get('log_dir'), "trained_model", f"fragment_{f_num}.pt")
            #     if os.path.exists(path):
            #         agent_model_paths.append(path)
            # if not agent_model_paths:
            #     print("警告：没有找到任何Agent模型用于平均。", file=f)
            #     # logger.warning(f"Iteration {iter}: No agent models found for averaging.")
            #     continue  # 如果没有模型，就直接进入下一轮
            # # 2. 执行模型平均
            # averaged_weights = average_model_weights(agent_model_paths)
            # # 3. 保存新的全局模型
            # if averaged_weights:
            #     # 为了能被agent的load_checkpoint兼容，我们保存一个完整的checkpoint结构
            #     # 注意：这里只保存了模型权重，其他如epsilon等状态由agent自己维护
            #     global_checkpoint = {
            #         'model_state': averaged_weights,
            #         # 可以添加一些元数据
            #         'averaged_from_agents': len(agent_model_paths),
            #         'update_timestamp': time.time(),
            #     }
            #     os.makedirs(os.path.dirname(global_model_path), exist_ok=True)
            #     torch.save(global_checkpoint, global_model_path)
            #     print(f"新的全局模型已保存至: {global_model_path}", file=f)
            #     # logger.info(f"Iteration {iter}: New global model saved to {global_model_path}")
            # else:
            #     print("模型平均失败，跳过本轮全局模型更新。", file=f)
            #     # logger.error(f"Iteration {iter}: Model averaging failed.")

        from metrics_logger import merge_iterations
        metrics_dir = os.path.join(config.get('log_dir'), 'metrics')
        for f_num in range(len(serializable_model.fragment_list)):
            try:
                merged_file = merge_iterations(
                    fragment_id=f_num,
                    log_dir=metrics_dir,
                    iter_num=iter_num
                )
                print(f"✓ Fragment {f_num}: Merged to {merged_file}")
            except Exception as e:
                print(f"✗ Fragment {f_num}: Merge failed - {str(e)}")
        print("\n" + "=" * 60)
        print("All metrics saved and merged successfully!")
        print("=" * 60 + "\n")

        # 合并片段解决方案（保持原有逻辑）
        print("Combining fragment solutions...", file=f)
        sol = Sol.BestSol()
        frag_link = []

        for i in fragment_sol:
            if not fragment_sol[i]:
                continue

            # 获取该片段的最佳解
            f_sol_index = max(fragment_sol[i], key=lambda x: fragment_sol[i][x].link_time if fragment_sol[i][x] else 0)

            if f_sol_index in fragment_sol[i] and fragment_sol[i][f_sol_index]:
                frag_link.extend(may_conf_arc[i][f_sol_index])
                f_sol = fragment_sol[i][f_sol_index]
                sol.best_sol_ls.extend(f_sol.Arc_list_id)
                sol.sum_lt += f_sol.link_time

                print(f"Fragment {i} contributed {len(f_sol.Arc_list_id)} arcs, link_time: {f_sol.link_time}", file=f)

        before_conf_dele = copy.copy(sol.sum_lt)

        # 解决片段间冲突（保持原有逻辑）
        print("Resolving conflicts between fragments...", file=f)
        resolve_and_repair_conflicts(sol, mode, frag_link)

        from improved_constraint_repairer import ImprovedConstraintRepairer
        repairer = ImprovedConstraintRepairer(mode)
        # 注意：传入弧段ID列表，而不是Sol对象
        repaired_arc_list = repairer.repair_solution(sol.best_sol_ls, verbose=True)
        # # 验证修复结果
        # is_valid, details = repairer.verify_solution_like_check(repaired_arc_list)
        # if is_valid:
        #     print("✓ Final solution passed constraint verification", file=f)
        #     final_arc_list = repaired_arc_list
        # else:
        #     print(f"⚠ Final solution still has conflicts: {details[:3]}", file=f)
        #     # 使用保守策略再次修复
        #     final_arc_list = repairer._resolve_conflicts_conservative(final_arc_list, [])
        print("✓ Conflict resolution completed successfully", file=f)
        sol.best_sol_ls = repaired_arc_list
        sol.sum_lt=sum(mode.Arc_list[arc_id].link_time for arc_id in repaired_arc_list)

        # 记录最终解决方案质量
        sol.sum_ln = len(sol.best_sol_ls)
        print(f"冲突消解前目标值: {before_conf_dele}", file=f)
        print(f"冲突消解前目标值: {before_conf_dele}")
        print(f"冲突消解后目标值: {sol.sum_lt}", file=f)
        print(f"冲突消解后目标值: {sol.sum_lt}")
        print(f"Final solution: {sol.sum_ln} arcs, total link_time: {sol.sum_lt}")

        # 计算冲突解决的改进/损失
        if before_conf_dele > 0:
            change_pct = (sol.sum_lt - before_conf_dele) / before_conf_dele * 100
            print(f"Change after conflict resolution: {change_pct:.2f}%", file=f)
            print(f"Change after conflict resolution: {change_pct:.2f}%")

        # # 在 fixed_main2.py 的最后添加
        # from check import Check
        # # 最终解决方案验证
        # print("正在进行约束验证...", file=f)
        # checker = Check(mode, sol.best_sol_ls)
        # check = checker.check_result()
        # print("如果为1则通过:", check, file=f)
        
        # 记录时间和解决方案质量
        t2 = time.time()
        lls.append(sol.sum_lt)
        t.append(t2 - t1)
        print(f"Total continuous training time: {t2 - t1:.2f} seconds", file=f)
        
        # 记录本文件的结果
        rrrrrrrr.append([lls, sum(lls) / len(lls) if lls else 0])
        tttttttt.append([t, sum(t) / len(t) if t else 0])
        iiiii += 1
    
    # 打印最终结果
    print("\n--------------------- FINAL RESULTS ---------------------", file=f)
    print(f"Final solution link time: {sol.sum_lt}", file=f)
    print(f"Final solution link count: {sol.sum_ln}", file=f)
    print(f"Results summary: {rrrrrrrr}", file=f)
    print(f"Timing summary: {tttttttt}", file=f)
    
    return sol.sum_lt

# Define incremental local search function
def incremental_local_search(sol, mode, removed_arcs):
    """
    Function: Incrementally insert high-value links to recover objective value lost due to conflict resolution.
    """
    # Sort removed arcs by value (link_time / conflicts)
    removed_arcs.sort(
        key=lambda x: mode.Arc_list[x].link_time / (
        len(mode.Arc_list[x].confArc_list)) if len(mode.Arc_list[x].confArc_list) > 0 else float('inf'),
        reverse=True
        )

    # Incremental insertion
    for arc in removed_arcs:
        # Check if current link can be inserted without causing conflict
        if can_insert_without_conflict(arc, sol, mode):
            sol.best_sol_ls.append(arc)
            sol.sum_lt += mode.Arc_list[arc].link_time
            sol.sum_ln += 1

def can_insert_without_conflict(arc, sol, mode):
    """
    Function: Check if inserting a link will cause conflict.
    """
    for existing_arc in sol.best_sol_ls:
        if existing_arc in mode.Arc_list[arc].confArc_list:
            return False
    return True

# Improved conflict resolution and completion logic
def resolve_and_repair_conflicts(sol, mode, frag_link):
    """
    Function: Conflict resolution and incremental completion.
    """
    # Record links removed due to conflict resolution
    removed_arcs = frag_link.copy()

    # Use IP for conflict resolution
    ip = IP(mode, frag_link)
    add_arc = ip.const_ip_model()  # Solution obtained by cplex at fragment junction
    for i in add_arc:
        if i in frag_link:
            frag_link.remove(i)
    for i in frag_link:
        if i in sol.best_sol_ls:
            sol.best_sol_ls.remove(i)
            sol.sum_lt -= mode.Arc_list[i].link_time
    sol.sum_ln = len(sol.best_sol_ls)

    # Incremental local search completion
    incremental_local_search(sol, mode, removed_arcs)

    # Further complement remaining available links
    all_arcs = [arc.id for arc in mode.Arc_list]
    remove = []
    for arc in sol.best_sol_ls:
        try:
            all_arcs.remove(arc)
            remove.extend(mode.Arc_list[arc].confArc_list)
        except ValueError:
            # Skip if arc is not in all_arcs
            pass
    remove = list(set(remove))
    for c_arc in remove:
        if c_arc in all_arcs:
            all_arcs.remove(c_arc)

    # Use IP to complete remaining links
    ip1 = IP(mode, all_arcs)
    after_conf_add = ip1.const_ip_model()

    # If cplex didn't finish solving within given time, use rules
    remove1 = []
    for arc in after_conf_add:
        if arc in all_arcs:
            all_arcs.remove(arc)
        remove1.extend(mode.Arc_list[arc].confArc_list)
    remove1 = list(set(remove1))
    for c_arc in remove1:
        if c_arc in all_arcs:
            all_arcs.remove(c_arc)

    # Rule-based conflict resolution
    cs = confRes(mode, all_arcs)
    add_arc = cs.start()
    after_conf_add += add_arc

    # Insert completed links
    for arc in after_conf_add:
        sol.best_sol_ls.append(arc)
        sol.sum_lt += mode.Arc_list[arc].link_time
    sol.sum_ln = len(sol.best_sol_ls)

if __name__ == '__main__':
    from Config import Config
    import os

    config = Config('args.yaml', 'yaml')
    os.makedirs(config.get('log_dir'), exist_ok=True)
    config.save_config_to_log('args.yaml', config.get('log_dir'))

    outpath = os.path.join(config.get('log_dir'), 'improved_corrected_output.txt')
    with open(outpath, "w") as f:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("改进修正后程序开始时间:", start_time, file=f)
        print("Improved corrected program start time:", start_time)

        try:
            main(f)
        except Exception as e:
            print(f"Program execution error: {str(e)}", file=f)
            import traceback
            traceback.print_exc()

        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("改进修正后程序结束时间:", end_time, file=f)
        print("改进修正后程序结束时间:", end_time)
