# -*- coding: utf-8 -*-
# @Time    : 2025/10/10 14:14
# @Author  : JIA
# @FileName: metrics_logger.py
# @Software: PyCharm
# @Blog    ：
"""
MetricsLogger - 轻量级指标记录器
特点：
1. 自动跟踪累计统计（counts, frequencies）
2. 延迟计算汇总指标（operator_stats, summary）
3. 支持多iteration训练，独立保存每个iter
4. 提供合并函数，生成完整训练历史
"""

import json
import os
import time
from collections import defaultdict
import numpy as np


class MetricsLogger:
    """指标记录器 - 每个(fragment, iteration)组合独立实例"""

    def __init__(self, fragment_id, iteration_id=0, log_dir="logs/metrics",
                 hyperparameters=None):
        """
        初始化记录器

        Args:
            fragment_id: 片段编号
            iteration_id: 迭代编号（外层iter）
            log_dir: 日志目录
            hyperparameters: 超参数字典
        """
        self.fragment_id = fragment_id
        self.iteration_id = iteration_id
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 元数据
        self.metadata = {
            "fragment_id": fragment_id,
            "iteration_id": iteration_id,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": hyperparameters or {}
        }

        # 步骤数据（运行时累积）
        self.steps = []
        self.step_counter = 0
        self.start_timestamp = time.time()

        # 自动维护累计统计
        self.destroy_counts = np.zeros(8, dtype=int)
        self.repair_counts = np.zeros(8, dtype=int)

        # 初始状态记录
        self.initial_objective = None
        self.best_objective_history = []

        # Step偏移量（用于跨iter累积编号）
        self.step_offset = 0

    def set_step_offset(self, offset):
        """设置step偏移量（用于跨iter累积）"""
        self.step_offset = offset

    def log_step(self,
                 # ========== 必需参数 ==========
                 destroy_id, repair_id, is_exploration,
                 current_obj, new_obj, best_obj, improvement,
                 state, epsilon, reward, is_accepted,

                 # ========== 可选参数（有默认值）==========
                 iteration=None, it_index=0, epoch=0, inner_iteration=0,
                 is_best_improved=False,
                 loss=None,
                 td_error=None,
                 q_metrics=None,
                 operator_values=None,
                 conflicts=None,
                 reward_components=None,
                 solution_sizes=None,
                 acceptance_reason="improvement",
                 step_time=0.0,
                 iterations_without_improvement=0,
                 replay_buffer_size=0,
                 target_network_updates=0,
                 **kwargs):
        """
        记录单步数据

        必需参数：
            destroy_id, repair_id: 算子选择
            is_exploration: 是否探索
            current_obj, new_obj, best_obj, improvement: 目标值
            state: 17维状态向量
            epsilon, reward, is_accepted: 训练状态

        可选参数：
            q_metrics: {avg, max, min, selected, std, destroy_avg, repair_avg}
            operator_values: {destroy_values, repair_values}
            conflicts: {internal, external_count, external_time, boundary_risk}
            reward_components: {link_time, conflict, diversity, efficiency, boundary}
            solution_sizes: {current, new, best, diversity}
            loss, step_time, etc.
        """

        # 自动更新累计计数
        self.destroy_counts[destroy_id] += 1
        self.repair_counts[repair_id] += 1

        # 自动计算频率
        total_steps = self.step_counter + 1
        destroy_freq = (self.destroy_counts / total_steps).tolist()
        repair_freq = (self.repair_counts / total_steps).tolist()

        # 记录初始目标值
        if self.initial_objective is None:
            self.initial_objective = current_obj

        # 记录最优历史
        self.best_objective_history.append(best_obj)

        # 使用iteration_id作为默认iteration值
        if iteration is None:
            iteration = self.iteration_id

        # 构建步骤记录
        step_record = {
            # ========== 基本信息 ==========
            "step": self.step_counter + self.step_offset,
            "iteration": iteration,
            "it_index": it_index,
            "epoch": epoch,
            "inner_iteration": inner_iteration,
            "timestamp": time.time() - self.start_timestamp,

            # ========== 状态信息 ==========
            "state": state if isinstance(state, list) else state.tolist() if hasattr(state, 'tolist') else list(state),
            "state_summary": self._extract_state_summary(state, current_obj),

            # ========== 算子选择 ==========
            "destroy_id": int(destroy_id),
            "repair_id": int(repair_id),
            "action_id": int(destroy_id * 8 + repair_id),
            "is_exploration": bool(is_exploration),

            # ========== 算子频率（自动计算）==========
            "destroy_selection_freq": destroy_freq,
            "repair_selection_freq": repair_freq,
            "destroy_selection_counts": self.destroy_counts.tolist(),
            "repair_selection_counts": self.repair_counts.tolist(),

            # ========== 算子价值 ==========
            "destroy_values": self._safe_get_list(operator_values, 'destroy_values', 8),
            "repair_values": self._safe_get_list(operator_values, 'repair_values', 8),

            # ========== 目标值 ==========
            "current_obj": float(current_obj),
            "new_obj": float(new_obj),
            "best_obj": float(best_obj),
            "improvement": float(improvement),
            "improvement_rate": float(improvement / current_obj) if current_obj > 0 else 0.0,
            "cumulative_improvement": float(best_obj - self.initial_objective),

            # ========== 解的规模 ==========
            "current_solution_size": int(solution_sizes.get('current', 0)) if solution_sizes else 0,
            "new_solution_size": int(solution_sizes.get('new', 0)) if solution_sizes else 0,
            "best_solution_size": int(solution_sizes.get('best', 0)) if solution_sizes else 0,
            "solution_diversity": float(solution_sizes.get('diversity', 0.0)) if solution_sizes else 0.0,

            # ========== Q值统计 ==========
            **self._format_q_metrics(q_metrics),

            # ========== 训练状态 ==========
            "epsilon": float(epsilon),
            "loss": float(loss) if loss is not None else 0.0,
            "reward": float(reward),
            "td_error": float(td_error) if td_error is not None else 0.0,  # <-- 新增td_error字段
            "is_accepted": bool(is_accepted),
            "is_best_improved": bool(is_best_improved),
            "acceptance_reason": str(acceptance_reason),

            # ========== 奖励分解 ==========
            **self._format_reward_components(reward_components),

            # ========== 冲突信息 ==========
            **self._format_conflicts(conflicts),

            # ========== 其他关键指标 ==========
            "iterations_without_improvement": int(iterations_without_improvement),
            "replay_buffer_size": int(replay_buffer_size),
            "target_network_updates": int(target_network_updates),
            "step_time": float(step_time)
        }

        # 添加额外的kwargs
        for key, value in kwargs.items():
            if key not in step_record:
                step_record[key] = value

        self.steps.append(step_record)
        self.step_counter += 1

    def _safe_get_list(self, dict_obj, key, expected_length):
        """安全获取列表，确保长度正确"""
        if not dict_obj or key not in dict_obj:
            return [0.0] * expected_length

        values = dict_obj[key]
        if isinstance(values, (list, np.ndarray)):
            values = list(values)
            # 确保长度
            if len(values) < expected_length:
                values.extend([0.0] * (expected_length - len(values)))
            elif len(values) > expected_length:
                values = values[:expected_length]
            return values
        else:
            return [0.0] * expected_length

    # def _extract_state_summary(self, state, current_obj):
    #     """从状态向量提取关键特征"""
    #     if not state or len(state) < 17:
    #         return {
    #             "current_link_time": float(current_obj),
    #             "current_link_num": 0,
    #             "solution_density": 0.0,
    #             "avg_link_time_per_arc": 0.0,
    #             "ground_station_count": 0,
    #             "conflict_ratio": 0.0,
    #             "external_conflict_ratio": 0.0,
    #             "fragment_position": 0.0
    #         }
    #
    #     # 将state转为列表（如果是numpy数组）
    #     if hasattr(state, 'tolist'):
    #         state = state.tolist()
    #
    #     return {
    #         "current_link_time": float(current_obj),
    #         "current_link_num": int(state[1] * 1000) if len(state) > 1 else 0,
    #         "solution_density": float(state[2]) if len(state) > 2 else 0.0,
    #         "avg_link_time_per_arc": float(state[3] * 10000) if len(state) > 3 else 0.0,
    #         "ground_station_count": int(state[4] * 100) if len(state) > 4 else 0,
    #         "conflict_ratio": float(state[7]) if len(state) > 7 else 0.0,
    #         "external_conflict_ratio": float(state[13]) if len(state) > 13 else 0.0,
    #         "fragment_position": float(state[12]) if len(state) > 12 else 0.0
    #     }


    # 【请用此版本完整替换旧函数】
    def _extract_state_summary(self, state, current_obj):
        """从15维状态向量提取关键特征"""
        if not state or len(state) < 15:  # <-- 修改维度检查为15
            # 返回一个匹配新维度的空结构
            return {
                "current_link_time": float(current_obj),
                "solution_density": 0.0,
                "avg_link_time_per_arc": 0.0,
                "ground_station_count": 0,
                "internal_conflict_ratio": 0.0,
                "stagnation": 0.0,
                "improvement_rate": 0.0,
                "boundary_density": 0.0,
                "left_neighbor_conflict": 0.0,
                "right_neighbor_conflict": 0.0
            }

        if hasattr(state, 'tolist'):
            state = state.tolist()

        return {
            "current_link_time": float(state[0] * 1e6),  # 反归一化
            "solution_density": float(state[1]),
            "avg_link_time_per_arc": float(state[2] * 10000),  # 反归一化
            "ground_station_count": int(state[3] * 100),  # 反归一化
            "internal_conflict_ratio": float(state[5]),
            "stagnation": float(state[11]),
            "improvement_rate": float(state[12]),
            "boundary_density": float(state[10]),
            "left_neighbor_conflict": float(state[13]),
            "right_neighbor_conflict": float(state[14])
        }

    def _format_q_metrics(self, q_metrics):
        """格式化Q值指标"""
        if not q_metrics:
            return {
                "q_avg": 0.0,
                "q_max": 0.0,
                "q_min": 0.0,
                "q_selected": 0.0,
                "q_std": 0.0,
                "q_gap": 0.0,
                "q_all_values": [0.0] * 64,  # ✅ 新增：完整64维Q值
                "q_destroy_avg": [0.0] * 8,
                "q_repair_avg": [0.0] * 8
            }

        q_avg = float(q_metrics.get('avg', 0.0))
        q_max = float(q_metrics.get('max', 0.0))
        q_min = float(q_metrics.get('min', 0.0))
        q_selected = float(q_metrics.get('selected', 0.0))
        q_std = float(q_metrics.get('std', 0.0))

        return {
            "q_avg": q_avg,
            "q_max": q_max,
            "q_min": q_min,
            "q_selected": q_selected,
            "q_std": q_std,
            "q_gap": q_max - q_selected,
            "q_all_values": q_metrics.get('all_q_values', [0.0] * 64),  # ✅ 新增
            "q_destroy_avg": self._safe_get_list(q_metrics, 'destroy_avg', 8),
            "q_repair_avg": self._safe_get_list(q_metrics, 'repair_avg', 8)
        }

    # def _format_reward_components(self, reward_components):
    #     """格式化奖励分解"""
    #     if not reward_components:
    #         return {
    #             "reward_link_time": 0.0,
    #             "reward_conflict": 0.0,
    #             "reward_diversity": 0.0,
    #             "reward_efficiency": 0.0,
    #             "reward_boundary": 0.0
    #         }
    #
    #     return {
    #         "reward_link_time": float(reward_components.get('link_time', 0.0)),
    #         "reward_conflict": float(reward_components.get('conflict', 0.0)),
    #         "reward_diversity": float(reward_components.get('diversity', 0.0)),
    #         "reward_efficiency": float(reward_components.get('efficiency', 0.0)),
    #         "reward_boundary": float(reward_components.get('boundary', 0.0))
    #     }

    # metrics_logger.py
    # 【请用此版本完整替换旧函数】
    def _format_reward_components(self, reward_components):
        """格式化新的奖励分解"""
        if not reward_components:
            return {
                "reward_link_time": 0.0,
                "reward_improvement": 0.0,
                # "reward_structure": 0.0,
                "reward_boundary": 0.0,
                "raw_link_time_change": 0.0,
                "raw_conflict_improvement": 0.0
            }

        return {
            "reward_link_time": float(reward_components.get('link_time', 0.0)),
            "reward_improvement": float(reward_components.get('improvement', 0.0)),
            # "reward_structure": float(reward_components.get('structure', 0.0)),
            "reward_boundary": float(reward_components.get('boundary', 0.0)),
            "raw_link_time_change": float(reward_components.get('raw_link_time_change', 0.0)),
            "raw_conflict_improvement": float(reward_components.get('raw_conflict_improvement', 0.0))
        }

    def _format_conflicts(self, conflicts):
        """格式化冲突信息"""
        if not conflicts:
            return {
                "internal_conflicts": 0,
                "external_conflict_count": 0,
                "external_conflict_time": 0.0,
                "boundary_risk_density": 0.0
            }

        return {
            "internal_conflicts": int(conflicts.get('internal', 0)),
            "external_conflict_count": int(conflicts.get('external_count', 0)),
            "external_conflict_time": float(conflicts.get('external_time', 0.0)),
            "boundary_risk_density": float(conflicts.get('boundary_risk', 0.0))
        }

    def save(self, compute_summary=False):
        """
        保存数据到JSON文件

        Args:
            compute_summary: 是否计算汇总统计（iter结束时设为True）

        Returns:
            保存的文件路径
        """
        # 更新元数据
        self.metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["total_time_seconds"] = time.time() - self.start_timestamp
        self.metadata["total_steps"] = self.step_counter
        self.metadata["actual_steps"] = len(self.steps)

        # 构建数据结构
        data = {
            "metadata": self.metadata,
            "steps": self.steps
        }

        # 延迟计算汇总统计
        if compute_summary and self.steps:
            data["operator_stats"] = self._compute_operator_stats()
            data["summary"] = self._compute_summary()

        # 文件名包含iteration标识
        filename = f"fragment_{self.fragment_id}_iter{self.iteration_id}.json"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        status = "with summary" if compute_summary else "raw data"
        print(f"Fragment {self.fragment_id} Iter {self.iteration_id}: "
              f"Saved {len(self.steps)} steps ({status})")

        return filepath

    def _compute_operator_stats(self):
        """计算算子统计（从steps数据）"""
        if not self.steps:
            return []

        stats = []

        for d_id in range(8):
            for r_id in range(8):
                # 筛选该算子组合的所有步骤
                subset = [s for s in self.steps
                         if s['destroy_id'] == d_id and s['repair_id'] == r_id]

                if not subset:
                    continue

                improvements = [s['improvement'] for s in subset]
                q_values = [s['q_selected'] for s in subset]
                accepted = [s['is_accepted'] for s in subset]

                # 获取价值演化
                initial_value = subset[0]['destroy_values'][d_id] if subset[0]['destroy_values'] else 0.0
                final_value = subset[-1]['destroy_values'][d_id] if subset[-1]['destroy_values'] else 0.0

                stats.append({
                    "destroy_id": d_id,
                    "repair_id": r_id,
                    "action_id": d_id * 8 + r_id,

                    # 选择统计
                    "selection_count": len(subset),
                    "selection_freq": len(subset) / len(self.steps),
                    "avg_selection_freq": len(subset) / len(self.steps),

                    # 效果统计
                    "total_improvement": float(sum(improvements)),
                    "avg_improvement": float(np.mean(improvements)),
                    "max_improvement": float(max(improvements)),
                    "min_improvement": float(min(improvements)),

                    # 接受统计
                    "accepted_count": sum(accepted),
                    "acceptance_rate": float(np.mean(accepted)),
                    "rejected_count": len(subset) - sum(accepted),

                    # Q值统计
                    "total_q_value": float(sum(q_values)),
                    "avg_q_value": float(np.mean(q_values)),
                    "max_q_value": float(max(q_values)),
                    "min_q_value": float(min(q_values)),

                    # 价值演化
                    "initial_value": float(initial_value),
                    "final_value": float(final_value),
                    "value_change": float(final_value - initial_value)
                })

        return stats

    def _compute_summary(self):
        """计算汇总统计"""
        if not self.steps:
            return {}

        first_step = self.steps[0]
        last_step = self.steps[-1]

        # 找出最优算子组合
        operator_improvements = defaultdict(float)
        operator_counts = defaultdict(int)
        for s in self.steps:
            key = (s['destroy_id'], s['repair_id'])
            operator_improvements[key] += s['improvement']
            operator_counts[key] += 1

        if operator_improvements:
            best_combo = max(operator_improvements.items(), key=lambda x: x[1])
            best_combo_key = best_combo[0]
            best_combo_improvement = best_combo[1]

            # 计算该组合的接受率
            best_combo_steps = [s for s in self.steps
                               if s['destroy_id'] == best_combo_key[0]
                               and s['repair_id'] == best_combo_key[1]]
            best_combo_acceptance = np.mean([s['is_accepted'] for s in best_combo_steps]) if best_combo_steps else 0
        else:
            best_combo_key = (0, 0)
            best_combo_improvement = 0
            best_combo_acceptance = 0

        # 找出最高频率算子
        final_d_counts = np.array(last_step['destroy_selection_counts'])
        final_r_counts = np.array(last_step['repair_selection_counts'])

        # 找出最高价值算子
        final_d_values = last_step['destroy_values']
        final_r_values = last_step['repair_values']

        # 收敛信息
        converged = self._check_convergence()
        convergence_step = self._find_convergence_step()

        return {
            # 目标值统计
            "initial_obj": float(self.initial_objective),
            "final_obj": float(last_step['best_obj']),
            "best_obj": float(max(self.best_objective_history)),
            "total_improvement": float(last_step['best_obj'] - self.initial_objective),
            "improvement_rate": float((last_step['best_obj'] - self.initial_objective) / self.initial_objective) if self.initial_objective > 0 else 0.0,

            # 步数统计
            "total_steps": len(self.steps),
            "actual_steps": len(self.steps),
            "accepted_steps": sum(s['is_accepted'] for s in self.steps),
            "acceptance_rate": float(np.mean([s['is_accepted'] for s in self.steps])),
            "best_improved_steps": sum(s['is_best_improved'] for s in self.steps),

            # 时间统计
            "total_time": float(last_step['timestamp']),
            "avg_step_time": float(last_step['timestamp'] / len(self.steps)),
            "total_destroy_time": 0.0,
            "total_repair_time": 0.0,
            "total_training_time": 0.0,

            # 算子使用摘要
            # ✅ 修复：所有numpy数组的操作结果也要转换
            "most_used_destroy": int(np.argmax(final_d_counts)),
            "most_used_repair": int(np.argmax(final_r_counts)),
            "most_effective_combination": list(best_combo_key),
            "best_combination_improvement": float(best_combo_improvement),
            "best_combination_acceptance_rate": float(best_combo_acceptance),

            # 最高频率算子
            # ✅ 修复：字典中的numpy类型
            "highest_freq_destroy": {
                "id": int(np.argmax(final_d_counts)),
                "freq": float(np.max(final_d_counts)) / len(self.steps),
                "count": int(np.max(final_d_counts))
            },
            "highest_freq_repair": {
                "id": int(np.argmax(final_r_counts)),
                "freq": float(np.max(final_r_counts) / len(self.steps)),
                "count": int(np.max(final_r_counts))
            },

            # 最高价值算子
            "highest_value_destroy": {
                "id": int(np.argmax(final_d_values)),
                "value": float(np.max(final_d_values)),
                "avg_q": 0.0
            },
            "highest_value_repair": {
                "id": int(np.argmax(final_r_values)),
                "value": float(np.max(final_r_values)),
                "avg_q": 0.0
            },

            # Q值演化
            "initial_avg_q": float(first_step['q_avg']),
            "final_avg_q": float(last_step['q_avg']),
            "max_avg_q": float(max(s['q_avg'] for s in self.steps)),
            "q_growth": float(last_step['q_avg'] - first_step['q_avg']),

            # 探索vs利用
            "exploration_count": sum(s['is_exploration'] for s in self.steps),
            "exploitation_count": sum(not s['is_exploration'] for s in self.steps),
            "exploration_rate": float(np.mean([s['is_exploration'] for s in self.steps])),
            "final_epsilon": float(last_step['epsilon']),

            # 收敛信息
            # ✅ 修复：显式转换为Python原生类型
            "converged": bool(converged),  # 确保是Python bool
            "convergence_step": int(convergence_step),  # 确保是Python int
            "convergence_threshold": 0.001,
            "plateau_length": len(self.steps) - convergence_step if converged else 0,

            # 状态统计
            # 【状态统计 - 全面更新】
            # 使用 .get(key, 0.0) 来安全地访问，避免未来再出现KeyError
            "avg_solution_density": float(np.mean([s['state_summary'].get('solution_density', 0.0) for s in self.steps])),
            "avg_internal_conflict_ratio": float(np.mean([s['state_summary'].get('internal_conflict_ratio', 0.0) for s in self.steps])),

            # 【新增：计算新状态特征的平均值】
            "avg_stagnation": float(np.mean([s['state_summary'].get('stagnation', 0.0) for s in self.steps])),
            "avg_improvement_rate": float(np.mean([s['state_summary'].get('improvement_rate', 0.0) for s in self.steps])),
            "avg_boundary_density": float(np.mean([s['state_summary'].get('boundary_density', 0.0) for s in self.steps])),
            "avg_left_neighbor_conflict": float(np.mean([s['state_summary'].get('left_neighbor_conflict', 0.0) for s in self.steps])),
            "avg_right_neighbor_conflict": float(np.mean([s['state_summary'].get('right_neighbor_conflict', 0.0) for s in self.steps]))

        }

    def _check_convergence(self, window=100, threshold=0.001):
        """检查是否收敛"""
        if len(self.steps) < window:
            return False

        recent_objs = [s['best_obj'] for s in self.steps[-window:]]
        variance = np.var(recent_objs)
        return variance < threshold

    def _find_convergence_step(self, window=100, threshold=0.001):
        """找到收敛点"""
        if len(self.steps) < window:
            return len(self.steps)

        for i in range(window, len(self.steps)):
            recent_objs = [s['best_obj'] for s in self.steps[i-window:i]]
            if np.var(recent_objs) < threshold:
                return i

        return len(self.steps)


# ========== 合并函数 ==========

def merge_iterations(fragment_id, log_dir="logs/metrics", iter_num=5):
    """
    合并同一片段的多个iter数据

    Args:
        fragment_id: 片段编号
        log_dir: 日志目录
        iter_num: 迭代次数

    Returns:
        合并后的文件路径

    输入：fragment_0_iter0.json, fragment_0_iter1.json, ...
    输出：fragment_0_complete.json
    """
    merged_data = {
        "metadata": None,
        "steps": [],
        "iteration_summaries": []
    }

    step_offset = 0

    for iter_id in range(iter_num):
        filename = f"fragment_{fragment_id}_iter{iter_id}.json"
        filepath = os.path.join(log_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found, skipping")
            continue

        with open(filepath, 'r') as f:
            iter_data = json.load(f)

        # 第一个iter的metadata作为基础
        if merged_data["metadata"] is None:
            merged_data["metadata"] = iter_data["metadata"].copy()
            merged_data["metadata"]["iterations"] = iter_num
            merged_data["metadata"]["iteration_id"] = "merged"

        # 调整step编号（累积）
        for step in iter_data["steps"]:
            step["step"] = step["step"] + step_offset
            step["iteration"] = iter_id
            merged_data["steps"].append(step)

        step_offset += len(iter_data["steps"])

        # 保存每个iter的摘要
        if "summary" in iter_data:
            iter_summary = iter_data["summary"].copy()
            iter_summary["iteration"] = iter_id
            merged_data["iteration_summaries"].append(iter_summary)

    # --- 【核心修正】开始：重新计算累计频率 ---
    if merged_data["steps"]:
        print(
            f"Fragment {fragment_id}: Recalculating cumulative frequencies for {len(merged_data['steps'])} steps...")

        # 初始化累计计数器
        cumulative_destroy_counts = np.zeros(8, dtype=int)
        cumulative_repair_counts = np.zeros(8, dtype=int)

        for i, step in enumerate(merged_data["steps"]):
            # 更新累计计数
            cumulative_destroy_counts[step['destroy_id']] += 1
            cumulative_repair_counts[step['repair_id']] += 1

            # 当前总步数 (i+1)
            total_cumulative_steps = i + 1

            # 重新计算并覆盖旧的频率值
            step['destroy_selection_freq'] = (cumulative_destroy_counts / total_cumulative_steps).tolist()
            step['repair_selection_freq'] = (cumulative_repair_counts / total_cumulative_steps).tolist()

            # （可选但推荐）同时更新计数值，使其也反映累计状态
            step['destroy_selection_counts'] = cumulative_destroy_counts.tolist()
            step['repair_selection_counts'] = cumulative_repair_counts.tolist()
    # --- 【核心修正】结束 ---

    if not merged_data["steps"]:
        print(f"Warning: No data found for fragment {fragment_id}")
        return None

    # 重新计算合并后的汇总统计
    logger_temp = MetricsLogger(fragment_id)
    logger_temp.steps = merged_data["steps"]
    logger_temp.initial_objective = merged_data["steps"][0]["current_obj"]
    logger_temp.best_objective_history = [s["best_obj"] for s in merged_data["steps"]]
    logger_temp.destroy_counts = np.array(merged_data["steps"][-1]["destroy_selection_counts"])
    logger_temp.repair_counts = np.array(merged_data["steps"][-1]["repair_selection_counts"])

    merged_data["operator_stats"] = logger_temp._compute_operator_stats()
    merged_data["summary"] = logger_temp._compute_summary()
    merged_data["summary"]["total_iterations"] = iter_num

    # 更新元数据
    merged_data["metadata"]["total_steps"] = len(merged_data["steps"])
    merged_data["metadata"]["end_time"] = merged_data["steps"][-1].get("timestamp", 0)

    # 保存合并后的文件
    output_file = f"fragment_{fragment_id}_complete.json"
    output_path = os.path.join(log_dir, output_file)

    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Fragment {fragment_id}: Merged {iter_num} iterations into {output_file}")
    print(f"  Total steps: {len(merged_data['steps'])}")
    print(f"  Total improvement: {merged_data['summary']['total_improvement']:.2f}")

    return output_path


def merge_all_fragments(log_dir="logs/metrics", n_fragments=12, iter_num=5):
    """
    合并所有片段的数据

    Args:
        log_dir: 日志目录
        n_fragments: 片段数量
        iter_num: 迭代次数

    Returns:
        合并成功的片段列表
    """
    print("\n" + "="*60)
    print("Merging iteration data for all fragments...")
    print("="*60 + "\n")

    merged_files = []

    for f_num in range(n_fragments):
        try:
            merged_file = merge_iterations(
                fragment_id=f_num,
                log_dir=log_dir,
                iter_num=iter_num
            )
            if merged_file:
                merged_files.append(merged_file)
                print(f"✓ Fragment {f_num}: Successfully merged\n")
        except Exception as e:
            print(f"✗ Fragment {f_num}: Merge failed - {str(e)}\n")

    print("="*60)
    print(f"Merge completed: {len(merged_files)}/{n_fragments} fragments successful")
    print("="*60 + "\n")

    return merged_files


# ========== 测试代码 ==========

# if __name__ == '__main__':
#     """简单测试"""
#     import random
#
#     # 创建测试logger
#     logger = MetricsLogger(
#         fragment_id=0,
#         iteration_id=0,
#         hyperparameters={'epochs': 5, 'q': 50}
#     )
#
#     # 模拟记录几步
#     for step in range(10):
#         logger.log_step(
#             destroy_id=random.randint(0, 7),
#             repair_id=random.randint(0, 7),
#             is_exploration=random.random() < 0.5,
#             current_obj=1000 + step * 10,
#             new_obj=1000 + step * 10 + random.randint(-5, 15),
#             best_obj=1000 + step * 10,
#             improvement=random.randint(-5, 15),
#             state=[random.random() for _ in range(17)],
#             epsilon=0.9 - step * 0.01,
#             reward=random.random() * 10,
#             is_accepted=random.random() < 0.6
#         )
#
#     # 保存
#     saved_file = logger.save(compute_summary=True)
#     print(f"\nTest file saved: {saved_file}")
#
#     # 验证
#     with open(saved_file) as f:
#         data = json.load(f)
#
#     print(f"\nVerification:")
#     print(f"  Steps: {len(data['steps'])}")
#     print(f"  Has operator_stats: {'operator_stats' in data}")
#     print(f"  Has summary: {'summary' in data}")
#     print(f"  First step keys: {len(data['steps'][0].keys())} fields")