# improved_constraint_repairer.py
# 改进的约束修复器：与check.py完全一致的约束检测和修复

import copy
import random
from collections import defaultdict

class ImprovedConstraintRepairer:
    """
    改进的约束修复器：确保与check.py的约束检测逻辑完全一致
    """
    
    def __init__(self, model):
        self.model = model
        # 使用与check.py完全一致的约束参数
        self.GROUND_TRANS_TIME = 340  # 地面站转换时间
        self.SATELLITE_CHANGE_TIME = 150  # 卫星切换时间
        self.SATELLITE_TRANS_TIME = 300   # 卫星转换时间
        
    def repair_solution(self, solution_arcs, fragment_id=None, verbose=False):
        """
        修复解中的内部冲突，使用与check.py完全一致的逻辑
        """
        if not solution_arcs:
            return []
            
        if verbose:
            print(f"开始修复解，包含 {len(solution_arcs)} 个弧段")
            
        # 1. 使用与check.py一致的冲突检测
        conflicts = self._detect_conflicts_like_check(solution_arcs)
        if not conflicts:
            if verbose:
                print("未发现冲突，直接返回原解")
            return solution_arcs
            
        if verbose:
            print(f"发现 {len(conflicts)} 个冲突")
            
        # 2. 多策略修复
        strategies = [
            self._resolve_conflicts_value_based,
            self._resolve_conflicts_time_based,
            self._resolve_conflicts_conservative
        ]
        
        best_solution = []
        best_value = 0
        
        for i, strategy in enumerate(strategies):
            try:
                repaired = strategy(solution_arcs, conflicts)
                if repaired:
                    # 验证修复结果
                    final_conflicts = self._detect_conflicts_like_check(repaired)
                    if not final_conflicts:
                        value = sum(self.model.Arc_list[arc_id].link_time for arc_id in repaired)
                        if value > best_value:
                            best_solution = repaired
                            best_value = value
                            if verbose:
                                print(f"策略 {i+1} 成功修复，价值: {value}")
                    elif verbose:
                        print(f"策略 {i+1} 修复后仍有 {len(final_conflicts)} 个冲突")
                        
            except Exception as e:
                if verbose:
                    print(f"策略 {i+1} 执行失败: {str(e)}")
                continue
                
        if verbose:
            print(f"最终修复结果：{len(best_solution)} 个弧段，价值: {best_value}")
            
        return best_solution
    
    def _detect_conflicts_like_check(self, solution_arcs):
        """
        使用与check.py完全一致的冲突检测逻辑
        """
        conflicts = []
        
        # 按开始时间排序（与check.py一致）
        sort_arc = sorted(copy.copy(solution_arcs), 
                         key=lambda x: self.model.Arc_list[x].link_st)
        
        for i in range(len(sort_arc)):
            a1_id = sort_arc[i]
            a1 = self.model.Arc_list[a1_id]
            
            for j in range(i+1, len(sort_arc)):
                a2_id = sort_arc[j]
                a2 = self.model.Arc_list[a2_id]
                
                # 地面站约束检测（与check.py完全一致）
                if a1.ground == a2.ground:
                    if a2.link_st < a1.link_et + self.GROUND_TRANS_TIME:
                        conflicts.append({
                            'type': 'ground_station',
                            'arc1': a1_id,
                            'arc2': a2_id,
                            'violation': a1.link_et + self.GROUND_TRANS_TIME - a2.link_st,
                            'error_code': 1
                        })
                
                # 卫星约束检测（与check.py完全一致）
                if a1.satellite == a2.satellite:
                    # 同时开始冲突
                    if a2.link_st == a1.link_st:
                        conflicts.append({
                            'type': 'satellite_same_start',
                            'arc1': a1_id,
                            'arc2': a2_id,
                            'violation': 0,
                            'error_code': 2
                        })
                    elif a2.link_st < a1.link_et:
                        # 包含关系冲突
                        if a2.link_et <= a1.link_et:
                            conflicts.append({
                                'type': 'satellite_contained',
                                'arc1': a1_id,
                                'arc2': a2_id,
                                'violation': a1.link_et - a2.link_et,
                                'error_code': 3
                            })
                        # 重叠但切换时间不足
                        elif a1.link_et - a2.link_st < self.SATELLITE_CHANGE_TIME:
                            conflicts.append({
                                'type': 'satellite_overlap',
                                'arc1': a1_id,
                                'arc2': a2_id,
                                'violation': self.SATELLITE_CHANGE_TIME - (a1.link_et - a2.link_st),
                                'error_code': 4
                            })
                    # 转换时间不足
                    elif a2.link_st - a1.link_et < self.SATELLITE_TRANS_TIME:
                        conflicts.append({
                            'type': 'satellite_transition',
                            'arc1': a1_id,
                            'arc2': a2_id,
                            'violation': self.SATELLITE_TRANS_TIME - (a2.link_st - a1.link_et),
                            'error_code': 5
                        })
        
        return conflicts
    
    def _resolve_conflicts_value_based(self, solution_arcs, conflicts):
        """
        基于价值的冲突解决策略
        """
        # 计算每个弧段的价值密度
        arc_values = {}
        for arc_id in solution_arcs:
            arc = self.model.Arc_list[arc_id]
            arc_values[arc_id] = arc.link_time / max(arc.link_time / 3600, 0.1)  # 价值密度
        
        # 构建冲突图
        conflict_graph = defaultdict(set)
        for conflict in conflicts:
            arc1, arc2 = conflict['arc1'], conflict['arc2']
            conflict_graph[arc1].add(arc2)
            conflict_graph[arc2].add(arc1)
        
        # 贪心选择
        selected = []
        remaining = sorted(solution_arcs, key=lambda x: arc_values[x], reverse=True)
        
        for arc in remaining:
            # 检查是否与已选弧段冲突
            has_conflict = any(selected_arc in conflict_graph[arc] for selected_arc in selected)
            if not has_conflict:
                selected.append(arc)
        
        return selected
    
    def _resolve_conflicts_time_based(self, solution_arcs, conflicts):
        """
        基于时间的冲突解决策略
        """
        # 按开始时间排序
        sorted_arcs = sorted(solution_arcs, key=lambda x: self.model.Arc_list[x].link_st)
        
        selected = []
        last_ground_end = {}  # 每个地面站的最后结束时间
        last_satellite_end = {}  # 每个卫星的最后结束时间
        
        for arc_id in sorted_arcs:
            arc = self.model.Arc_list[arc_id]
            can_add = True
            
            # 检查地面站约束
            if arc.ground in last_ground_end:
                if arc.link_st < last_ground_end[arc.ground] + self.GROUND_TRANS_TIME:
                    can_add = False
            
            # 检查卫星约束
            if arc.satellite in last_satellite_end and can_add:
                if arc.link_st < last_satellite_end[arc.satellite] + self.SATELLITE_TRANS_TIME:
                    can_add = False
            
            if can_add:
                selected.append(arc_id)
                last_ground_end[arc.ground] = arc.link_et
                last_satellite_end[arc.satellite] = arc.link_et
        
        return selected
    
    def _resolve_conflicts_conservative(self, solution_arcs, conflicts):
        """
        保守的冲突解决策略：逐一检查，确保绝对无冲突
        """
        sorted_arcs = sorted(solution_arcs, key=lambda x: self.model.Arc_list[x].link_st)
        selected = []
        
        for arc_id in sorted_arcs:
            # 与已选弧段逐一检查冲突
            can_add = True
            for selected_arc in selected:
                test_list = [selected_arc, arc_id]
                test_conflicts = self._detect_conflicts_like_check(test_list)
                if test_conflicts:
                    can_add = False
                    break
            
            if can_add:
                selected.append(arc_id)
        
        return selected
    
    def verify_solution_like_check(self, solution_arcs):
        """
        使用与check.py完全一致的验证逻辑
        """
        conflicts = self._detect_conflicts_like_check(solution_arcs)
        
        if not conflicts:
            return True, "congratulations!!!"
        else:
            error_details = []
            for conflict in conflicts:
                error_details.append(f"error!!!! {conflict['arc1']} {conflict['arc2']} {conflict['error_code']}")
            return False, error_details
    
    def repair_multiple_solutions(self, solutions_list, verbose=False):
        """
        批量修复多个解
        """
        repaired_solutions = []
        
        for i, solution in enumerate(solutions_list):
            if verbose:
                print(f"修复第 {i+1}/{len(solutions_list)} 个解...")
            
            repaired = self.repair_solution(solution, verbose=False)
            if repaired:
                # 再次验证
                is_valid, _ = self.verify_solution_like_check(repaired)
                if is_valid:
                    repaired_solutions.append(repaired)
                elif verbose:
                    print(f"第 {i+1} 个解修复后仍有冲突")
        
        if verbose:
            print(f"成功修复 {len(repaired_solutions)}/{len(solutions_list)} 个解")
            
        return repaired_solutions

# 测试函数
def test_repairer_with_model(model, test_solution):
    """
    测试修复器是否能正确处理冲突
    """
    repairer = ImprovedConstraintRepairer(model)
    
    print("原始解验证:")
    is_valid_before, details_before = repairer.verify_solution_like_check(test_solution)
    print(f"验证结果: {is_valid_before}")
    if not is_valid_before:
        print("冲突详情:", details_before[:5])  # 只显示前5个冲突
    
    print("\n开始修复...")
    repaired_solution = repairer.repair_solution(test_solution, verbose=True)
    
    print("\n修复后验证:")
    is_valid_after, details_after = repairer.verify_solution_like_check(repaired_solution)
    print(f"验证结果: {is_valid_after}")
    
    if is_valid_after:
        original_value = sum(model.Arc_list[arc_id].link_time for arc_id in test_solution)
        repaired_value = sum(model.Arc_list[arc_id].link_time for arc_id in repaired_solution)
        print(f"价值变化: {original_value} -> {repaired_value} ({repaired_value/original_value*100:.1f}%)")
    
    return repaired_solution, is_valid_after