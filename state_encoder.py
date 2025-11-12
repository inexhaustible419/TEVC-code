# -*- coding: utf-8 -*-
# @Time    : 2025/11/11 15:24
# @Author  : JIA
# @FileName: state_encoder.py
# @Software: PyCharm
# @Blog    ：
"""
增强状态编码器 - 添加多种哈希签名来增强状态区分度
最小侵入式设计：充分利用现有Sol对象和Model对象
"""
import numpy as np
from collections import defaultdict


class EnhancedStateEncoder:
    """
    增强状态编码器
    核心思想：用多种哈希方法捕捉"选了哪些弧段"的选择模式
    """

    def __init__(self, model, n_hash_functions=10, n_bins=10):
        """
        Args:
            model: 你们的Model对象（包含Arc_list等信息）
            n_hash_functions: MinHash的哈希函数数量
            n_bins: 直方图分箱数
        """
        self.model = model
        self.n_hash_functions = n_hash_functions
        self.n_bins = n_bins

        # 预计算一些统计量，避免重复计算
        self.max_arc_id = len(model.Arc_list)

        # 按link_time排序的弧段索引（用于位置敏感哈希）
        self.sorted_arcs_by_lt = sorted(
            range(len(model.Arc_list)),
            key=lambda i: model.Arc_list[i].link_time,
            reverse=True
        )

        # 质数列表（用于模运算哈希）
        self.primes = [7, 11, 13, 17, 19]

    def encode_selection_signature(self, arc_ids):
        """
        核心方法：生成选择签名

        Args:
            arc_ids: list[int] - 选中的弧段ID列表（Sol.Arc_list_id）

        Returns:
            list[float] - 固定维度的签名特征（约50维）
        """
        if not arc_ids:
            # 空解返回零向量
            return [0.0] * self._get_signature_dim()

        signature = []
        arc_id_set = set(arc_ids)

        # ===== 1. MinHash签名（10维）=====
        # 捕捉集合相似度
        for seed in range(self.n_hash_functions):
            min_hash = min(hash((arc_id, seed)) % 100000 for arc_id in arc_ids)
            signature.append(min_hash / 100000.0)  # 归一化到[0,1]

        # ===== 2. 区间分布（10维）=====
        # 捕捉选中弧段在ID空间的分布
        bins = np.linspace(0, self.max_arc_id, self.n_bins + 1)
        hist, _ = np.histogram(arc_ids, bins=bins)
        hist_norm = hist / max(hist.sum(), 1)  # 归一化
        signature.extend(hist_norm)

        # ===== 3. 位置敏感哈希（3维）=====
        # 捕捉是否选中了高质量弧段
        top_positions = []
        for i, arc_id in enumerate(self.sorted_arcs_by_lt[:100]):  # 只看前100
            if arc_id in arc_id_set:
                top_positions.append(i)

        if top_positions:
            signature.extend([
                len(top_positions) / 20.0,  # 选中了多少个top弧段
                np.mean(top_positions) / 100.0,  # 平均位置
                np.std(top_positions) / 100.0 if len(top_positions) > 1 else 0
            ])
        else:
            signature.extend([0, 0, 0])

        # ===== 4. 模运算哈希（25维 = 5个质数 * 5维统计）=====
        # 快速捕捉ID分布模式
        for prime in self.primes:
            mod_counts = [0] * prime
            for arc_id in arc_ids:
                mod_counts[arc_id % prime] += 1

            # 归一化为分布
            mod_dist = np.array(mod_counts) / len(arc_ids)
            # signature.extend(mod_dist)
            # 【修复】计算5个统计数据，而不是 extend 整个列表
            stats = [
                np.mean(mod_dist),
                np.std(mod_dist),
                np.max(mod_dist),
                np.min(mod_dist),
                np.median(mod_dist)  # 或者 np.argmax(mod_dist) / prime
            ]
            signature.extend(stats)  # <--- 修复后的代码

        # ===== 5. 质量签名（2维）=====
        # 选中弧段的质量特征快速hash
        link_times = [self.model.Arc_list[i].link_time for i in arc_ids]
        signature.extend([
            hash(tuple(sorted(link_times[:10]))) % 10000 / 10000.0,  # 前10个的hash
            hash(sum(link_times)) % 10000 / 10000.0  # 总和的hash
        ])
        # print(signature)

        return signature

    def encode_quality_distribution(self, sol):
        """
        质量分布特征（基于现有Sol对象快速提取）

        Args:
            sol: Sol对象

        Returns:
            list[float] - 质量分布特征（约20维）
        """
        if not sol.Arc_list_id:
            return [0.0] * 20

        features = []
        arcs = [self.model.Arc_list[i] for i in sol.Arc_list_id]

        # 建链时长统计（6维）
        link_times = [arc.link_time for arc in arcs]
        features.extend([
            np.mean(link_times) / 1000.0,
            np.std(link_times) / 1000.0,
            np.min(link_times) / 1000.0,
            np.max(link_times) / 1000.0,
            np.percentile(link_times, 25) / 1000.0,
            np.percentile(link_times, 75) / 1000.0
        ])

        # 冲突度统计（3维）
        conflicts = [arc.conf for arc in arcs]
        features.extend([
            np.mean(conflicts) / 10000.0,
            np.std(conflicts) / 10000.0,
            sum(1 for c in conflicts if c == 0) / len(conflicts)  # 无冲突占比
        ])

        # 建链时长直方图（10维）
        hist, _ = np.histogram(link_times, bins=self.n_bins)
        hist_norm = hist / max(hist.sum(), 1)
        features.extend(hist_norm)

        # 地面站负载（1维 - 简化版）
        ground_usage = defaultdict(int)
        for arc in arcs:
            ground_usage[arc.ground] += 1
        features.append(np.std(list(ground_usage.values())) / 10.0 if ground_usage else 0)

        return features

    def encode_conflict_structure(self, sol):
        """
        冲突结构特征（快速版本）

        Args:
            sol: Sol对象

        Returns:
            list[float] - 冲突结构特征（约10维）
        """
        if not sol.Arc_list_id:
            return [0.0] * 10

        features = []
        arc_id_set = set(sol.Arc_list_id)

        # 内部冲突（3维）
        internal_conflicts = []
        for arc_id in sol.Arc_list_id:
            arc = self.model.Arc_list[arc_id]
            conf_count = sum(1 for conf_id in arc.confArc_list if conf_id in arc_id_set)
            internal_conflicts.append(conf_count)

        features.extend([
            np.mean(internal_conflicts) / 10.0 if internal_conflicts else 0,
            np.max(internal_conflicts) / 20.0 if internal_conflicts else 0,
            sum(1 for c in internal_conflicts if c == 0) / len(internal_conflicts) if internal_conflicts else 0
        ])

        # 外部冲突潜力（2维）
        external_conflict_arcs = set()
        for arc_id in sol.Arc_list_id:
            arc = self.model.Arc_list[arc_id]
            for c_id in arc.confArc_list:
                if c_id not in arc_id_set:
                    external_conflict_arcs.add(c_id)

        features.extend([
            len(external_conflict_arcs) / len(self.model.Arc_list),
            len(external_conflict_arcs) / max(len(sol.Arc_list_id), 1)
        ])

        # 资源使用分布熵（2维）
        satellite_usage = defaultdict(int)
        ground_usage = defaultdict(int)
        for arc_id in sol.Arc_list_id:
            arc = self.model.Arc_list[arc_id]
            satellite_usage[arc.satellite] += 1
            ground_usage[arc.ground] += 1

        def entropy(usage_dict):
            if not usage_dict:
                return 0
            total = sum(usage_dict.values())
            probs = [count / total for count in usage_dict.values()]
            return -sum(p * np.log(p + 1e-10) for p in probs)

        features.extend([
            entropy(satellite_usage) / 5.0,
            entropy(ground_usage) / 5.0
        ])

        # 填充到10维
        while len(features) < 10:
            features.append(0)

        return features[:10]

    def _get_signature_dim(self):
        """返回签名特征的维度"""
        # MinHash(10) + 区间分布(10) + 位置哈希(3) + 模运算(25) + 质量签名(2)
        return 50

    def get_enhanced_features(self, sol):
        """
        获取完整的增强特征
        这是主要的对外接口

        Args:
            sol: Sol对象

        Returns:
            list[float] - 增强特征向量（约80维）
        """
        features = []

        # 1. 选择签名（50维）
        features.extend(self.encode_selection_signature(sol.Arc_list_id))
        # print(features)

        # # 2. 质量分布（20维）
        # features.extend(self.encode_quality_distribution(sol))
        #
        # # 3. 冲突结构（10维）
        # features.extend(self.encode_conflict_structure(sol))

        return features


# ===== 便捷函数：用于快速集成到现有代码 =====
def create_encoder(model):
    """
    工厂函数：创建编码器实例

    Usage in ALNS.__init__:
        from StateEncoder import create_encoder
        self.state_encoder = create_encoder(self.model)
    """
    return EnhancedStateEncoder(model)


def get_signature_dim():
    """返回增强特征的总维度（用于DQN网络初始化）"""
    # 选择签名(50) + 质量分布(20) + 冲突结构(10)
    return 80