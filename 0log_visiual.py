# -*- coding: utf-8 -*-
# @Time    : 2025/10/11 11:34
# @Author  : JIA
# @FileName: 0log_visiual.py
# @Software: PyCharm
# @Blog    ：
"""
单Agent训练过程可视化分析工具
功能：
1. 全面的训练过程可视化
2. 算子效果深度分析
3. 学习行为洞察
4. 自动生成分析报告
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class AgentVisualizer:
    """单Agent可视化分析器"""

    def __init__(self, json_path, output_dir="visualization_output"):
        """
        初始化可视化器

        Args:
            json_path: JSON数据文件路径
            output_dir: 输出目录
        """
        self.json_path = json_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 加载数据
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # 提取关键数据
        self.metadata = self.data.get('metadata', {})
        self.steps_df = pd.DataFrame(self.data.get('steps', []))
        self.operator_stats = pd.DataFrame(self.data.get('operator_stats', []))
        self.summary = self.data.get('summary', {})

        self.fragment_id = self.metadata.get('fragment_id', 'unknown')
        self.iteration_id = self.metadata.get('iteration_id', 'unknown')

        print(f"✓ Loaded {len(self.steps_df)} steps for Fragment {self.fragment_id}")
        print(f"  Time span: {self.steps_df['timestamp'].iloc[-1]:.2f} seconds")
        print(f"  Best objective: {self.summary.get('best_obj', 0):.2f}")

    def generate_all_plots(self):
        """生成所有可视化图表"""
        print("\n" + "=" * 60)
        print("Generating all visualizations...")
        print("=" * 60 + "\n")

        # 1. 学习过程总览
        self.plot_training_overview()

        # 2. 算子效果分析
        self.plot_operator_analysis()

        # 3. Q值深度分析
        self.plot_q_value_analysis()

        # 4. 探索vs利用
        self.plot_exploration_exploitation()

        # 5. 收敛分析
        self.plot_convergence_analysis()

        # 6. 冲突演化
        self.plot_conflict_evolution()

        # 7. 算子热力图
        self.plot_operator_heatmaps()

        # 8. 综合仪表盘
        self.plot_dashboard()

        # --- 【新增】调用新的可视化函数 ---
        # 9. 状态演化
        self.plot_state_evolution()

        # 10. 奖励演化
        self.plot_reward_evolution()

        # 11. TD-Error演化
        self.plot_td_error_evolution()

        self.plot_boundary_conflict_evolution()

        print("\n" + "=" * 60)
        print(f"All visualizations saved to {self.output_dir}")
        print("=" * 60 + "\n")

    def plot_training_overview(self):
        """1. 训练过程总览 - 最核心的图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Training Overview - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        # (1) 目标值演化
        ax = axes[0, 0]
        ax.plot(self.steps_df['step'], self.steps_df['current_obj'],
                label='Current', alpha=0.6, linewidth=1)
        ax.plot(self.steps_df['step'], self.steps_df['best_obj'],
                label='Best', linewidth=2, color='red')

        # 标注关键点
        best_idx = self.steps_df['best_obj'].idxmax()
        ax.scatter(self.steps_df.loc[best_idx, 'step'],
                   self.steps_df.loc[best_idx, 'best_obj'],
                   color='gold', s=200, marker='*', zorder=5,
                   label=f'Peak: {self.steps_df.loc[best_idx, "best_obj"]:.0f}')

        ax.set_title('Objective Value Evolution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Link Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2) 改进率变化
        ax = axes[0, 1]
        # 计算滑动窗口改进率
        window = min(50, len(self.steps_df) // 10)
        improvement_rate = self.steps_df['improvement'].rolling(window).mean()

        ax.plot(self.steps_df['step'], improvement_rate, linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(self.steps_df['step'], 0, improvement_rate,
                        where=(improvement_rate > 0), alpha=0.3, color='green',
                        label='Improvement')
        ax.fill_between(self.steps_df['step'], 0, improvement_rate,
                        where=(improvement_rate < 0), alpha=0.3, color='red',
                        label='Deterioration')

        ax.set_title(f'Improvement Rate (Window={window})', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg Improvement')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (3) Loss和Q值
        ax = axes[1, 0]
        ax2 = ax.twinx()

        # Loss
        # valid_loss = self.steps_df[self.steps_df['loss'] > 0]
        # if len(valid_loss) > 0:
        #     ax.plot(valid_loss['step'], valid_loss['loss'],
        #             color='blue', alpha=0.7, label='Loss')
        #     ax.set_ylabel('Loss', color='blue')
        #     ax.tick_params(axis='y', labelcolor='blue')
        # 将loss为0的点替换为NaN，这样在绘图时会自动产生断点
        loss_series = self.steps_df['loss'].replace(0, np.nan)
        if not loss_series.isnull().all():
            ax.plot(self.steps_df['step'], loss_series,
                    color='blue', alpha=0.7, label='Loss')
            ax.set_ylabel('Loss', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

        # Q值
        ax2.plot(self.steps_df['step'], self.steps_df['q_avg'],
                 color='red', alpha=0.7, label='Avg Q-Value')
        ax2.set_ylabel('Q-Value', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title('Training Loss & Q-Value', fontweight='bold')
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)

        # (4) Epsilon衰减与接受率
        ax = axes[1, 1]
        ax2 = ax.twinx()

        # Epsilon
        ax.plot(self.steps_df['step'], self.steps_df['epsilon'],
                color='purple', linewidth=2, label='Epsilon')
        ax.set_ylabel('Epsilon', color='purple')
        ax.tick_params(axis='y', labelcolor='purple')

        # 接受率（滑动窗口）
        acceptance = self.steps_df['is_accepted'].rolling(window).mean()
        ax2.plot(self.steps_df['step'], acceptance,
                 color='green', linewidth=2, label='Acceptance Rate')
        ax2.set_ylabel('Acceptance Rate', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        ax.set_title('Exploration Strategy', fontweight='bold')
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_1_training_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 1_training_overview.png")

    def plot_operator_analysis(self):
        """2. 算子效果分析 - 关键洞察"""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Operator Analysis - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        # # (1) 算子选择频率对比
        # ax = fig.add_subplot(gs[0, 0])
        # destroy_counts = self.steps_df['destroy_selection_counts'].iloc[-1]
        # repair_counts = self.steps_df['repair_selection_counts'].iloc[-1]
        #
        # x = np.arange(8)
        # width = 0.35
        # ax.bar(x - width / 2, destroy_counts, width, label='Destroy', alpha=0.8)
        # ax.bar(x + width / 2, repair_counts, width, label='Repair', alpha=0.8)
        #
        # ax.set_title('Operator Selection Frequency', fontweight='bold')
        # ax.set_xlabel('Operator ID')
        # ax.set_ylabel('Selection Count')
        # ax.set_xticks(x)
        # ax.legend()
        # ax.grid(True, alpha=0.3, axis='y')

        # # (1) 算子选择频率演化（版本2）
        # ax = fig.add_subplot(gs[0, 0])
        # # 提取步数
        # steps = self.steps_df['step'].values
        # # 方式1：分别绘制Destroy和Repair（推荐，清晰度高）
        # # 绘制Destroy算子频率演化
        # for d_id in range(8):
        #     freq_series = self.steps_df['destroy_selection_freq'].apply(
        #         lambda x: x[d_id] if isinstance(x, (list, np.ndarray)) and len(x) > d_id else 0
        #     )
        #     ax.plot(steps, freq_series, label=f'D{d_id}', linewidth=2, alpha=0.8)
        # # 绘制Repair算子频率演化（使用虚线区分）
        # for r_id in range(8):
        #     freq_series = self.steps_df['repair_selection_freq'].apply(
        #         lambda x: x[r_id] if isinstance(x, (list, np.ndarray)) and len(x) > r_id else 0
        #     )
        #     ax.plot(steps, freq_series, label=f'R{r_id}', linewidth=2, alpha=0.7, linestyle='--')
        # ax.set_title('Operator Selection Frequency Evolution', fontweight='bold')
        # ax.set_xlabel('Step')
        # ax.set_ylabel('Cumulative Frequency')
        # ax.legend(ncol=4, fontsize=8, loc='best')
        # ax.grid(True, alpha=0.3)
        # ax.set_ylim([0, None])
        # # 添加参考线：如果均匀分布，每个算子频率应该是1/8
        # ax.axhline(y=1 / 8, color='red', linestyle=':', alpha=0.5, linewidth=1, label='Uniform (1/8)')

        # # (1a) Destroy算子频率演化
        # ax = fig.add_subplot(gs[0, 0])
        # steps = self.steps_df['step'].values
        #
        # for d_id in range(8):
        #     freq_series = self.steps_df['destroy_selection_freq'].apply(
        #         lambda x: x[d_id] if isinstance(x, (list, np.ndarray)) and len(x) > d_id else 0
        #     )
        #     ax.plot(steps, freq_series, label=f'D{d_id}', linewidth=2, alpha=0.8)
        #
        # # ax.axhline(y=1 / 8, color='red', linestyle='--', alpha=0.5, linewidth=1)
        # ax.axhline(y=1 / 8, color='red', linestyle='--', alpha=0.5, linewidth=1)
        # ax.set_title('Destroy Operator Frequency Evolution', fontweight='bold')
        # ax.set_xlabel('Step')
        # ax.set_ylabel('Frequency')
        # ax.legend(ncol=2, fontsize=9)
        # ax.grid(True, alpha=0.3)
        # ax.set_ylim([0, None])
        #
        # # (1b) Repair算子频率演化
        # ax = fig.add_subplot(gs[0, 1])  # 如果需要调整位置
        #
        # for r_id in range(8):
        #     freq_series = self.steps_df['repair_selection_freq'].apply(
        #         lambda x: x[r_id] if isinstance(x, (list, np.ndarray)) and len(x) > r_id else 0
        #     )
        #     ax.plot(steps, freq_series, label=f'R{r_id}', linewidth=2, alpha=0.8)
        #
        # # ax.axhline(y=1 / 8, color='red', linestyle='--', alpha=0.5, linewidth=1)
        # ax.axhline(x=1 / 8, color='red', linestyle='--', alpha=0.5, linewidth=1)
        # ax.set_title('Repair Operator Frequency Evolution', fontweight='bold')
        # ax.set_xlabel('Step')
        # ax.set_ylabel('Frequency')
        # ax.legend(ncol=2, fontsize=9)
        # ax.grid(True, alpha=0.3)
        # ax.set_ylim([0, None])

        # (1a) Destroy算子频率演化
        ax = fig.add_subplot(gs[0, 0])
        steps = self.steps_df['step'].values

        for d_id in range(8):
            freq_series = self.steps_df['destroy_selection_freq'].apply(
                lambda x: x[d_id] if isinstance(x, (list, np.ndarray)) and len(x) > d_id else 0
            )
            ax.plot(steps, freq_series, label=f'D{d_id}', linewidth=2, alpha=0.8)

        # ========== 标记关键训练阶段 ==========

        # 1. 标记首次训练步（batch_size步）
        batch_size = self.metadata.get('hyperparameters', {}).get('batch_size', 128)
        ax.axvline(x=batch_size, color='orange', linestyle='--', alpha=0.6, linewidth=2,
                   label=f'Training Start (step {batch_size})')

        # 2. 找出epsilon=0.5的步数（探索/利用平衡点）
        if 'epsilon' in self.steps_df.columns:
            epsilon_half_idx = (self.steps_df['epsilon'] - 0.5).abs().idxmin()
            epsilon_half_step = self.steps_df.loc[epsilon_half_idx, 'step']
            ax.axvline(x=epsilon_half_step, color='red', linestyle='--', alpha=0.6, linewidth=2,
                       label=f'ε=0.5 (step {epsilon_half_step:.0f})')

        # 3. 找出epsilon接近最小值的步数（主要使用Q值）
        epsilon_min = self.metadata.get('hyperparameters', {}).get('epsilon_min', 0.01)
        epsilon_low = epsilon_min * 2  # 2倍epsilon_min作为"主要利用"的阈值
        epsilon_low_mask = self.steps_df['epsilon'] <= epsilon_low
        if epsilon_low_mask.any():
            epsilon_low_step = self.steps_df.loc[epsilon_low_mask.idxmax(), 'step']
            ax.axvline(x=epsilon_low_step, color='green', linestyle='--', alpha=0.6, linewidth=2,
                       label=f'Q-dominant (ε≤{epsilon_low:.3f}, step {epsilon_low_step:.0f})')

        ax.set_title('Destroy Operator Frequency Evolution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Frequency')
        ax.legend(ncol=2, fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, None])

        # (1b) Repair算子频率演化
        ax = fig.add_subplot(gs[0, 1])  # 如果需要调整位置
        steps = self.steps_df['step'].values

        for r_id in range(8):
            freq_series = self.steps_df['repair_selection_freq'].apply(
                lambda x: x[r_id] if isinstance(x, (list, np.ndarray)) and len(x) > r_id else 0
            )
            ax.plot(steps, freq_series, label=f'R{r_id}', linewidth=2, alpha=0.8)

        # ========== 标记关键训练阶段 ==========

        # 1. 标记首次训练步（batch_size步）
        batch_size = self.metadata.get('hyperparameters', {}).get('batch_size', 128)
        ax.axvline(x=batch_size, color='orange', linestyle='--', alpha=0.6, linewidth=2,
                   label=f'Training Start (step {batch_size})')

        # 2. 找出epsilon=0.5的步数（探索/利用平衡点）
        if 'epsilon' in self.steps_df.columns:
            epsilon_half_idx = (self.steps_df['epsilon'] - 0.5).abs().idxmin()
            epsilon_half_step = self.steps_df.loc[epsilon_half_idx, 'step']
            ax.axvline(x=epsilon_half_step, color='red', linestyle='--', alpha=0.6, linewidth=2,
                       label=f'ε=0.5 (step {epsilon_half_step:.0f})')

        # 3. 找出epsilon接近最小值的步数（主要使用Q值）
        epsilon_min = self.metadata.get('hyperparameters', {}).get('epsilon_min', 0.01)
        epsilon_low = epsilon_min * 2  # 2倍epsilon_min作为"主要利用"的阈值
        epsilon_low_mask = self.steps_df['epsilon'] <= epsilon_low
        if epsilon_low_mask.any():
            epsilon_low_step = self.steps_df.loc[epsilon_low_mask.idxmax(), 'step']
            ax.axvline(x=epsilon_low_step, color='green', linestyle='--', alpha=0.6, linewidth=2,
                       label=f'Q-dominant (ε≤{epsilon_low:.3f}, step {epsilon_low_step:.0f})')

        ax.set_title('Repair Operator Frequency Evolution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Frequency')
        ax.legend(ncol=2, fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, None])


        # (2) 算子平均改进量
        ax = fig.add_subplot(gs[0, 2])
        if not self.operator_stats.empty:
            top_operators = self.operator_stats.nlargest(10, 'avg_improvement')
            colors = ['green' if x > 0 else 'red' for x in top_operators['avg_improvement']]

            bars = ax.barh(range(len(top_operators)),
                           top_operators['avg_improvement'],
                           color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_operators)))
            ax.set_yticklabels([f"D{row['destroy_id']}-R{row['repair_id']}"
                                for _, row in top_operators.iterrows()])
            ax.set_title('Top 10 Operators by Avg Improvement', fontweight='bold')
            ax.set_xlabel('Avg Improvement')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')

        # # (3) 算子接受率
        # ax = fig.add_subplot(gs[0, 2])
        # if not self.operator_stats.empty:
        #     sorted_ops = self.operator_stats.nlargest(10, 'selection_count')
        #
        #     ax.scatter(sorted_ops['selection_count'],
        #                sorted_ops['acceptance_rate'],
        #                s=sorted_ops['avg_improvement'].abs() * 5,
        #                alpha=0.6, c=sorted_ops['avg_improvement'],
        #                cmap='RdYlGn')
        #
        #     for _, row in sorted_ops.iterrows():
        #         ax.annotate(f"D{row['destroy_id']}-R{row['repair_id']}",
        #                     (row['selection_count'], row['acceptance_rate']),
        #                     fontsize=8, alpha=0.7)
        #
        #     ax.set_title('Operator Performance Map', fontweight='bold')
        #     ax.set_xlabel('Selection Count')
        #     ax.set_ylabel('Acceptance Rate')
        #     ax.grid(True, alpha=0.3)

        # (4) 算子价值演化（Destroy）
        ax = fig.add_subplot(gs[1, 0])
        for i in range(8):
            values = [step['destroy_values'][i] for step in self.data['steps']]
            ax.plot(values, label=f'D{i}', alpha=0.7)

        ax.set_title('Destroy Operator Value Evolution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Operator Value')
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)

        # (5) 算子价值演化（Repair）
        ax = fig.add_subplot(gs[1, 1])
        for i in range(8):
            values = [step['repair_values'][i] for step in self.data['steps']]
            ax.plot(values, label=f'R{i}', alpha=0.7)

        ax.set_title('Repair Operator Value Evolution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Operator Value')
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)

        # (6) 算子效率对比
        ax = fig.add_subplot(gs[1, 2])
        if not self.operator_stats.empty:
            # 计算效率：改进量/选择次数
            ops = self.operator_stats[self.operator_stats['selection_count'] > 5].copy()
            ops['efficiency'] = ops['total_improvement'] / ops['selection_count']
            ops = ops.nlargest(10, 'efficiency')

            colors = ['green' if x > 0 else 'red' for x in ops['efficiency']]
            ax.barh(range(len(ops)), ops['efficiency'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(ops)))
            ax.set_yticklabels([f"D{row['destroy_id']}-R{row['repair_id']}"
                                for _, row in ops.iterrows()])
            ax.set_title('Operator Efficiency (Improvement/Selection)', fontweight='bold')
            ax.set_xlabel('Efficiency')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')

        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_2_operator_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 2_operator_analysis.png")

    def plot_q_value_analysis(self):
        """3. Q值深度分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Q-Value Analysis - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        # (1) Q值分布演化
        ax = axes[0, 0]
        # 分段采样避免过密
        sample_steps = np.linspace(0, len(self.steps_df) - 1, min(10, len(self.steps_df)), dtype=int)

        for idx in sample_steps:
            step = self.data['steps'][idx]
            q_all = step['q_all_values']
            ax.violinplot([q_all], positions=[step['step']], widths=len(self.steps_df) // 20)

        ax.plot(self.steps_df['step'], self.steps_df['q_avg'],
                'r-', linewidth=2, label='Avg Q')
        ax.plot(self.steps_df['step'], self.steps_df['q_max'],
                'g--', linewidth=1, label='Max Q')

        ax.set_title('Q-Value Distribution Evolution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Q-Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2) Q值方差变化（学习稳定性）
        ax = axes[0, 1]
        ax.plot(self.steps_df['step'], self.steps_df['q_std'], linewidth=2)
        ax.fill_between(self.steps_df['step'], 0, self.steps_df['q_std'], alpha=0.3)

        ax.set_title('Q-Value Standard Deviation (Learning Stability)', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Q-Value Std')
        ax.grid(True, alpha=0.3)

        # (3) Q值Gap（探索空间）
        ax = axes[1, 0]
        ax.plot(self.steps_df['step'], self.steps_df['q_gap'], linewidth=2)

        # 添加趋势线
        z = np.polyfit(self.steps_df['step'], self.steps_df['q_gap'], 2)
        p = np.poly1d(z)
        ax.plot(self.steps_df['step'], p(self.steps_df['step']),
                "r--", alpha=0.5, linewidth=2, label='Trend')

        ax.set_title('Q-Gap (Max Q - Selected Q)', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Q-Gap')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (4) Destroy vs Repair Q值对比
        ax = axes[1, 1]

        # 计算平均Q值
        destroy_q_avg = [np.mean([step['q_destroy_avg'][i] for step in self.data['steps']])
                         for i in range(8)]
        repair_q_avg = [np.mean([step['q_repair_avg'][i] for step in self.data['steps']])
                        for i in range(8)]

        x = np.arange(8)
        width = 0.35
        ax.bar(x - width / 2, destroy_q_avg, width, label='Destroy', alpha=0.8)
        ax.bar(x + width / 2, repair_q_avg, width, label='Repair', alpha=0.8)

        ax.set_title('Average Q-Value by Operator Type', fontweight='bold')
        ax.set_xlabel('Operator ID')
        ax.set_ylabel('Avg Q-Value')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_3_q_value_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 3_q_value_analysis.png")

    def plot_exploration_exploitation(self):
        """4. 探索vs利用分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Exploration vs Exploitation - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        # (1) 探索/利用比例变化
        ax = axes[0, 0]
        window = min(100, len(self.steps_df) // 10)
        exploration_rate = self.steps_df['is_exploration'].rolling(window).mean()

        ax.plot(self.steps_df['step'], exploration_rate, linewidth=2, label='Exploration Rate')
        ax.plot(self.steps_df['step'], 1 - exploration_rate, linewidth=2, label='Exploitation Rate')
        ax.fill_between(self.steps_df['step'], 0, exploration_rate, alpha=0.3)
        ax.fill_between(self.steps_df['step'], exploration_rate, 1, alpha=0.3)

        ax.set_title(f'Exploration Rate (Window={window})', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2) 探索步骤的效果
        ax = axes[0, 1]
        explore_steps = self.steps_df[self.steps_df['is_exploration'] == True]
        exploit_steps = self.steps_df[self.steps_df['is_exploration'] == False]

        data = [
            explore_steps['improvement'].values,
            exploit_steps['improvement'].values
        ]
        labels = ['Exploration', 'Exploitation']

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
            patch.set_facecolor(color)

        ax.set_title('Improvement Distribution by Strategy', fontweight='bold')
        ax.set_ylabel('Improvement')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # (3) 探索成功率随时间变化
        ax = axes[1, 0]
        explore_accept = (self.steps_df['is_exploration'] & self.steps_df['is_accepted']).rolling(window).sum()
        explore_total = self.steps_df['is_exploration'].rolling(window).sum()
        explore_success = explore_accept / explore_total.replace(0, 1)

        exploit_accept = (~self.steps_df['is_exploration'] & self.steps_df['is_accepted']).rolling(window).sum()
        exploit_total = (~self.steps_df['is_exploration']).rolling(window).sum()
        exploit_success = exploit_accept / exploit_total.replace(0, 1)

        ax.plot(self.steps_df['step'], explore_success, label='Exploration', linewidth=2)
        ax.plot(self.steps_df['step'], exploit_success, label='Exploitation', linewidth=2)

        ax.set_title('Success Rate by Strategy', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Success Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (4) Epsilon vs Performance
        ax = axes[1, 1]
        ax2 = ax.twinx()

        # Epsilon
        ax.plot(self.steps_df['step'], self.steps_df['epsilon'],
                color='purple', linewidth=2, label='Epsilon')
        ax.set_ylabel('Epsilon', color='purple')
        ax.tick_params(axis='y', labelcolor='purple')

        # 累计最优值
        cumulative_best = self.steps_df['best_obj'].expanding().max()
        ax2.plot(self.steps_df['step'], cumulative_best,
                 color='red', linewidth=2, label='Cumulative Best')
        ax2.set_ylabel('Cumulative Best Obj', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title('Epsilon Decay vs Performance', fontweight='bold')
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_4_exploration_exploitation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 4_exploration_exploitation.png")

    def plot_convergence_analysis(self):
        """5. 收敛分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Convergence Analysis - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        # (1) 目标值收敛曲线
        ax = axes[0, 0]
        ax.plot(self.steps_df['step'], self.steps_df['best_obj'], linewidth=2)

        # 标注收敛点
        converged = self.summary.get('converged', False)
        if converged:
            conv_step = self.summary.get('convergence_step', len(self.steps_df))
            ax.axvline(x=conv_step, color='red', linestyle='--',
                       label=f'Convergence at step {conv_step}')
            ax.legend()

        ax.set_title('Convergence Curve', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Best Objective')
        ax.grid(True, alpha=0.3)

        # (2) 改进幅度衰减
        ax = axes[0, 1]
        # 计算每次更新最优解时的改进幅度
        best_improved = self.steps_df[self.steps_df['is_best_improved'] == True]

        if len(best_improved) > 0:
            improvements = best_improved['improvement'].values
            steps = best_improved['step'].values

            ax.scatter(steps, improvements, s=100, alpha=0.6)

            # 拟合指数衰减
            if len(improvements) > 3:
                z = np.polyfit(range(len(improvements)), improvements, 2)
                p = np.poly1d(z)
                ax.plot(steps, p(range(len(improvements))),
                        'r--', linewidth=2, label='Trend')
                ax.legend()

        ax.set_title('Improvement Magnitude Decay', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Improvement when Best Updated')
        ax.grid(True, alpha=0.3)

        # (3) 滑动方差（稳定性）
        ax = axes[1, 0]
        windows = [10, 50, 100]
        for w in windows:
            if len(self.steps_df) > w:
                variance = self.steps_df['best_obj'].rolling(w).var()
                ax.plot(self.steps_df['step'], variance, label=f'Window={w}')

        ax.set_title('Best Objective Variance (Stability)', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # (4) 无改进持续时间分布
        ax = axes[1, 1]

        # 计算连续无改进的步数分布
        no_improve_lengths = []
        current_length = 0

        for _, row in self.steps_df.iterrows():
            if row['is_best_improved']:
                if current_length > 0:
                    no_improve_lengths.append(current_length)
                current_length = 0
            else:
                current_length += 1

        if no_improve_lengths:
            ax.hist(no_improve_lengths, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(x=np.mean(no_improve_lengths), color='red',
                       linestyle='--', label=f'Mean={np.mean(no_improve_lengths):.1f}')
            ax.legend()

        ax.set_title('Distribution of Plateau Lengths', fontweight='bold')
        ax.set_xlabel('Steps without Improvement')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_5_convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 5_convergence_analysis.png")

    def plot_conflict_evolution(self):
        """6. 冲突演化分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Conflict Evolution - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        # (1) 冲突数量变化
        ax = axes[0, 0]
        ax.plot(self.steps_df['step'], self.steps_df['internal_conflicts'],
                label='Internal', linewidth=2)
        ax.plot(self.steps_df['step'], self.steps_df['external_conflict_count'],
                label='External', linewidth=2)

        ax.set_title('Conflict Count Evolution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Conflict Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2) 边界风险密度
        ax = axes[0, 1]
        ax.plot(self.steps_df['step'], self.steps_df['boundary_risk_density'],
                linewidth=2, color='orange')
        ax.fill_between(self.steps_df['step'], 0, self.steps_df['boundary_risk_density'],
                        alpha=0.3, color='orange')

        ax.set_title('Boundary Risk Density', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Risk Density')
        ax.grid(True, alpha=0.3)

        # (3) 冲突vs目标值
        ax = axes[1, 0]
        ax2 = ax.twinx()

        ax.plot(self.steps_df['step'], self.steps_df['internal_conflicts'],
                color='red', alpha=0.7, label='Conflicts')
        ax2.plot(self.steps_df['step'], self.steps_df['best_obj'],
                 color='blue', alpha=0.7, label='Best Obj')

        ax.set_ylabel('Internal Conflicts', color='red')
        ax2.set_ylabel('Best Objective', color='blue')
        ax.set_xlabel('Step')
        ax.set_title('Conflicts vs Objective Trade-off', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # (4) 解的规模变化
        ax = axes[1, 1]
        ax.plot(self.steps_df['step'], self.steps_df['current_solution_size'],
                label='Current Size', alpha=0.7)
        ax.plot(self.steps_df['step'], self.steps_df['best_solution_size'],
                label='Best Size', linewidth=2)

        ax.set_title('Solution Size Evolution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Number of Arcs')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_6_conflict_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 6_conflict_evolution.png")

    def plot_operator_heatmaps(self):
        """7. 算子热力图"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Operator Heatmaps - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        if self.operator_stats.empty:
            plt.close()
            return

        # 创建8x8矩阵
        selection_matrix = np.zeros((8, 8))
        improvement_matrix = np.zeros((8, 8))
        acceptance_matrix = np.zeros((8, 8))

        for _, row in self.operator_stats.iterrows():
            d, r = int(row['destroy_id']), int(row['repair_id'])
            selection_matrix[d, r] = row['selection_count']
            improvement_matrix[d, r] = row['avg_improvement']
            acceptance_matrix[d, r] = row['acceptance_rate']

        # (1) 选择频率热力图
        ax = axes[0]
        sns.heatmap(selection_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                    ax=ax, cbar_kws={'label': 'Selection Count'})
        ax.set_title('Selection Frequency Heatmap', fontweight='bold')
        ax.set_xlabel('Repair Operator')
        ax.set_ylabel('Destroy Operator')

        # (2) 平均改进热力图
        ax = axes[1]
        sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                    center=0, ax=ax, cbar_kws={'label': 'Avg Improvement'})
        ax.set_title('Average Improvement Heatmap', fontweight='bold')
        ax.set_xlabel('Repair Operator')
        ax.set_ylabel('Destroy Operator')

        # (3) 接受率热力图
        ax = axes[2]
        sns.heatmap(acceptance_matrix, annot=True, fmt='.2f', cmap='Blues',
                    ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Acceptance Rate'})
        ax.set_title('Acceptance Rate Heatmap', fontweight='bold')
        ax.set_xlabel('Repair Operator')
        ax.set_ylabel('Destroy Operator')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_7_operator_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 7_operator_heatmaps.png")

    def plot_dashboard(self):
        """8. 综合仪表盘"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Training Dashboard - Fragment {self.fragment_id} (Iteration {self.iteration_id})',
                     fontsize=18, fontweight='bold')

        # 左上：关键指标卡片
        ax = fig.add_subplot(gs[0, :2])
        ax.axis('off')

        metrics_text = f"""
        TRAINING SUMMARY

        Initial Objective:     {self.summary.get('initial_obj', 0):>12,.0f}
        Final Objective:       {self.summary.get('final_obj', 0):>12,.0f}
        Best Objective:        {self.summary.get('best_obj', 0):>12,.0f}
        Total Improvement:     {self.summary.get('total_improvement', 0):>12,.0f}
        Improvement Rate:      {self.summary.get('improvement_rate', 0) * 100:>11.2f}%

        Total Steps:           {self.summary.get('total_steps', 0):>12,}
        Accepted Steps:        {self.summary.get('accepted_steps', 0):>12,}
        Acceptance Rate:       {self.summary.get('acceptance_rate', 0) * 100:>11.2f}%

        Training Time:         {self.summary.get('total_time', 0):>11.2f}s
        Avg Step Time:         {self.summary.get('avg_step_time', 0) * 1000:>11.2f}ms

        Converged:             {str(self.summary.get('converged', False)):>12}
        Final Epsilon:         {self.summary.get('final_epsilon', 0):>12.4f}
        """

        ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # 右上：最佳算子组合
        ax = fig.add_subplot(gs[0, 2:])
        ax.axis('off')

        if not self.operator_stats.empty:
            top5 = self.operator_stats.nlargest(5, 'total_improvement')

            top_text = "TOP 5 OPERATOR COMBINATIONS\n\n"
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                top_text += f"{i}. D{row['destroy_id']}-R{row['repair_id']}: "
                top_text += f"Improvement={row['total_improvement']:.0f}, "
                top_text += f"Count={row['selection_count']}, "
                top_text += f"Accept={row['acceptance_rate']:.2%}\n"

            ax.text(0.1, 0.5, top_text, transform=ax.transAxes,
                    fontsize=11, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 中间：主要曲线
        # 目标值
        ax = fig.add_subplot(gs[1, :2])
        ax.plot(self.steps_df['step'], self.steps_df['best_obj'], 'r-', linewidth=2)
        ax.set_title('Best Objective Evolution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Objective')
        ax.grid(True, alpha=0.3)

        # Q值
        ax = fig.add_subplot(gs[1, 2:])
        ax.plot(self.steps_df['step'], self.steps_df['q_avg'], linewidth=2)
        ax.fill_between(self.steps_df['step'],
                        self.steps_df['q_avg'] - self.steps_df['q_std'],
                        self.steps_df['q_avg'] + self.steps_df['q_std'],
                        alpha=0.3)
        ax.set_title('Q-Value Evolution (Mean ± Std)', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Q-Value')
        ax.grid(True, alpha=0.3)

        # 底部：算子分析
        # 选择频率
        ax = fig.add_subplot(gs[2, 0])
        destroy_counts = self.steps_df['destroy_selection_counts'].iloc[-1]
        ax.bar(range(8), destroy_counts, alpha=0.7)
        ax.set_title('Destroy Operator Usage', fontweight='bold')
        ax.set_xlabel('Operator ID')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')

        ax = fig.add_subplot(gs[2, 1])
        repair_counts = self.steps_df['repair_selection_counts'].iloc[-1]
        ax.bar(range(8), repair_counts, alpha=0.7, color='green')
        ax.set_title('Repair Operator Usage', fontweight='bold')
        ax.set_xlabel('Operator ID')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')

        # 探索vs利用
        ax = fig.add_subplot(gs[2, 2])
        explore_count = self.summary.get('exploration_count', 0)
        exploit_count = self.summary.get('exploitation_count', 0)
        ax.pie([explore_count, exploit_count],
               labels=['Exploration', 'Exploitation'],
               autopct='%1.1f%%', startangle=90,
               colors=['lightblue', 'lightgreen'])
        ax.set_title('Exploration vs Exploitation', fontweight='bold')

        # 接受vs拒绝
        ax = fig.add_subplot(gs[2, 3])
        accepted = self.summary.get('accepted_steps', 0)
        rejected = self.summary.get('total_steps', 0) - accepted
        ax.pie([accepted, rejected],
               labels=['Accepted', 'Rejected'],
               autopct='%1.1f%%', startangle=90,
               colors=['lightgreen', 'lightcoral'])
        ax.set_title('Acceptance vs Rejection', fontweight='bold')

        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_8_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 8_dashboard.png")

        # ... (在 plot_dashboard 方法之前添加以下三个函数) ...

    # def plot_state_evolution(self):
    #     """新增：9. 状态特征演化图"""
    #     if 'state' not in self.steps_df.columns or self.steps_df['state'].isnull().all():
    #         print("✗ Skipping state evolution plot: No state data found.")
    #         return
    #
    #     fig, ax = plt.subplots(figsize=(18, 8))
    #     fig.suptitle(f'State Feature Evolution - Fragment {self.fragment_id}',
    #                  fontsize=16, fontweight='bold')
    #
    #     # 将DataFrame中的state列表转换为一个包含17列的新DataFrame
    #     state_df = pd.DataFrame(self.steps_df['state'].tolist(),
    #                             columns=[f'S{i}' for i in range(17)])
    #
    #     # 归一化，便于在同一张图上显示
    #     normalized_state_df = (state_df - state_df.mean()) / (state_df.std() + 1e-8)
    #
    #     # 绘制每一维状态的变化
    #     for i in range(4):
    #         ax.plot(self.steps_df['step'], normalized_state_df[f'S{i}'],
    #                 label=f'State_{i}', alpha=0.7, linewidth=1.5)
    #
    #     ax.set_title('Normalized State Features Over Time', fontweight='bold')
    #     ax.set_xlabel('Step')
    #     ax.set_ylabel('Normalized Value (Z-score)')
    #     ax.legend(ncol=6, fontsize=9, loc='upper right')
    #     ax.grid(True, alpha=0.3)
    #     ax.set_ylim(-5, 5)  # 限制Y轴范围，避免极端值影响
    #
    #     plt.tight_layout()
    #     plt.savefig(self.output_dir / '9_state_evolution.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print("✓ Generated: 9_state_evolution.png")
    # def plot_state_evolution(self):
    #     """新增：9. 状态特征演化图"""
    #     if 'state' not in self.steps_df.columns or self.steps_df['state'].isnull().all():
    #         print("✗ Skipping state evolution plot: No state data found.")
    #         return
    #
    #     # 将DataFrame中的state列表转换为一个包含17列的新DataFrame
    #     state_df = pd.DataFrame(self.steps_df['state'].tolist(),
    #                             columns=[f'S{i}' for i in range(17)])
    #
    #     # 归一化，便于在同一张图上显示
    #     normalized_state_df = (state_df - state_df.mean()) / (state_df.std() + 1e-8)
    #
    #     # 计算需要多少个子图（4条线一组）
    #     num_features = 17
    #     lines_per_plot = 4
    #     num_plots = (num_features + lines_per_plot - 1) // lines_per_plot  # 向上取整
    #
    #     # 创建子图
    #     fig, axes = plt.subplots(num_plots, 1, figsize=(18, 5 * num_plots))
    #     fig.suptitle(f'State Feature Evolution - Fragment {self.fragment_id}',
    #                  fontsize=16, fontweight='bold')
    #
    #     # 如果只有一个子图，确保axes是列表形式
    #     if num_plots == 1:
    #         axes = [axes]
    #
    #     # 在每个子图中绘制4条线
    #     for plot_idx in range(num_plots):
    #         start_idx = plot_idx * lines_per_plot
    #         end_idx = min((plot_idx + 1) * lines_per_plot, num_features)
    #
    #         ax = axes[plot_idx]
    #         for i in range(start_idx, end_idx):
    #             ax.plot(self.steps_df['step'], normalized_state_df[f'S{i}'],
    #                     label=f'State_{i}', alpha=0.7, linewidth=1.5)
    #
    #         ax.set_title(f'State Features {start_idx}-{end_idx - 1}', fontweight='bold')
    #         ax.set_xlabel('Step')
    #         ax.set_ylabel('Normalized Value (Z-score)')
    #         ax.legend(ncol=4, fontsize=9, loc='upper right')
    #         ax.grid(True, alpha=0.3)
    #         ax.set_ylim(-5, 5)  # 限制Y轴范围，避免极端值影响
    #
    #     plt.tight_layout()
    #     plt.savefig(self.output_dir / '9_state_evolution.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print("✓ Generated: 9_state_evolution.png")
    #
    # def plot_reward_evolution(self):
    #     """新增：10. 奖励变化与构成图"""
    #     fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    #     fig.suptitle(f'Reward Evolution - Fragment {self.fragment_id}',
    #                  fontsize=16, fontweight='bold')
    #
    #     window = min(20, len(self.steps_df) // 10)
    #
    #     # (1) 总奖励滑动平均
    #     ax = axes[0]
    #     reward_smooth = self.steps_df['reward'].rolling(window).mean()
    #     ax.plot(self.steps_df['step'], reward_smooth, linewidth=2, label=f'Total Reward (smooth, w={window})')
    #     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    #
    #     ax.set_title('Smoothed Total Reward', fontweight='bold')
    #     ax.set_ylabel('Reward')
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)
    #
    #     # (2) 奖励构成（堆叠面积图）
    #     ax = axes[1]
    #     reward_components = [
    #         'reward_link_time',
    #         'reward_diversity',
    #         'reward_efficiency',
    #         'reward_conflict',  # 负向奖励
    #         'reward_boundary'  # 负向奖励
    #     ]
    #
    #     # 对每个组件进行滑动平均
    #     component_data = {
    #         comp: self.steps_df[comp].rolling(window).mean() for comp in reward_components
    #     }
    #
    #     labels = [label.replace('reward_', '').capitalize() for label in reward_components]
    #
    #     # 逐条绘制线条
    #     for comp, label in zip(reward_components, labels):
    #         ax.plot(self.steps_df['step'], component_data[comp], linewidth=2, label=label)
    #     # ax.stackplot(self.steps_df['step'], component_data.values(),
    #     #              labels=labels, alpha=0.8)
    #
    #     ax.set_title(f'Reward Composition (smooth, w={window})', fontweight='bold')
    #     ax.set_xlabel('Step')
    #     ax.set_ylabel('Reward Value')
    #     ax.legend(loc='lower left')
    #     ax.grid(True, alpha=0.3)
    #     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    #
    #     plt.tight_layout()
    #     plt.savefig(self.output_dir / '10_reward_evolution.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print("✓ Generated: 10_reward_evolution.png")
    # 【请用此版本完整替换旧函数】
    def plot_state_evolution(self):
        """新增：9. 状态特征演化图 (适配15维)"""
        if 'state' not in self.steps_df.columns or self.steps_df['state'].isnull().all():
            print("✗ Skipping state evolution plot: No state data found.")
            return

        # 动态获取状态维度
        num_features = len(self.steps_df['state'].iloc[0])
        # print(f"  Detected {num_features}-dimensional state vector.")

        # 将DataFrame中的state列表转换为一个包含多列的新DataFrame
        state_df = pd.DataFrame(self.steps_df['state'].tolist(),
                                columns=[f'S{i}' for i in range(num_features)])

        # 归一化
        normalized_state_df = (state_df - state_df.mean()) / (state_df.std() + 1e-8)

        lines_per_plot = 4
        num_plots = (num_features + lines_per_plot - 1) // lines_per_plot

        fig, axes = plt.subplots(num_plots, 1, figsize=(18, 5 * num_plots), sharex=True)
        fig.suptitle(f'State Feature Evolution - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        if num_plots == 1: axes = [axes]

        for plot_idx in range(num_plots):
            ax = axes[plot_idx]
            start_idx = plot_idx * lines_per_plot
            end_idx = min((plot_idx + 1) * lines_per_plot, num_features)

            for i in range(start_idx, end_idx):
                # 为新增的特征提供更清晰的标签
                state_labels = {
                    10: "Boundary_Density", 11: "Stagnation", 12: "Improvement_Rate",
                    13: "Left_Conflict", 14: "Right_Conflict"
                }
                label = f'State_{i} ({state_labels.get(i, "")})'
                ax.plot(self.steps_df['step'], normalized_state_df[f'S{i}'], label=label, alpha=0.8, linewidth=1.5)

            ax.set_title(f'State Features {start_idx}-{end_idx - 1}', fontweight='bold')
            ax.set_ylabel('Normalized Value (Z-score)')
            ax.legend(ncol=4, fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-5, 5)

        axes[-1].set_xlabel('Step')  # 只在最下面的图表显示x轴标签
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_9_state_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 9_state_evolution.png")

    # 【请用此版本完整替换旧函数】
    # def plot_reward_evolution(self):
    #     """新增：10. 奖励变化与构成图 (适配新奖励)"""
    #     fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    #     fig.suptitle(f'Reward Evolution - Fragment {self.fragment_id}',
    #                  fontsize=16, fontweight='bold')
    #
    #     window = min(20, len(self.steps_df) // 10)
    #
    #     # (1) 总奖励滑动平均
    #     ax = axes[0]
    #     reward_smooth = self.steps_df['reward'].rolling(window).mean()
    #     ax.plot(self.steps_df['step'], reward_smooth, linewidth=2, label=f'Total Reward (smooth, w={window})',
    #             color='red')
    #     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    #     ax.set_title('Smoothed Total Reward', fontweight='bold')
    #     ax.set_ylabel('Reward')
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)
    #
    #     # (2) 新奖励构成
    #     ax = axes[1]
    #     # 这是我们新的奖励分量
    #     reward_components = [
    #         'reward_link_time',
    #         'reward_improvement',
    #         'reward_structure'
    #     ]
    #     labels = ['Link Time (Main Obj)', 'Conflict Improvement (Coordination)', 'Structure (Efficiency/Diversity)']
    #
    #     for comp, label in zip(reward_components, labels):
    #         if comp in self.steps_df.columns:
    #             comp_smooth = self.steps_df[comp].rolling(window).mean()
    #             ax.plot(self.steps_df['step'], comp_smooth, linewidth=2, label=label)
    #
    #     ax.set_title(f'Reward Composition (smooth, w={window})', fontweight='bold')
    #     ax.set_xlabel('Step')
    #     ax.set_ylabel('Reward Value')
    #     ax.legend(loc='lower left')
    #     ax.grid(True, alpha=0.3)
    #     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    #
    #     plt.tight_layout(rect=[0, 0, 1, 0.97])
    #     plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_10_reward_evolution.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print("✓ Generated: 10_reward_evolution.png")

    # 【请用此版本完整替换旧函数】
    def plot_reward_evolution(self):
        """新增：10. 奖励变化与构成图 (适配新奖励)"""
        fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
        fig.suptitle(f'Reward Evolution - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        # 确保窗口大小在合理范围内
        window = min(50, len(self.steps_df) // 10) if len(self.steps_df) > 10 else 1

        # (1) 总奖励滑动平均
        ax = axes[0]
        reward_smooth = self.steps_df['reward'].rolling(window).mean()
        ax.plot(self.steps_df['step'], reward_smooth, linewidth=2, label=f'Total Reward (smooth, w={window})',
                color='red')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Smoothed Total Reward', fontweight='bold')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2) 新奖励构成
        ax = axes[1]
        # 这是我们新的奖励分量键名
        reward_components = [
            'reward_link_time',
            'reward_improvement',
            # 'reward_structure',
            'reward_boundary'
        ]
        labels = [
            'Link Time (Main Obj)',
            'Conflict Improvement',
            # 'Structure (Efficiency/Diversity)',
            'Boundary Penalty'
        ]

        for comp, label in zip(reward_components, labels):
            if comp in self.steps_df.columns:
                comp_smooth = self.steps_df[comp].rolling(window).mean()
                ax.plot(self.steps_df['step'], comp_smooth, linewidth=2, label=label)

        ax.set_title(f'Reward Composition (smooth, w={window})', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward Value')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_10_reward_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 10_reward_evolution.png")

    # 【新增此函数】
    def plot_boundary_conflict_evolution(self):
        """新增：12. 边界冲突演化图 (衡量协同效果)"""
        # 从state向量中提取左右邻居冲突数据
        left_conflict = self.steps_df['state'].apply(lambda s: s[13] if len(s) > 13 else 0)
        right_conflict = self.steps_df['state'].apply(lambda s: s[14] if len(s) > 14 else 0)

        fig, ax = plt.subplots(figsize=(18, 6))
        fig.suptitle(f'Boundary Conflict Evolution - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        window = min(50, len(self.steps_df) // 10)

        # 绘制滑动平均值
        ax.plot(self.steps_df['step'], left_conflict.rolling(window).mean(),
                label=f'Left Neighbor Conflict (smooth, w={window})')
        ax.plot(self.steps_df['step'], right_conflict.rolling(window).mean(),
                label=f'Right Neighbor Conflict (smooth, w={window})')

        ax.set_title('Real-time Conflict with Neighbors (from Blackboard)', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Normalized Conflict Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_12_boundary_conflict_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 12_boundary_conflict_evolution.png")



    def plot_td_error_evolution(self):
        """新增：11. TD-Error演化图"""
        if 'td_error' not in self.steps_df.columns or (self.steps_df['td_error'] == 0).all():
            print("✗ Skipping TD-Error plot: No TD-Error data found.")
            return


        fig, ax = plt.subplots(figsize=(18, 6))
        fig.suptitle(f'TD-Error Evolution - Fragment {self.fragment_id}',
                     fontsize=16, fontweight='bold')

        window = min(20, len(self.steps_df) // 10)

        # 过滤掉为0的无效值
        # td_error_series = self.steps_df['td_error'].replace(0, np.nan)
        # td_error_smooth = td_error_series.rolling(window).mean()
        # ax.plot(self.steps_df['step'], td_error_smooth, linewidth=2, label=f'TD-Error (smooth, w={window})')
        ax.plot(self.steps_df['step'], self.steps_df['td_error'], linewidth=2, label=f'TD-Error (smooth, w={window})')

        ax.set_title('Temporal-Difference Error (Learning Signal)', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Average TD-Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'Fragment_{self.fragment_id}_11_td_error_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 11_td_error_evolution.png")

    def export_to_excel(self, excel_path=None):
        """导出数据到Excel（可选）"""
        if excel_path is None:
            excel_path = self.output_dir / f'fragment_{self.fragment_id}_data.xlsx'

        print(f"\nExporting data to Excel: {excel_path}")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Steps详细数据
            self.steps_df.to_excel(writer, sheet_name='Steps', index=False)

            # Sheet 2: Operator统计
            if not self.operator_stats.empty:
                self.operator_stats.to_excel(writer, sheet_name='Operator_Stats', index=False)

            # Sheet 3: Summary
            summary_df = pd.DataFrame([self.summary])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Sheet 4: Metadata
            metadata_df = pd.DataFrame([self.metadata])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

        print(f"✓ Excel file saved: {excel_path}")


# ========== 主函数 ==========

def main():
    """主函数 - 示例用法"""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize agent training data')
    parser.add_argument('json_file', help='Path to the JSON data file')
    parser.add_argument('--output_dir', default='visualization_output',
                        help='Output directory for visualizations')
    parser.add_argument('--export_excel', action='store_true',
                        help='Also export data to Excel')

    args = parser.parse_args()

    # 创建可视化器
    visualizer = AgentVisualizer(args.json_file, args.output_dir)

    # 生成所有可视化
    visualizer.generate_all_plots()

    # 可选：导出Excel
    if args.export_excel:
        visualizer.export_to_excel()

    print("\n" + "=" * 60)
    print("Visualization completed successfully!")
    print(f"All files saved to: {visualizer.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    # 示例：直接使用
    visualizer = AgentVisualizer('runs/vision/hash_state3/metrics/fragment_0_complete.json','runs/vision/hash_state3/'+'visualization_output')

    visualizer.generate_all_plots()
    visualizer.export_to_excel()

    # # 或者使用命令行
    # main()