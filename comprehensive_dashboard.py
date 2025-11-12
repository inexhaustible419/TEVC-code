# -*- coding: utf-8 -*-
# @Time    : 2025/10/16 15:01
# @Author  : JIA
# @FileName: comprehensive_dashboard.py.py
# @Software: PyCharm
# @Blog    ：
# comprehensive_dashboard.py
# 功能：生成一个包含所有关键指标的综合仪表盘，用于全面分析单次训练过程。

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# --- 初始化设置 ---
warnings.filterwarnings('ignore')
# 设置中文字体和风格，确保图表中的中文能正确显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class ComprehensiveDashboard:
    """
    一个综合的可视化分析器，用于将所有关键训练指标汇总到一张大图上。
    """

    def __init__(self, json_path, output_dir="visualization_output"):
        """
        初始化可视化器

        Args:
            json_path: MetricsLogger生成的JSON数据文件路径。
            output_dir: 可视化结果的输出目录。
        """
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"错误：找不到指定的JSON文件 -> {json_path}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print(f"正在从 {self.json_path} 加载数据...")
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

        # 提取关键数据
        self.metadata = self.data.get('metadata', {})
        self.steps_df = pd.DataFrame(self.data.get('steps', []))

        if self.steps_df.empty:
            raise ValueError("错误：JSON文件中没有找到 'steps' 数据，无法生成图表。")

        self.fragment_id = self.metadata.get('fragment_id', 'unknown')
        self.iteration_id = self.metadata.get('iteration_id', 'unknown')

        print(f"✓ 数据加载成功: {len(self.steps_df)} steps，来自 Fragment {self.fragment_id}")

    def generate_dashboard(self):
        """
        生成并保存包含13个关键子图的综合仪表盘。
        """
        print("正在生成综合仪表盘...")

        # 创建一个 4x4 的网格布局，尺寸足够大以容纳所有细节
        fig, axes = plt.subplots(4, 4, figsize=(32, 24))
        fig.suptitle(f'Comprehensive Training Dashboard - Fragment {self.fragment_id} Iteration {self.iteration_id}',
                     fontsize=24, fontweight='bold')

        # --- 1. 训练过程总览 (第1行) ---
        self._plot_objective_evolution(axes[0, 0])
        self._plot_improvement_rate(axes[0, 1])
        self._plot_loss_and_q_value(axes[0, 2])
        self._plot_exploration_strategy(axes[0, 3])

        # --- 2. 算子分析 (第2行) ---
        self._plot_destroy_freq_evolution(axes[1, 0])
        self._plot_repair_freq_evolution(axes[1, 1])
        self._plot_destroy_value_evolution(axes[1, 2])
        self._plot_repair_value_evolution(axes[1, 3])

        # --- 3. Q值与冲突分析 (第3行) ---
        self._plot_q_value_distribution(axes[2, 0])
        self._plot_q_value_std(axes[2, 1])
        self._plot_conflicts_vs_objective(axes[2, 2])
        self._plot_total_reward_evolution(axes[2, 3])  # 总奖励图放在这里

        # --- 4. 状态与奖励分量 (第4行，使用跨列子图) ---
        # 清理第4行的预置子图
        gs = axes[3, 0].get_gridspec()
        for ax in axes[3, :]:
            ax.remove()

        # 创建跨列的新子图
        ax_state = fig.add_subplot(gs[3, 0:2])
        ax_reward_comp = fig.add_subplot(gs[3, 2:4])

        self._plot_state_evolution(ax_state)
        self._plot_reward_components(ax_reward_comp)

        # --- 最终调整和保存 ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # 使用您指定的命名格式
        save_name = f'Fragment_{self.fragment_id}_Comprehensive_Dashboard.png'
        save_path = self.output_dir / save_name

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 综合仪表盘已成功保存至: {save_path}")

    # --- 以下是各个子图的绘制函数 ---

    def _plot_objective_evolution(self, ax):
        ax.plot(self.steps_df['step'], self.steps_df['current_obj'], label='Current Obj', alpha=0.6, linewidth=1.5)
        ax.plot(self.steps_df['step'], self.steps_df['best_obj'], label='Best Obj', linewidth=2.5, color='red')
        ax.set_title('1. Objective Value Evolution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Link Time')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_improvement_rate(self, ax):
        window = max(1, min(50, len(self.steps_df) // 10))
        improvement_rate = self.steps_df['improvement'].rolling(window).mean()
        ax.plot(self.steps_df['step'], improvement_rate, linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'2. Improvement Rate (Window={window})', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg Improvement')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_loss_and_q_value(self, ax):
        ax2 = ax.twinx()
        loss_series = self.steps_df['loss'].replace(0, np.nan)
        if not loss_series.isnull().all():
            ax.plot(self.steps_df['step'], loss_series, color='blue', alpha=0.7, label='Loss')
        ax.set_ylabel('Loss', color='blue')
        ax2.plot(self.steps_df['step'], self.steps_df['q_avg'], color='red', alpha=0.7, label='Avg Q-Value')
        ax2.set_ylabel('Q-Value', color='red')
        ax.set_title('3. Training Loss & Q-Value', fontweight='bold')
        ax.set_xlabel('Step')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_exploration_strategy(self, ax):
        ax2 = ax.twinx()
        ax.plot(self.steps_df['step'], self.steps_df['epsilon'], color='purple', linewidth=2, label='Epsilon')
        ax.set_ylabel('Epsilon', color='purple')
        window = max(1, min(50, len(self.steps_df) // 10))
        acceptance = self.steps_df['is_accepted'].rolling(window).mean()
        ax2.plot(self.steps_df['step'], acceptance, color='green', linewidth=2, label='Acceptance Rate')
        ax2.set_ylabel('Acceptance Rate', color='green')
        ax.set_title('4. Exploration Strategy', fontweight='bold')
        ax.set_xlabel('Step')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_destroy_freq_evolution(self, ax):
        for d_id in range(8):
            freq_series = self.steps_df['destroy_selection_freq'].apply(
                lambda x: x[d_id] if isinstance(x, (list, np.ndarray)) and len(x) > d_id else 0)
            ax.plot(self.steps_df['step'], freq_series, label=f'D{d_id}', linewidth=1.5, alpha=0.9)
        ax.set_title('5. Destroy Operator Frequency', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Frequency')
        ax.legend(ncol=4, fontsize=9)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_repair_freq_evolution(self, ax):
        for r_id in range(8):
            freq_series = self.steps_df['repair_selection_freq'].apply(
                lambda x: x[r_id] if isinstance(x, (list, np.ndarray)) and len(x) > r_id else 0)
            ax.plot(self.steps_df['step'], freq_series, label=f'R{r_id}', linewidth=1.5, alpha=0.9)
        ax.set_title('6. Repair Operator Frequency', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Frequency')
        ax.legend(ncol=4, fontsize=9)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_destroy_value_evolution(self, ax):
        for i in range(8):
            values = [step['destroy_values'][i] for step in self.data['steps'] if step.get('destroy_values')]
            ax.plot(values, label=f'D{i}', alpha=0.8)
        ax.set_title('7. Destroy Operator Value', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend(ncol=4, fontsize=9)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_repair_value_evolution(self, ax):
        for i in range(8):
            values = [step['repair_values'][i] for step in self.data['steps'] if step.get('repair_values')]
            ax.plot(values, label=f'R{i}', alpha=0.8)
        ax.set_title('8. Repair Operator Value', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend(ncol=4, fontsize=9)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_q_value_distribution(self, ax):
        ax.plot(self.steps_df['step'], self.steps_df['q_avg'], 'r-', linewidth=2, label='Avg Q')
        ax.fill_between(self.steps_df['step'], self.steps_df['q_min'], self.steps_df['q_max'], color='red', alpha=0.2,
                        label='Min/Max Range')
        ax.set_title('9. Q-Value Distribution', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Q-Value')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_q_value_std(self, ax):
        ax.plot(self.steps_df['step'], self.steps_df['q_std'], linewidth=2)
        ax.set_title('10. Q-Value Std (Learning Stability)', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Q-Value Std')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_conflicts_vs_objective(self, ax):
        ax2 = ax.twinx()
        ax.plot(self.steps_df['step'], self.steps_df['internal_conflicts'], color='red', alpha=0.7, label='Conflicts')
        ax.set_ylabel('Internal Conflicts', color='red')
        ax2.plot(self.steps_df['step'], self.steps_df['best_obj'], color='blue', alpha=0.7, label='Best Obj')
        ax2.set_ylabel('Best Objective', color='blue')
        ax.set_title('11. Conflicts vs Objective', fontweight='bold')
        ax.set_xlabel('Step')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_total_reward_evolution(self, ax):
        window = max(1, min(50, len(self.steps_df) // 10))
        reward_smooth = self.steps_df['reward'].rolling(window).mean()
        ax.plot(self.steps_df['step'], reward_smooth, linewidth=2, label=f'Total Reward (smooth, w={window})',
                color='green')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('12. Total Reward Evolution', fontweight='bold')
        ax.set_ylabel('Reward')
        ax.set_xlabel('Step')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def _plot_state_evolution(self, ax):
        if 'state' not in self.steps_df.columns or self.steps_df['state'].isnull().all():
            ax.text(0.5, 0.5, 'No State Data Found', ha='center', va='center', fontsize=12)
            return

        num_features = len(self.steps_df['state'].iloc[0])
        state_df = pd.DataFrame(self.steps_df['state'].tolist(), columns=[f'S{i}' for i in range(num_features)])
        normalized_state_df = (state_df - state_df.mean()) / (state_df.std() + 1e-8)

        state_labels = {
            0: "LinkTime", 1: "Density", 5: "ConflictRatio", 10: "BoundaryDensity",
            11: "Stagnation", 12: "ImproveRate", 13: "LeftConflict", 14: "RightConflict"
        }

        for i in range(num_features):
            label = f'S{i}'
            if i in state_labels:
                label = f'S{i}-{state_labels[i]}'
            ax.plot(self.steps_df['step'], normalized_state_df[f'S{i}'], label=label, alpha=0.8, linewidth=1.5)

        ax.set_title(f'13. State Feature Evolution ({num_features}-dim, Normalized)', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Normalized Value (Z-score)')
        ax.legend(ncol=4, fontsize=9, loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(-5, 5)

    def _plot_reward_components(self, ax):
        window = max(1, min(50, len(self.steps_df) // 10))
        reward_components = ['reward_link_time', 'reward_improvement', 'reward_structure', 'reward_boundary']
        labels = ['Link Time', 'Improvement', 'Structure', 'Boundary']

        for comp, label in zip(reward_components, labels):
            if comp in self.steps_df.columns:
                comp_smooth = self.steps_df[comp].rolling(window).mean()
                ax.plot(self.steps_df['step'], comp_smooth, linewidth=2, label=label)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'14. Reward Composition (Smooth, w={window})', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward Value')
        ax.legend(loc='lower left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)


if __name__ == '__main__':
    # --- 使用方法 ---
    # 1. 将此文件 (comprehensive_dashboard.py) 放置在您的项目根目录下。
    # 2. 修改下面的JSON文件路径为您需要分析的实际文件路径。
    # 3. 运行此脚本: python comprehensive_dashboard.py

    # 使用您在问题中提供的新路径格式
    json_file_path = 'runs/vision/rep_8/metrics/fragment_0_complete.json'
    output_directory = 'runs/vision/rep_8/visualization_output'

    # 创建并运行Dashboard生成器
    dashboard_generator = ComprehensiveDashboard(json_path=json_file_path, output_dir=output_directory)
    dashboard_generator.generate_dashboard()