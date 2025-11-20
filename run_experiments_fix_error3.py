# -*- coding: utf-8 -*-
# @Time    : 2025/11/18 13:44
# @Author  : JIA
# @FileName: run_experiments_fix_error3.py
# @Software: PyCharm
# @Blog    ：
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import yaml
import time
import shutil
import itertools
from datetime import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_experiment_configs(base_config_path, output_dir, experiment_params):
    """
    Create config files for each experiment combination

    Args:
        base_config_path: Path to the base YAML config file
        output_dir: Directory to save generated config files
        experiment_params: Dictionary of parameters to vary across experiments

    Returns:
        List of paths to generated config files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Generate parameter combinations
    param_names = list(experiment_params.keys())
    param_values = list(experiment_params.values())
    combinations = list(itertools.product(*param_values))

    config_paths = []

    # Create a config file for each combination
    for i, combination in enumerate(combinations):
        # Create a copy of the base config
        config = base_config.copy()

        # Update with current parameter combination
        param_dict = {param_names[j]: combination[j] for j in range(len(param_names))}

        # Apply parameters to config
        for key, value in param_dict.items():
            config[key] = value

        # Create descriptive name for this configuration
        # Use all parameters being varied in the experiment for the name
        main_params = ['set_time', 'data', 'fragments', 'weight2']
        key_params = {k: v for k, v in param_dict.items() if k in main_params}
        param_str = "_".join([f"{k}_{v}" for k, v in key_params.items()])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = f"config_{i:02d}_{param_str}_{timestamp}.yaml"

        config_path = os.path.join(output_dir, config_name)

        # Set log directory based on parameters
        log_dir = f"runs/{timestamp}_{param_str}"
        config['log_dir'] = log_dir

        # Save config to file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        config_paths.append(config_path)

    return config_paths


def run_experiment(config_path, repetitions=3):
    """
    Run a single experiment with the specified config

    Args:
        config_path: Path to the YAML config file
        repetitions: Number of times to repeat the experiment

    Returns:
        List of result metrics for each repetition
    """
    # Load config to get log_dir
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    log_dir = config['log_dir']
    results = []

    # Create temporary args.yaml that points to this config
    shutil.copy(config_path, 'args.yaml')

    for rep in range(repetitions):
        # Create repetition-specific log directory
        rep_log_dir = f"{log_dir}/rep_{rep}"

        # Update config with repetition-specific log_dir
        config['log_dir'] = rep_log_dir
        with open('args.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Create log directory
        os.makedirs(rep_log_dir, exist_ok=True)

        print(f"Running experiment with config {os.path.basename(config_path)}, repetition {rep + 1}/{repetitions}")

        # Run the main script
        start_time = time.time()

        try:
            # 【修改】运行fixed_main2.py而不是fixed-main.py
            # 使用 fixed_main2_addoffline_log.py
            subprocess.run(['python', 'fixed_main2_addoffline_log.py'], check=True)

            # Extract results from output file
            # 【修改】适配新的输出文件名（来自fixed_main2.py）
            output_file = os.path.join(rep_log_dir, 'improved_corrected_output.txt')
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read()
                    # Extract the needed metrics (adapt according to your output format)
                    # Improved parsing logic to handle your specific output format
                    try:
                        # Look for objective values before and after conflict resolution
                        lines = [line.strip() for line in content.split('\n') if line.strip()]
                        before_conflict_value = None
                        after_conflict_value = None

                        for line in lines:
                            # Extract pre-conflict objective value
                            if "冲突消解前目标值:" in line:
                                before_conflict_value = float(line.split(":")[-1].strip())

                            # Extract post-conflict objective value
                            if "冲突消解后目标值:" in line:
                                after_conflict_value = float(line.split(":")[-1].strip())

                            # 【新增】处理fixed_main2.py的新输出格式
                            if "Final solution:" in line and "total link_time:" in line:
                                # Extract from "Final solution: X arcs, total link_time: Y"
                                parts = line.split("total link_time:")
                                if len(parts) > 1:
                                    try:
                                        after_conflict_value = float(parts[1].strip())
                                    except ValueError:
                                        pass

                        # If no specific markers found, try to find a standalone number for compatibility
                        if after_conflict_value is None:
                            for line in lines:
                                if line.isdigit() or (line and line[0].isdigit()):
                                    try:
                                        after_conflict_value = float(line)
                                        break
                                    except ValueError:
                                        continue

                        # Use the after_conflict_value as final_link_time
                        final_link_time = after_conflict_value

                        result = {
                            'config': os.path.basename(config_path),
                            'repetition': rep,
                            'final_link_time': final_link_time,
                            'runtime': time.time() - start_time,
                            'log_dir': rep_log_dir
                        }

                        # Only add before_conflict_value if it was found
                        if before_conflict_value is not None:
                            result['before_conflict_value'] = before_conflict_value

                        results.append(result)
                    except Exception as e:
                        print(f"Error parsing results: {e}")
                        results.append({
                            'config': os.path.basename(config_path),
                            'repetition': rep,
                            'final_link_time': None,
                            'runtime': time.time() - start_time,
                            'log_dir': rep_log_dir,
                            'error_message': str(e)
                        })
            else:
                print(f"Warning: Output file not found at {output_file}")
                results.append({
                    'config': os.path.basename(config_path),
                    'repetition': rep,
                    'final_link_time': None,
                    'runtime': time.time() - start_time,
                    'log_dir': rep_log_dir,
                    'error_message': 'Output file not found'
                })

        except Exception as e:
            print(f"Error running experiment: {e}")
            results.append({
                'config': os.path.basename(config_path),
                'repetition': rep,
                'final_link_time': None,
                'runtime': time.time() - start_time,
                'log_dir': rep_log_dir,
                'error_message': str(e)
            })

    # Restore config log_dir
    config['log_dir'] = log_dir
    with open('args.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return results


# ==============================================================================
# ===
# ===               [函数已修改] generate_summary_report
# ===
# ==============================================================================

def generate_summary_report(all_results, output_dir):
    """
    Generate summary report of all experiments in the format requested.

    Args:
        all_results: List of dictionaries containing experiment results
        output_dir: Directory to save the report
    """
    if not all_results:
        print("No results to generate report.")
        return

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_results)

    # Create summary report directory
    report_dir = os.path.join(output_dir, 'summary_report')
    os.makedirs(report_dir, exist_ok=True)

    # --- 1. 提取超参数 ---

    # 定义所有可能在文件名中查找的参数
    # (您在 create_experiment_configs 中使用的所有 key 都应在此)
    param_keys = [
        'set_time', 'data', 'fragments', 'weight1', 'weight2', 'weight3',
        'iter_num', 'batch_size', 'it_num', 'epochs', 'q',
        'learning_rate', 'epsilon_decay', 'gamma', 'phi',
        'vae_epochs', 'vae_learning_rate', 'vae_beta', 'vae_batch_size'
    ]

    config_params = {}
    for config_name in df['config'].unique():
        config_params[config_name] = {}
        parts = config_name.split('_')
        i = 0
        while i < len(parts):
            if parts[i] in param_keys:
                if i + 1 < len(parts):
                    try:
                        # 尝试转换为浮点数
                        config_params[config_name][parts[i]] = float(parts[i + 1])
                    except ValueError:
                        # 如果失败，作为字符串
                        config_params[config_name][parts[i]] = parts[i + 1]
            i += 1

    # --- 2. 将超参数映射到DataFrame的新列 ---

    # 定义您关心的列
    param_cols_to_add = ['set_time', 'fragments', 'iter_num', 'it_num']

    for param in param_cols_to_add:
        df[param] = df['config'].map(lambda x: config_params.get(x, {}).get(param, np.nan))

    # --- 3. 计算 "Conflict Loss %" ---
    df['Conflict Loss %'] = np.where(
        (df['before_conflict_value'].notna()) & (df['before_conflict_value'] > 0),
        (df['before_conflict_value'] - df['final_link_time']) / df['before_conflict_value'] * 100,
        0.0
    )

    # 确保 repetition 是整数
    df['repetition'] = df['repetition'].astype(int)

    # --- 4. 保存详细的扁平化 CSV ---
    # (这对于后续分析很有用)
    flat_cols = param_cols_to_add + [
        'repetition', 'runtime', 'before_conflict_value',
        'final_link_time', 'Conflict Loss %', 'log_dir', 'config'
    ]
    # 确保所有列都存在
    for col in flat_cols:
        if col not in df.columns:
            df[col] = np.nan

    df_flat = df[flat_cols]
    df_flat.to_csv(os.path.join(report_dir, 'detailed_results.csv'), index=False, float_format='%.2f')

    # --- 5. 创建并保存数据透视表 (Pivoted Table) ---
    try:
        index_cols = ['set_time', 'fragments', 'iter_num', 'it_num']
        value_cols = ['runtime', 'before_conflict_value', 'final_link_time', 'Conflict Loss %']

        # 移除索引列中的 NaN，否则 pivot 会失败
        df_pivot_source = df.dropna(subset=index_cols)

        if not df_pivot_source.empty:
            df_pivot = df_pivot_source.pivot_table(
                index=index_cols,
                columns='repetition',
                values=value_cols
            )

            # 重新排序列，使指标按 Run 分组
            max_reps = df_pivot_source['repetition'].max() + 1
            new_cols = []
            for i in range(max_reps):
                for val_col in value_cols:
                    if (val_col, i) in df_pivot.columns:
                        new_cols.append((val_col, i))

            df_pivot = df_pivot.reindex(columns=new_cols)

            # 调整列名以便阅读
            df_pivot.columns = pd.MultiIndex.from_tuples([
                (f"Run {rep + 1}", metric.replace('_', ' ').title())
                for (metric, rep) in new_cols
            ])

            # 保存数据透视表CSV
            df_pivot.to_csv(os.path.join(report_dir, 'summary_pivot_table.csv'), float_format='%.2f')

        else:
            print("No valid data to pivot.")
            df_pivot = pd.DataFrame()  # 创建一个空的
            max_reps = 0

    except Exception as e:
        print(f"Error creating pivot table: {e}")
        df_pivot = pd.DataFrame()  # 创建一个空的
        max_reps = df.get('repetition', pd.Series([0])).max() + 1

    # --- 6. 生成新的 HTML 报告 (基于数据透视表) ---

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fixed Main2 实验汇总报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: auto; min-width: 80%; border: 1px solid #aaa; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            h1 {{ color: #333; }}
            .container {{ margin-bottom: 30px; overflow-x: auto; }}
            .header-params {{ min-width: 100px; }}
            .header-run {{ background-color: #e0e0e0; text-align: center; }}
            .header-metric {{ background-color: #f0f0f0; min-width: 110px; }}
            .cell-na {{ color: #999; }}
        </style>
    </head>
    <body>
        <h1>Fixed Main2 实验汇总报告</h1>
    """

    # --- 错误表格 ---
    if 'error_message' in df.columns and not df['error_message'].isna().all():
        df_errors = df[df['error_message'].notna()]
        if not df_errors.empty:
            html_content += """
            <div class="container">
                <h2>实验运行错误</h2>
                <table>
                    <tr>
                        <th>Config</th>
                        <th>Repetition</th>
                        <th>Error</th>
                    </tr>
            """
            for _, row in df_errors.iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['config']}</td>
                        <td>{row['repetition']}</td>
                        <td>{row['error_message']}</td>
                    </tr>
                """
            html_content += "</table></div>"

    # --- 汇总结果表格 ---
    html_content += """
        <div class="container">
            <h2>实验结果汇总</h2>
            <table>
    """

    # 表头 - 第1行 (参数 + Run N)
    html_content += "<thead><tr>"
    html_content += '<th rowspan="2" class="header-params">时间设定 (set_time)</th>'
    html_content += '<th rowspan="2" class="header-params">分组设定 (fragments)</th>'
    html_content += '<th rowspan="2" class="header-params">外部迭代 (iter_num)</th>'
    html_content += '<th rowspan="2" class="header-params">内部挖掘 (it_num)</th>'

    for i in range(max_reps):
        html_content += f'<th colspan="4" class="header-run">Run {i + 1}</th>'

    html_content += "</tr>"

    # 表头 - 第2行 (指标)
    html_content += "<tr>"
    for i in range(max_reps):
        html_content += '<th class="header-metric">Runtime (s)</th>'
        html_content += '<th class="header-metric">Before Conflict</th>'
        html_content += '<th class="header-metric">Final Link Time</th>'
        html_content += '<th class="header-metric">Conflict Loss %</th>'
    html_content += "</tr></thead>"

    # 表格内容
    html_content += "<tbody>"

    for index, row in df_pivot.iterrows():
        html_content += "<tr>"
        # 索引列 (set_time, fragments, etc.)
        for item in index:
            html_content += f"<td>{item}</td>"

        # 数据列
        for i in range(max_reps):
            # (metric_name, rep_index)
            rt = row.get((f"Run {i + 1}", "Runtime"))
            bc = row.get((f"Run {i + 1}", "Before Conflict Value"))
            fl = row.get((f"Run {i + 1}", "Final Link Time"))
            cl = row.get((f"Run {i + 1}", "Conflict Loss %"))

            html_content += f'<td>{f"{rt:.2f}" if pd.notna(rt) else "<span class=cell-na>N/A</span>"}</td>'
            html_content += f'<td>{f"{bc:.2f}" if pd.notna(bc) else "<span class=cell-na>N/A</span>"}</td>'
            html_content += f'<td>{f"{fl:.2f}" if pd.notna(fl) else "<span class=cell-na>N/A</span>"}</td>'
            html_content += f'<td>{f"{cl:.2f}%" if pd.notna(cl) else "<span class=cell-na>N/A</span>"}</td>'

        html_content += "</tr>"

    html_content += "</tbody></table></div>"

    # --- 移除旧的图像部分 ---

    html_content += """
    </body>
    </html>
    """

    with open(os.path.join(report_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Summary report generated at {os.path.join(report_dir, 'report.html')}")


def main():
    parser = argparse.ArgumentParser(description='Run parameter sweep experiments for fixed_main2')
    parser.add_argument('--base-config', type=str, default='args.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                        help='Directory to save results')
    parser.add_argument('--repetitions', type=int, default=3,
                        help='Number of repetitions for each configuration')
    parser.add_argument('--param-file', type=str, default=None,
                        help='Path to a YAML file containing parameter combinations to explore')
    args = parser.parse_args()

    # Define parameter combinations to explore
    if args.param_file:
        # Load parameter combinations from file
        with open(args.param_file, 'r') as f:
            experiment_params = yaml.safe_load(f)
        print(f"Loaded experiment parameters from {args.param_file}")
    else:
        # 【修改】更新默认参数组合以包含新的VAE相关参数
        experiment_params = {
            'set_time': [1200, 1800],  # 20min, 30min
            'it_num': [2, 3],  # 连续训练迭代次数
            'iter_num': [5, 8],  # 外部迭代
            'fragments': [6, 8],  # 片段
            # 'weight1': [0.8, 1.0, 1.2],
            # 'weight2': [0, 0.3, 0.5],
            # 'weight3': [0, 0.05, 0.1],
            # 'vae_epochs': [8, 12, 16],  # VAE训练epoch数
            # 'vae_learning_rate': [0.001, 0.002, 0.003],  # VAE学习率
        }

    # Create timestamp for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"{args.output_dir}_{timestamp}"

    # Create experiment configurations
    config_paths = create_experiment_configs(
        args.base_config,
        os.path.join(experiment_dir, 'configs'),
        experiment_params
    )

    print(f"Created {len(config_paths)} experiment configurations for fixed_main2")

    # Backup original args.yaml if it exists
    if os.path.exists('args.yaml'):
        shutil.copy('args.yaml', 'args.yaml.backup')

    # Run experiments
    all_results = []
    for config_path in config_paths:
        results = run_experiment(config_path, repetitions=args.repetitions)
        all_results.extend(results)

        # Generate intermediate report after each configuration
        generate_summary_report(all_results, experiment_dir)

    # Restore original args.yaml if backup exists
    if os.path.exists('args.yaml.backup'):
        shutil.move('args.yaml.backup', 'args.yaml')

    print(f"All fixed_main2 experiments completed. Results saved to {experiment_dir}")


if __name__ == "__main__":
    main()