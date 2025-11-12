#!/usr/bin/env python3
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
        # Use only key parameters for the file name to keep it manageable
        # Include all parameters being varied in the experiment
        # key_params = {k: v for k, v in param_dict.items() if k in experiment_params}
        # param_str = "_".join([f"{k}_{v}" for k, v in key_params.items()])
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # config_name = f"config_{i}_{param_str}_{timestamp}.yaml"
        # Use only main parameters for the file name to keep it manageable
        main_params = ['set_time', 'data', 'fragments','weight2']
        key_params = {k: v for k, v in param_dict.items() if k in main_params}
        param_str = "_".join([f"{k}_{v}" for k, v in key_params.items()])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = f"config_{i:02d}_{param_str}_{timestamp}.yaml"

        config_path = os.path.join(output_dir, config_name)
        
        # Set log directory based on parameters
        log_dir = f"runs/{timestamp}_{'_'.join([f'{k}_{v}' for k, v in key_params.items()])}"
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
        
        print(f"Running experiment with config {os.path.basename(config_path)}, repetition {rep+1}/{repetitions}")
        
        # Run the main script
        start_time = time.time()
        
        try:
            # 【修改】运行fixed_main2.py而不是fixed-main.py
            subprocess.run(['python', 'fixed_main2.py'], check=True)
            
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

def generate_summary_report(all_results, output_dir):
    """
    Generate summary report of all experiments
    
    Args:
        all_results: List of dictionaries containing experiment results
        output_dir: Directory to save the report
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_results)
    
    # Create summary report directory
    report_dir = os.path.join(output_dir, 'summary_report')
    os.makedirs(report_dir, exist_ok=True)
    
    # Save raw results as CSV
    df.to_csv(os.path.join(report_dir, 'all_results.csv'), index=False)
    
    # Filter out rows with None final_link_time before calculating statistics
    df_valid = df.dropna(subset=['final_link_time'])
    
    if len(df_valid) > 0:
        # Create summary by configuration
        summary_columns = ['final_link_time', 'runtime']
        if 'before_conflict_value' in df_valid.columns and not df_valid['before_conflict_value'].isna().all():
            summary_columns.insert(0, 'before_conflict_value')
            
        summary = df_valid.groupby('config').agg({
            col: ['mean', 'std', 'min', 'max'] for col in summary_columns
        }).reset_index()
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary.rename(columns={'config_': 'config'}, inplace=True)
        
        # Save summary as CSV
        summary.to_csv(os.path.join(report_dir, 'summary.csv'), index=False)
        
        # Extract configuration parameters from config names
        config_params = {}
        for config in df_valid['config'].unique():
            # Extract parameters from config name
            parts = config.split('_')
            i = 0
            while i < len(parts):
                # 【修改】添加新的VAE参数到解析列表
                if parts[i] in ['weight1', 'weight2', 'weight3', 'set_time', 'epochs', 'q', 
                                'learning_rate', 'epsilon_decay', 'batch_size', 'gamma', 'phi',
                                'it_num', 'vae_epochs', 'vae_learning_rate', 'vae_beta', 
                                'vae_batch_size', 'fragments']:
                    # The next part should be the value
                    if i+1 < len(parts):
                        if config not in config_params:
                            config_params[config] = {}
                        try:
                            config_params[config][parts[i]] = float(parts[i+1])
                        except:
                            pass
                i += 1
        
        # Add parameter columns to dataframe
        # 【修改】添加新的VAE参数列
        for param in ['weight1', 'weight2', 'weight3', 'set_time', 'epochs', 'q', 
                      'learning_rate', 'epsilon_decay', 'batch_size', 'gamma', 'phi',
                      'it_num', 'vae_epochs', 'vae_learning_rate', 'vae_beta', 
                      'vae_batch_size', 'fragments']:
            df_valid[param] = df_valid['config'].map(lambda x: config_params.get(x, {}).get(param, np.nan))
        
        # Generate plots for parameter analysis
        if len(df_valid) > 0:
            # Plot by set_time for both before and after conflict resolution
            if len(df_valid['set_time'].dropna().unique()) > 1:
                # Plot for final_link_time (post-conflict)
                plt.figure(figsize=(10, 6))
                for group, data in df_valid.groupby(['weight1', 'weight2', 'weight3']):
                    if pd.notna(data['final_link_time']).any():
                        plt.errorbar(
                            data['set_time'],
                            data.groupby('set_time')['final_link_time'].mean(),
                            yerr=data.groupby('set_time')['final_link_time'].std(),
                            label=f"w1={group[0]}, w2={group[1]}, w3={group[2]}"
                        )
                plt.xlabel('Set Time (s)')
                plt.ylabel('Average Post-Conflict Link Time')
                plt.title('Post-Conflict Performance by Set Time and Weights')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, 'post_conflict_performance_by_set_time.png'))
                plt.close()
                
                # Plot for before_conflict_value if available
                if 'before_conflict_value' in df_valid.columns and not df_valid['before_conflict_value'].isna().all():
                    plt.figure(figsize=(10, 6))
                    for group, data in df_valid.groupby(['weight1', 'weight2', 'weight3']):
                        if pd.notna(data['before_conflict_value']).any():
                            plt.errorbar(
                                data['set_time'],
                                data.groupby('set_time')['before_conflict_value'].mean(),
                                yerr=data.groupby('set_time')['before_conflict_value'].std(),
                                label=f"w1={group[0]}, w2={group[1]}, w3={group[2]}"
                            )
                    plt.xlabel('Set Time (s)')
                    plt.ylabel('Average Pre-Conflict Link Time')
                    plt.title('Pre-Conflict Performance by Set Time and Weights')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(report_dir, 'pre_conflict_performance_by_set_time.png'))
                    plt.close()
                    
                    # Plot for conflict resolution impact (difference)
                    plt.figure(figsize=(10, 6))
                    for group, data in df_valid.groupby(['weight1', 'weight2', 'weight3']):
                        if pd.notna(data['before_conflict_value']).any() and pd.notna(data['final_link_time']).any():
                            # Calculate average difference by set_time
                            diff_data = data.groupby('set_time').apply(
                                lambda x: (x['before_conflict_value'] - x['final_link_time']).mean()
                            )
                            std_diff = data.groupby('set_time').apply(
                                lambda x: (x['before_conflict_value'] - x['final_link_time']).std()
                            )
                            plt.errorbar(
                                diff_data.index,
                                diff_data.values,
                                yerr=std_diff.values,
                                label=f"w1={group[0]}, w2={group[1]}, w3={group[2]}"
                            )
                    plt.xlabel('Set Time (s)')
                    plt.ylabel('Average Link Time Reduction Due to Conflict Resolution')
                    plt.title('Impact of Conflict Resolution by Set Time and Weights')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(report_dir, 'conflict_resolution_impact_by_set_time.png'))
                    plt.close()
            
            # 【新增】VAE参数性能分析图
            # Plot by VAE epochs
            if len(df_valid['vae_epochs'].dropna().unique()) > 1:
                plt.figure(figsize=(10, 6))
                for group, data in df_valid.groupby(['vae_learning_rate', 'vae_beta']):
                    if pd.notna(data['final_link_time']).any():
                        plt.errorbar(
                            data['vae_epochs'],
                            data.groupby('vae_epochs')['final_link_time'].mean(),
                            yerr=data.groupby('vae_epochs')['final_link_time'].std(),
                            label=f"lr={group[0]}, beta={group[1]}"
                        )
                plt.xlabel('VAE Epochs')
                plt.ylabel('Average Final Link Time')
                plt.title('Performance by VAE Epochs and Parameters')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, 'performance_by_vae_epochs.png'))
                plt.close()
            
            # Plot by it_num (continuous training iterations)
            if len(df_valid['it_num'].dropna().unique()) > 1:
                plt.figure(figsize=(10, 6))
                for group, data in df_valid.groupby(['epochs', 'q']):
                    if pd.notna(data['final_link_time']).any():
                        plt.errorbar(
                            data['it_num'],
                            data.groupby('it_num')['final_link_time'].mean(),
                            yerr=data.groupby('it_num')['final_link_time'].std(),
                            label=f"epochs={group[0]}, q={group[1]}"
                        )
                plt.xlabel('Continuous Training Iterations (it_num)')
                plt.ylabel('Average Final Link Time')
                plt.title('Performance by Continuous Training Iterations')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, 'performance_by_it_num.png'))
                plt.close()
            
            # Plot by weights for each set_time
            for set_time in df_valid['set_time'].dropna().unique():
                subset = df_valid[df_valid['set_time'] == set_time]
                
                # If we have different weight combinations
                weight_combinations = subset[['weight1', 'weight2', 'weight3']].drop_duplicates()
                if len(weight_combinations) > 1:
                    plt.figure(figsize=(12, 8))
                    
                    # Create labels for x-axis
                    labels = [f"w1={row['weight1']}, w2={row['weight2']}, w3={row['weight3']}" 
                            for _, row in weight_combinations.iterrows()]
                    
                    # Get average and std for each weight combination
                    averages = []
                    errors = []
                    
                    for _, weights in weight_combinations.iterrows():
                        mask = ((subset['weight1'] == weights['weight1']) & 
                                (subset['weight2'] == weights['weight2']) & 
                                (subset['weight3'] == weights['weight3']))
                        
                        data = subset[mask]['final_link_time']
                        if len(data) > 0 and pd.notna(data).any():
                            averages.append(data.mean())
                            errors.append(data.std())
                        else:
                            averages.append(0)
                            errors.append(0)
                    
                    # Create bar chart
                    plt.bar(range(len(labels)), averages, yerr=errors)
                    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
                    plt.xlabel('Weight Combinations')
                    plt.ylabel('Average Final Link Time')
                    plt.title(f'Performance by Weight Combinations (Set Time = {set_time}s)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(report_dir, f'performance_by_weights_time_{int(set_time)}.png'))
                    plt.close()
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fixed Main2 Experiment Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            h1, h2 {{ color: #333; }}
            .container {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Fixed Main2 Experiment Summary Report</h1>
        <p>This report covers experiments using the improved continuous training with VAE (fixed_main2.py)</p>
    """
    
    # Add summary statistics table if we have valid results
    if len(df_valid) > 0:
        html_content += """
        <div class="container">
            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>Configuration</th>
        """
        
        # Add headers for before_conflict_value if it exists
        if 'before_conflict_value_mean' in summary.columns:
            html_content += """
                    <th>Mean Pre-Conflict Value</th>
                    <th>Std Pre-Conflict Value</th>
                    <th>Min Pre-Conflict Value</th>
                    <th>Max Pre-Conflict Value</th>
            """
            
        html_content += """
                    <th>Mean Post-Conflict Value</th>
                    <th>Std Post-Conflict Value</th>
                    <th>Min Post-Conflict Value</th>
                    <th>Max Post-Conflict Value</th>
                    <th>Mean Runtime (s)</th>
                </tr>
        """
        
        for _, row in summary.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['config']}</td>
            """
            
            # Add before_conflict_value data if it exists
            if 'before_conflict_value_mean' in row:
                html_content += f"""
                    <td>{row['before_conflict_value_mean']:.2f}</td>
                    <td>{row['before_conflict_value_std']:.2f}</td>
                    <td>{row['before_conflict_value_min']:.2f}</td>
                    <td>{row['before_conflict_value_max']:.2f}</td>
                """
                
            html_content += f"""
                    <td>{row['final_link_time_mean']:.2f}</td>
                    <td>{row['final_link_time_std']:.2f}</td>
                    <td>{row['final_link_time_min']:.2f}</td>
                    <td>{row['final_link_time_max']:.2f}</td>
                    <td>{row['runtime_mean']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    else:
        html_content += """
        <div class="container">
            <h2>No Valid Results Found</h2>
            <p>No valid results with final link time measurements were found. Check for errors in the experiment execution or output parsing.</p>
        </div>
        """
    
    # Add error table if we have any errors
    if 'error_message' in df.columns and not df['error_message'].isna().all():
        df_errors = df[df['error_message'].notna()]
        if len(df_errors) > 0:
            html_content += """
            <div class="container">
                <h2>Errors</h2>
                <table>
                    <tr>
                        <th>Configuration</th>
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
            
            html_content += """
                </table>
            </div>
            """
    
    html_content += """
        <div class="container">
            <h2>Performance Visualizations</h2>
    """
    
    # Add images to HTML
    image_files = [f for f in os.listdir(report_dir) if f.endswith('.png')]
    for img_file in image_files:
        html_content += f"""
            <div>
                <h3>{img_file.replace('.png', '').replace('_', ' ').title()}</h3>
                <img src="{img_file}" alt="{img_file}">
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(report_dir, 'report.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Summary report generated at {os.path.join(report_dir, 'report.html')}")

def main():
    parser = argparse.ArgumentParser(description='Run parameter sweep experiments for fixed_main2')
    parser.add_argument('--base-config', type=str, default='args.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--output-dir', type=str, default='experiment_results_main2',
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
            'set_time': [1200, 1800, 2400],  # 20min, 30min, 40min (适应连续训练)
            'it_num': [1, 2, 3],  # 连续训练迭代次数
            'weight1': [0.8, 1.0, 1.2],
            'weight2': [0, 0.3, 0.5],
            'weight3': [0, 0.05, 0.1],
            'vae_epochs': [8, 12, 16],  # VAE训练epoch数
            'vae_learning_rate': [0.001, 0.002, 0.003],  # VAE学习率
            # 可选的其他VAE参数
            # 'vae_beta': [0.2, 0.4, 0.6],  # VAE KL散度权重
            # 'vae_batch_size': [8, 16, 24],
            # 'epochs': [3, 5, 7],  # ALNS epochs
            # 'fragments': [4, 6, 8],  # 片段数量
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
