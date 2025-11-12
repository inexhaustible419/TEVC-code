# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 19:45
# @Author  : JIA
# @FileName: Config.py
# @Software: PyCharm
# @Blog    ：
import json
import yaml
import os
import shutil


def load_config(config_file):
    """根据文件类型加载配置"""
    if config_file.endswith('.json'):
        with open(config_file, 'r') as f:
            return json.load(f)
    elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .yaml file.")


class Config:
    def __init__(self, config_file, file_type='json'):
        self.config_file = config_file
        self.file_type = file_type
        self.hyperparameters = self.load_hyperparameters()

    def load_hyperparameters(self):
        """
        读取配置文件并返回超参数
        """
        if self.file_type == 'json':
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif self.file_type == 'yaml':
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file type. Use 'json' or 'yaml'.")

    def get(self, key):
        """
        获取配置项的值
        """
        return self.hyperparameters.get(key)

    def save_config_to_log(self, config_file,log_dir):
        """将 YAML 配置文件拷贝到 log_dir"""
        if config_file:  # 如果配置文件存在
            filename = os.path.basename(config_file)
            dest = os.path.join(log_dir, filename)
            shutil.copy(config_file, dest)  # 拷贝文件
            print(f"Configuration file copied to {dest}")