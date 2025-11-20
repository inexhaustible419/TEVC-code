import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import deque
import copy
import logging

from Config import Config
config = Config('args.yaml', 'yaml')

import pickle

class ReplayBuffer:
    """
    Simple replay buffer for each agent
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience):
        """Add experience to buffer"""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        if len(self.buffer) < batch_size:
            return None
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)
        
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()

    # --- 新增存盘和加载方法 ---
    def save_buffer(self, path):
        """将buffer保存到文件"""
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
        # print(f"Replay Buffer saved to {path}") # 可选的调试信息

    def load_buffer(self, path):
        """从文件加载buffer"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)
            # print(f"Replay Buffer loaded from {path}, size: {len(self.buffer)}") # 可选的调试信息
        else:
            # print(f"Warning: Replay Buffer file not found at {path}") # 可选的调试信息
            pass

class DuelingDQN(nn.Module):
    """
    Enhanced Dueling DQN network with a more expressive architecture
    to handle the richer state representation.
    """

    def __init__(self, n_actions, state_dim=65):
        super(DuelingDQN, self).__init__()

        # # Shared feature extraction layers
        # self.feature_layer = nn.Sequential(
        #     nn.Linear(state_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        # )
        #Shared feature extraction layers - 需要增强网络容量
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),  # 32 -> 128，增强特征提取能力
            nn.ReLU(),
            nn.Dropout(0.2),  # 新增：防止过拟合
            nn.Linear(128, 64),  # 新增层
            nn.ReLU(),
            nn.Linear(64, 32),  # 保留原有结构
            nn.ReLU(),
        )

        # Value stream - estimates the value of the state
        self.value_stream = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Advantage stream - estimates the advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions)
        )

        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x):
        # x = x[:, :15]
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class ParallelDQNAgent:
    """
    Individual DQN agent - each instance is independent
    """
    def __init__(self, agent_id, n_destroy_actions, n_repair_actions, state_dim,
                 global_model, learning_rate=None,auto_load=True,
                 offline_mode=False,
                 metrics_logger=None,
                 mu=0.0
                 ):
        # 【新增】offline模式开关
        self.offline_mode = offline_mode

        # 【新增】添加记录器
        self.metrics_logger = metrics_logger
        # 缓存最近的Q值和算子价值（用于记录）
        self._last_q_values = None
        self._last_operator_values = None
        self._last_selected_action_id = None

        # --- 新增FedProx相关属性 ---
        self.mu = mu
        self.global_model_weights = None  # 用于存储全局模型权重

        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Action space
        self.n_destroy_actions = n_destroy_actions
        self.n_repair_actions = n_repair_actions
        self.n_actions = n_destroy_actions * n_repair_actions

        # State dimensions
        self.state_dim = state_dim

        # Reference to global model (not used directly, just for initialization)
        self.global_model_ref = global_model
        
        # Create local models - these are separate copies
        self.model = DuelingDQN(self.n_actions, state_dim).to(self.device)
        self.target_model = DuelingDQN(self.n_actions, state_dim).to(self.device)

        # # Initialize with global model weights
        # self.model.load_state_dict(self.global_model_ref.state_dict())
        # self.target_model.load_state_dict(self.model.state_dict())
        
        # Initialize local replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        
        # Parameters
        self.batch_size = config.get('batch_size')
        self.learning_rate = learning_rate if learning_rate else config.get('learning_rate')
        self.gamma = config.get('gamma')
        self.epsilon = config.get('epsilon')
        self.epsilon_decay = config.get('epsilon_decay')
        self.epsilon_min = config.get('epsilon_min')
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.arc_count = 0  # 将在运行时设置
        self.current_solution = None
        self.high_quality_threshold = config.get('high_quality_threshold')  # 高质量解的奖励阈值

        # Metrics tracking
        # self.train_metrics = {
        #     "epsilon": [],
        #     "reward": [],
        #     "fragment_link_time": [],
        #     "loss": [],
        #     "avg_q_value": [],
        #     "vae_updates": []
        # }
        
        # Operator values for guided exploration
        self.operator_values = np.ones(self.n_destroy_actions + self.n_repair_actions) / (self.n_destroy_actions + self.n_repair_actions)
        
        # Training counters
        self.update_counter = 0
        self.target_update_frequency = 10

        self.loss_log=None

        # Logger
        self.logger = logging.getLogger(f'DQNAgent_{agent_id}')

        # 在初始化最后添加自动加载
        if auto_load or offline_mode:
            loaded = self.load_checkpoint()
            if not loaded:
                # 如果没有加载到检查点，使用全局模型初始化
                self.model.load_state_dict(self.global_model_ref.state_dict())
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Agent {self.agent_id} initialized with global model weights")
            else:
                print(f"Agent {self.agent_id} resumed from checkpoint")
        else:
            # 使用全局模型初始化
            self.model.load_state_dict(self.global_model_ref.state_dict())
            self.target_model.load_state_dict(self.model.state_dict())

    def set_global_weights(self, global_weights):
        """存储全局模型权重以用于FedProx计算"""
        self.global_model_weights = global_weights

    def save_checkpoint(self, save_dir="trained_model"):
        """在offline模式下跳过保存"""
        if self.offline_mode:
            return None
        """保存完整的训练状态"""
        dir=os.path.join(config.get('log_dir'), save_dir)
        os.makedirs(dir, exist_ok=True)

        checkpoint = {
            # 模型状态
            'model_state': self.model.state_dict(),
            'target_model_state': self.target_model.state_dict(),

            # 优化器状态
            'optimizer_state': self.optimizer.state_dict(),

            # 训练参数
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,

            # 网络结构参数
            'n_destroy_actions': self.n_destroy_actions,
            'n_repair_actions': self.n_repair_actions,
            'state_dim': self.state_dim,
            'n_actions': self.n_actions,

            # 操作算子权重
            'operator_values': self.operator_values,

            # 超参数（用于验证兼容性）
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update_frequency': self.target_update_frequency,

            # 元信息
            'agent_id': self.agent_id,
            'save_timestamp': time.time(),
            'arc_count': self.arc_count
        }

        save_path = os.path.join(dir, f"fragment_{self.agent_id}.pt")
        torch.save(checkpoint, save_path)

        # --- 新增：保存Replay Buffer ---
        buffer_save_path = os.path.join(dir, f"fragment_{self.agent_id}_buffer.pkl")
        self.memory.save_buffer(buffer_save_path)

        print(f"Agent {self.agent_id} checkpoint saved to {save_path}")
        return save_path

    def load_checkpoint(self, save_dir="trained_model"):
        """加载训练状态"""
        dir = os.path.join(config.get('log_dir'), save_dir)
        load_path = os.path.join(dir, f"fragment_{self.agent_id}.pt")

        if not os.path.exists(load_path):
            print(f"No checkpoint found for agent {self.agent_id} at {load_path}")
            return False

        checkpoint = torch.load(load_path, map_location=self.device)

        # 验证兼容性
        if (checkpoint.get('n_destroy_actions') != self.n_destroy_actions or
                checkpoint.get('n_repair_actions') != self.n_repair_actions or
                checkpoint.get('state_dim') != self.state_dim):
            print(f"Agent {self.agent_id}: Checkpoint incompatible with current network structure")
            return False

        # --- 新增：加载Replay Buffer ---
        buffer_load_path = os.path.join(dir, f"fragment_{self.agent_id}_buffer.pkl")
        self.memory.load_buffer(buffer_load_path)

        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state'])
        self.target_model.load_state_dict(checkpoint['target_model_state'])

        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        # 加载训练参数
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.update_counter = checkpoint.get('update_counter', 0)

        # 加载操作算子权重
        if 'operator_values' in checkpoint:
            self.operator_values = checkpoint['operator_values']

        # 加载其他状态
        self.arc_count = checkpoint.get('arc_count', self.arc_count)

        save_time = checkpoint.get('save_timestamp', 0)
        print(f"Agent {self.agent_id} checkpoint loaded from {load_path}")
        print(f"  - Saved at: {time.ctime(save_time)}")
        print(f"  - Epsilon: {self.epsilon:.4f}")
        print(f"  - Update counter: {self.update_counter}")

        return True

    def set_arc_count(self, arc_count):
        """设置弧段数量并初始化VAE管理器"""
        self.arc_count = arc_count

    def sync_with_global(self):
        """Synchronize local model with global model"""
        # This needs to be called explicitly when needed
        self.model.load_state_dict(self.global_model_ref.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in local buffer - 在offline模式下跳过"""
        if self.offline_mode:
            return  # 跳过经验存储
        """Store experience in local buffer"""
        self.memory.push((state, action, reward, next_state, done))

    def update_current_solution(self, solution):
        """
        更新当前解和对应的潜在向量

        Args:
            solution: 弧段ID列表
        """
        self.current_solution = solution

    def train_local(self):
        """Train the local model - 在offline模式下跳过"""
        if self.offline_mode:
            return None  # 跳过训练

        """Train the local model"""
        if len(self.memory) < self.batch_size:
            return None
        # Sample batch from buffer
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None
            
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        
        # Calculate current Q values
        q_values = self.model(states)
        
        # Get Q values for taken actions
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q values with Double DQN
        with torch.no_grad():
            # Select actions with online network
            next_q_values = self.model(next_states)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            
            # Evaluate with target network
            next_target_q_values = self.target_model(next_states)
            next_target_values = next_target_q_values.gather(1, next_actions).squeeze(1)
            
            # Calculate target values
            target = rewards + (1 - dones.float()) * self.gamma * next_target_values
        
        # Calculate loss with Huber loss (more robust than MSE)
        loss = nn.SmoothL1Loss()(current_q, target)

        # --- FedProx 核心改动开始 ---
        # 如果 mu > 0 且全局模型权重已设置，则添加近端项
        if self.mu > 0 and self.global_model_weights is not None:
            proximal_term = 0.0
            for local_param, global_param in zip(self.model.parameters(), self.global_model_weights):
                proximal_term += (local_param - global_param).norm(2)
            loss += (self.mu / 2) * proximal_term
        # --- FedProx 核心改动结束 ---
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.loss_log=loss.item()

        # --- 【新增】计算并记录TD-Error ---
        with torch.no_grad():
            td_error = (target - current_q).abs().mean().item()
        # 在方法末尾的 return loss.item() 之前，我们已经有了 td_error 变量
        # 注意：这个 td_error 需要被传递给 metrics_logger.log_step
        # 由于您的 ALNS.run 中调用 log_step 时没有直接传递 td_error
        # 一个简单的做法是将其暂存到 agent 的一个属性中
        self.last_td_error = td_error  # <-- 将td_error暂存起来

        # Update target network periodically
        self.update_counter += 1
        # if self.update_counter % self.target_update_frequency == 0:
        #     self.target_model.load_state_dict(self.model.state_dict())
        # 替换硬更新为软更新
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_network()
            
        # Track metrics
        # self.train_metrics["loss"].append(loss.item())
        # self.train_metrics["epsilon"].append(self.epsilon)
        # self.train_metrics["avg_q_value"].append(current_q.mean().item())
        
        return loss.item()
    
    def update_operator(self):
        """Update operator weights - 在offline模式下跳过"""
        if self.offline_mode:
            return  # 跳过算子权重更新
        """Update operator weights based on recent experiences"""

        # 确保所有值为正数
        self.operator_values = np.maximum(self.operator_values, 0.0)

        # Reset weights to avoid dominance
        if np.max(self.operator_values) > 5 * np.min(self.operator_values):
            self.operator_values = np.ones(len(self.operator_values)) / len(self.operator_values)
            
        # Simple normalization to ensure proper distribution
        total = np.sum(self.operator_values)
        if total > 0:
            self.operator_values = self.operator_values / total
        else:
            self.operator_values = np.ones(len(self.operator_values)) / len(self.operator_values)


    def act(self, state, temperature=1.0):
        """Select action with epsilon-greedy strategy"""
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: select randomly but with bias toward more promising operators
            if random.random() < 0.5:  # 50% chance of using operator values
                # Use learned operator values to guide exploration
                destroy_values = self.operator_values[:self.n_destroy_actions]
                repair_values = self.operator_values[self.n_destroy_actions:]

                # Normalize for probability distribution
                # destroy_probs = destroy_values / np.sum(destroy_values) if np.sum(destroy_values) > 0 else None
                # repair_probs = repair_values / np.sum(repair_values) if np.sum(repair_values) > 0 else None
                def safe_prob_normalization(values):
                    """保证计算出的概率分布有效"""
                    # 先将所有值偏移到非负区域
                    shifted = values - np.min(values) + 1e-10
                    # 标准化为概率
                    probs = shifted / np.sum(shifted)
                    # 确保概率之和为1
                    probs = probs / np.sum(probs)
                    return probs
                destroy_probs = safe_prob_normalization(destroy_values)
                repair_probs = safe_prob_normalization(repair_values)

                # Select based on weighted probabilities or uniform if normalization failed
                if destroy_probs is not None:
                    destroy_id = np.random.choice(self.n_destroy_actions, p=destroy_probs)
                else:
                    destroy_id = random.randint(0, self.n_destroy_actions - 1)

                if repair_probs is not None:
                    repair_id = np.random.choice(self.n_repair_actions, p=repair_probs)
                else:
                    repair_id = random.randint(0, self.n_repair_actions - 1)
            else:
                # Pure random selection
                destroy_id = random.randint(0, self.n_destroy_actions - 1)
                repair_id = random.randint(0, self.n_repair_actions - 1)
        else:
            # Exploit: select best action according to Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                # 【新增】添加记录器
                self._last_q_values = q_values.cpu().numpy().flatten()
                # print(self._last_q_values)

            # Get Q-values for each action combination
            q_values_np = q_values.cpu().numpy().flatten()

            # Get joint action with highest Q-value
            action_id = np.argmax(q_values_np)
            destroy_id = action_id // self.n_repair_actions
            repair_id = action_id % self.n_repair_actions

            # Update operator values for better exploration later
            for d in range(self.n_destroy_actions):
                d_indices = [d * self.n_repair_actions + r for r in range(self.n_repair_actions)]
                self.operator_values[d] = 0.9 * self.operator_values[d] + 0.1 * np.mean(q_values_np[d_indices])

            for r in range(self.n_repair_actions):
                r_indices = [d * self.n_repair_actions + r for d in range(self.n_destroy_actions)]
                self.operator_values[self.n_destroy_actions + r] = 0.9 * self.operator_values[self.n_destroy_actions + r] + 0.1 * np.mean(q_values_np[r_indices])

            # 【新增】 缓存算子价值
            self._last_operator_values = self.operator_values.copy()

            # ✅ 新增：记录选中的action（在所有分支最后统一记录）
            self._last_selected_action_id = action_id#destroy_id * self.n_repair_actions + repair_id

        return destroy_id, repair_id

    #softmax
    # def act(self, state,temperature=1.0):
    #     """Select action with epsilon-greedy strategy"""
    #     # Epsilon-greedy exploration
    #     if np.random.random() < self.epsilon:
    #         # Explore: select randomly but with bias toward more promising operators
    #         if random.random() < 0.5:  # 50% chance of using operator values
    #             # Use learned operator values to guide exploration
    #             destroy_values = self.operator_values[:self.n_destroy_actions]
    #             repair_values = self.operator_values[self.n_destroy_actions:]
    #
    #             # Normalize for probability distribution
    #             # destroy_probs = destroy_values / np.sum(destroy_values) if np.sum(destroy_values) > 0 else None
    #             # repair_probs = repair_values / np.sum(repair_values) if np.sum(repair_values) > 0 else None
    #             def safe_prob_normalization(values):
    #                 """保证计算出的概率分布有效"""
    #                 # 先将所有值偏移到非负区域
    #                 shifted = values - np.min(values) + 1e-10
    #                 # 标准化为概率
    #                 probs = shifted / np.sum(shifted)
    #                 # 确保概率之和为1
    #                 probs = probs / np.sum(probs)
    #                 return probs
    #
    #             destroy_probs = safe_prob_normalization(destroy_values)
    #             repair_probs = safe_prob_normalization(repair_values)
    #
    #             # Select based on weighted probabilities or uniform if normalization failed
    #             if destroy_probs is not None:
    #                 destroy_id = np.random.choice(self.n_destroy_actions, p=destroy_probs)
    #             else:
    #                 destroy_id = random.randint(0, self.n_destroy_actions - 1)
    #
    #             if repair_probs is not None:
    #                 repair_id = np.random.choice(self.n_repair_actions, p=repair_probs)
    #             else:
    #                 repair_id = random.randint(0, self.n_repair_actions - 1)
    #         else:
    #             # Pure random selection
    #             destroy_id = random.randint(0, self.n_destroy_actions - 1)
    #             repair_id = random.randint(0, self.n_repair_actions - 1)
    #     else:
    #         # Exploit: select best action according to Q-values
    #         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #         with torch.no_grad():
    #             q_values = self.model(state_tensor)
    #             # 【新增】添加记录器
    #             self._last_q_values = q_values.cpu().numpy().flatten()
    #             # print(self._last_q_values)
    #
    #         # Get Q-values for each action combination
    #         q_values_np = q_values.cpu().numpy().flatten()
    #
    #         # # Get joint action with highest Q-value
    #         # action_id = np.argmax(q_values_np)
    #         # destroy_id = action_id // self.n_repair_actions
    #         # repair_id = action_id % self.n_repair_actions
    #
    #         # Softmax采样（替换ε-greedy）
    #         q_norm = q_values_np - np.max(q_values_np)
    #         exp_q = np.exp(q_norm / temperature)
    #         probs = exp_q / np.sum(exp_q)
    #         action_id = np.random.choice(len(q_values_np), p=probs)
    #         destroy_id = action_id // self.n_repair_actions
    #         repair_id = action_id % self.n_repair_actions
    #
    #         # Update operator values for better exploration later
    #         for d in range(self.n_destroy_actions):
    #             d_indices = [d * self.n_repair_actions + r for r in range(self.n_repair_actions)]
    #             self.operator_values[d] = 0.9 * self.operator_values[d] + 0.1 * np.mean(q_values_np[d_indices])
    #
    #         for r in range(self.n_repair_actions):
    #             r_indices = [d * self.n_repair_actions + r for d in range(self.n_destroy_actions)]
    #             self.operator_values[self.n_destroy_actions + r] = 0.9 * self.operator_values[
    #                 self.n_destroy_actions + r] + 0.1 * np.mean(q_values_np[r_indices])
    #
    #         # 【新增】 缓存算子价值
    #         self._last_operator_values = self.operator_values.copy()
    #
    #         # ✅ 新增：记录选中的action（在所有分支最后统一记录）
    #         self._last_selected_action_id = action_id  # destroy_id * self.n_repair_actions + repair_id
    #
    #     return destroy_id, repair_id

    # 【新增】 新增：获取Q值和算子指标（供ALNS调用）
    def get_metrics_for_logging(self):
        """获取用于记录的指标"""
        if self._last_q_values is None:
            return None

        # 计算Q值统计
        q_metrics = {
            'avg': float(np.mean(self._last_q_values)),
            'max': float(np.max(self._last_q_values)),
            'min': float(np.min(self._last_q_values)),
            'std': float(np.std(self._last_q_values)),
            'selected': float(self._last_q_values[
                                  self._last_selected_action_id]) if self._last_selected_action_id is not None else 0.0,
            'all_q_values': self._last_q_values.tolist()  # ✅ 保存完整64维Q值
        }

        # 计算分组Q值
        q_destroy_avg = []
        for d in range(self.n_destroy_actions):
            indices = [d * self.n_repair_actions + r for r in range(self.n_repair_actions)]
            q_destroy_avg.append(float(np.mean(self._last_q_values[indices])))

        q_repair_avg = []
        for r in range(self.n_repair_actions):
            indices = [d * self.n_repair_actions + r for d in range(self.n_destroy_actions)]
            q_repair_avg.append(float(np.mean(self._last_q_values[indices])))

        q_metrics['destroy_avg'] = q_destroy_avg
        q_metrics['repair_avg'] = q_repair_avg

        # 算子价值
        operator_values = {
            'destroy_values': self._last_operator_values[:self.n_destroy_actions].tolist(),
            'repair_values': self._last_operator_values[self.n_destroy_actions:].tolist()
        }

        return {
            'q_metrics': q_metrics,
            'operator_values': operator_values
        }

    def update_target_network(self, tau=0.01):
        # 历史版本的软更新逻辑
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

