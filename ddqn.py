import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
from typing import Dict, Tuple, Union


class DQN_agent:
    """深度Q网络智能体（支持Double DQN、决斗网络、经验回放）"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Tuple[int, ...] = (64, 64),
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 exploration_noise: float = 0.1,
                 batch_size: int = 256,
                 replay_capacity: int = 1000000,
                 duel: bool = True,
                 double_dqn: bool = True,
                 device: Union[str, torch.device] = "auto"):
        """
        初始化DQN智能体

        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度（离散）
            hidden_dims (Tuple[int]): 隐藏层维度（默认: (64, 64)）
            learning_rate (float): 学习率（默认: 1e-4）
            gamma (float): 折扣因子（默认: 0.99）
            tau (float): 目标网络软更新系数（默认: 0.005）
            exploration_noise (float): ε-贪心探索率（默认: 0.1）
            batch_size (int): 训练批次大小（默认: 256）
            replay_capacity (int): 经验回放容量（默认: 1e6）
            duel (bool): 是否使用决斗网络（默认: True）
            double_dqn (bool): 是否使用Double DQN（默认: True）
            device (str/torch.device): 计算设备（"auto"自动选择GPU/CPU）
        """
        # 设备配置
        self.device = torch.device(device) if isinstance(device, str) else device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 超参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.exp_noise = exploration_noise
        self.batch_size = batch_size
        self.duel = duel
        self.double = double_dqn

        # 网络初始化
        self.q_net = self._build_network(duel).to(self.device)
        self.q_target = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=replay_capacity,
            device=self.device
        )

        # 冻结目标网络（仅软更新）
        for param in self.q_target.parameters():
            param.requires_grad = False

    def _build_network(self, duel: bool) -> nn.Module:
        """构建Q网络（支持普通/决斗结构）"""
        if duel:
            return DuelQNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=self.hidden_dims
            )
        else:
            return QNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=self.hidden_dims
            )

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        根据当前状态选择动作（ε-贪心策略）

        参数:
            state (np.ndarray): 状态（形状: (state_dim,)）
            deterministic (bool): 是否确定性选择（推理时设为True）

        返回:
            int: 选择的动作（0 ≤ action < action_dim）
        """
        if not isinstance(state, np.ndarray) or state.shape != (self.state_dim,):
            raise ValueError(f"状态格式错误，期望 (state_dim,), 得到 {state.shape}")

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)

            if deterministic or np.random.rand() >= self.exp_noise:
                action = q_values.argmax(dim=1).item()  # 贪心选择
            else:
                action = np.random.randint(0, self.action_dim)  # 随机探索

        return action

    def update(self) -> Dict[str, float]:
        """
        从经验回放中采样并更新Q网络

        返回:
            Dict[str, float]: 训练指标（如损失值）
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}  # 缓冲区未满，跳过训练

        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            if self.double:
                # 在线网络选动作，目标网络评估
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                target_q = self.q_target(next_states).gather(1, next_actions)
            else:
                # 传统DQN（目标网络选动作并评估）
                target_q = self.q_target(next_states).max(dim=1, keepdim=True)[0]

            # 折扣累积奖励
            target_q = rewards + self.gamma * (1 - dones) * target_q  # 处理终止状态

        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions)

        # 损失函数（Huber损失，更鲁棒）
        loss = F.smooth_l1_loss(current_q, target_q)

        # 优化步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络（Polyak平均）
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"loss": loss.item()}

    def store_experience(self,
                         state: np.ndarray,
                         action: int,
                         reward: float,
                         next_state: np.ndarray,
                         done: bool) -> None:
        """存储经验到回放缓冲区"""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save(self, path: str, filename: str) -> None:
        """保存模型参数到文件"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            "q_net_state_dict": self.q_net.state_dict(),
            "q_target_state_dict": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, os.path.join(path, filename))

    def load(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件 {file_path} 不存在")

        checkpoint = torch.load(file_path, map_location=self.dvc)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.q_target.load_state_dict(checkpoint["q_target_state_dict"])

    def __len__(self) -> int:
        """返回经验回放缓冲区大小"""
        return len(self.replay_buffer)


# === 网络结构定义 ===
class QNetwork(nn.Module):
    """普通Q网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super(QNetwork, self).__init__()
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DuelQNetwork(nn.Module):
    """决斗Q网络（状态值+优势函数）"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super(DuelQNetwork, self).__init__()
        # 共享特征提取层
        self.feature = nn.Sequential()
        in_dim = state_dim
        for i, h_dim in enumerate(hidden_dims):
            self.feature.add_module(f"fc{i}", nn.Linear(in_dim, h_dim))
            self.feature.add_module(f"relu{i}", nn.ReLU())
            in_dim = h_dim

        # 状态值函数（V(s)）
        self.value = nn.Linear(in_dim, 1)
        # 优势函数（A(s,a)）
        self.advantage = nn.Linear(in_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value(feat)
        advantage = self.advantage(feat)
        # 消除优势函数均值偏差
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# === 经验回放缓冲区 ===
class ReplayBuffer:
    """高效经验回放缓冲区（支持GPU加速）"""

    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.reset()

        # 预分配内存（使用PyTorch张量）
        self.states = torch.zeros((capacity, state_dim), device=device, dtype=torch.float32)
        self.actions = torch.zeros((capacity, 1), device=device, dtype=torch.int64)
        self.rewards = torch.zeros((capacity, 1), device=device, dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), device=device, dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), device=device, dtype=torch.bool)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """添加单条经验"""
        idx = self.ptr % self.capacity
        self.states[idx] = torch.FloatTensor(state).to(self.device)
        self.actions[idx] = torch.LongTensor([action]).to(self.device)
        self.rewards[idx] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[idx] = torch.FloatTensor(next_state).to(self.device)
        self.dones[idx] = torch.BoolTensor([done]).to(self.device)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """采样批量经验"""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def reset(self) -> None:
        """重置缓冲区"""
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return self.size