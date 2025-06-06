import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy
import time
import os
def build_net(layer_shape, activation, output_activation):
	'''Build networks with For loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)

class Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.Q = build_net(layers, nn.ReLU, nn.Identity)
	def forward(self, s):
		q = self.Q(s)
		return q


class Duel_Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Duel_Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape)
		self.hidden = build_net(layers, nn.ReLU, nn.ReLU)
		self.V = nn.Linear(hid_shape[-1], 1)
		self.A = nn.Linear(hid_shape[-1], action_dim)

	def forward(self, s):
		s = self.hidden(s)
		Adv = self.A(s)
		V = self.V(s)
		Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
		return Q


class DQN_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))
		if self.Duel:
			self.q_net = Duel_Q_Net(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		else:
			self.q_net = Q_Net(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False

	def select_action(self, state, deterministic):
		if not isinstance(state, np.ndarray):
			raise ValueError("Input state must be a numpy array.")
		if state.shape != (self.state_dim,):
			raise ValueError(f"Input state must have shape ({self.state_dim},).")

		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if deterministic:
				a = self.q_net(state).argmax().item()
			else:
				if np.random.rand() < self.exp_noise:
					a = np.random.randint(0, self.action_dim)
				else:
					a = self.q_net(state).argmax().item()
		return a

	def train(self):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		'''Compute the target Q value'''
		with torch.no_grad():
			if self.Double:
				argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
				max_q_next = self.q_target(s_next).gather(1,argmax_a)
			else:
				max_q_next = self.q_target(s_next).max(1)[0].unsqueeze(1)
			target_Q = r + self.gamma * max_q_next * (~dw)#dw: die or win

		# Get current Q estimates
		current_q = self.q_net(s)
		current_q_a = current_q.gather(1,a)

		q_loss = F.mse_loss(current_q_a, target_Q)
		#lr = lr* (self.κ_inc if error_ratio < self.δ else self.κ_dec)
		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		self.q_net_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return q_loss.item()
	def save(self, algo, EnvName, steps, train_round):
		# 创建保存目录
		model_dir = "./models"
		os.makedirs(model_dir, exist_ok=True)

		# 生成唯一模型文件名（加入训练轮次和时间戳）
		timestamp = int(time.time())
		file_name = f"{algo}_{EnvName}_step{steps}_round{train_round}_{timestamp}.pth"
		torch.save(
			{
				"q_net_state_dict": self.q_net.state_dict(),
				"q_target_state_dict": self.q_target.state_dict()
			},
			os.path.join(model_dir, file_name)
		)

	def load(self, file_path):
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"模型文件 {file_path} 不存在")

		checkpoint = torch.load(file_path, map_location=self.dvc)
		self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
		self.q_target.load_state_dict(checkpoint["q_target_state_dict"])


class ReplayBuffer(object):
	def __init__(self, state_dim, dvc, max_size=int(1e6)):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.dvc)
		self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = a
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]




