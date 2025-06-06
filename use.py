import argparse
import os
import numpy as np
import torch
from DQN import DQN_agent
from r_env4 import NetworkSwitchEnv
from utils import evaluate_policy, str2bool
from ParameterGenerator import ExternalParameterGenerator

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')
parser.add_argument('--EnvIdex', type=int, default=0, help='Index of the environment')

parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(8e5), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(50e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=int(3e2), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.95, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)  # 从字符串转换为 torch.device

# 初始化外部参数生成器


def inference():
    EnvName = 'NetworkSwitchEnv'
    BriefEnvName = 'NSE'
    # 创建推理环境
    env = NetworkSwitchEnv()
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = None

    # 算法设置
    if opt.Duel:
        algo_name = 'Duel'
    else:
        algo_name = ''
    if opt.Double:
        algo_name += 'DDQN'
    else:
        algo_name += 'DQN'

    # 随机种子设置
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:', algo_name, '  Env:', BriefEnvName[opt.EnvIdex], '  state_dim:', opt.state_dim,
          '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')
    param_generator = ExternalParameterGenerator(opt.seed)
    # 构建模型
    opt.exp_noise = 0.0  # 推理时关闭探索噪声
    agent = DQN_agent(**vars(opt))

    # 加载训练好的模型
    model_path = f'models/DuelDDQN_N_step3000_round11_1748198727.pth'
    agent.load(model_path)
    default_speed=10
    N =1000
    H=30#高度为30m
    action_change_count=0
    prev_action = param_generator.get_nearest_eNB(0)
    param_generator.calculate_all_rand_walk(ini_coordinate=np.array([-199.9, 199.9]),speed=default_speed)
    total_steps = 0
    env_seed = opt.seed

    s = env.reset(seed=env_seed)
    done = False
    inner_loop_count = 0
    total_reward = 0

    #print(f"state：{env.state}")
    while not done:
        # 生成实时的外部参数

        external_params = param_generator.get_t_moment_params(inner_loop_count + 1,H)
        #print(f"Generated external params: {external_params}")
        # 使用模型选择动作
        a = agent.select_action(s, deterministic=True)
        print(f"a: {a}")
        if prev_action is not None and (a != prev_action and a == 0):
            action_change_count += 1
        prev_action = a  # 更新上一次的动作
        # 与环境交互
        s_next, r, dw, tr= env.step(a, external_params)
        done = (dw or tr)
        #print(f"s_next: {s_next}")
        #print(f"info: {info}")
        print(f"r:{r}")
        s = s_next
        total_reward += r

        total_steps += 1
        inner_loop_count += 1

        if inner_loop_count >= N:
            done = True
            handover_rate = action_change_count / total_steps
            print(f"内层循环达到 {N} 次，终止当前回合。")
            print(f"切换率: {handover_rate}")  # 新增：打印总变化次数

    print(f"总奖励: {total_reward}")
    env.close()



if __name__ == '__main__':
    inference()