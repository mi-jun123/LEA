import argparse
import os
import shutil
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # 导入 tqdm 库F
from ParameterGenerator import ExternalParameterGenerator
from DQN import DQN_agent
from r_env4 import NetworkSwitchEnv
from utils import evaluate_policy, str2bool
from gym import spaces

'''Hyperparameter Setting'''
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
opt.dvc = torch.device(opt.dvc)  # from str to torch.device
print(opt)

# 初始化外部参数生成器
param_generator = ExternalParameterGenerator()
def main():
    EnvName = 'NetworkSwitchEnv'
    BriefEnvName = 'NSE'
    env = NetworkSwitchEnv()
    eval_env = NetworkSwitchEnv()
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = None

    # 算法设置...
    if opt.Duel:
        algo_name = 'Duel'
    else:
        algo_name = ''
    if opt.Double:
        algo_name += 'DDQN'
    else:
        algo_name += 'DQN'

    # 随机种子设置...
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:', algo_name, '  Env:', BriefEnvName[opt.EnvIdex], '  state_dim:', opt.state_dim,
          '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')

    if opt.write:
        # 生成精确到毫秒的时间戳
        timenow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # 移除最后三位微秒
        # 构建路径（无空格，唯一标识）
        writepath = f'runs/{algo_name}-{BriefEnvName[opt.EnvIdex]}_S{opt.seed}_{timenow}'

        # 创建父目录（如果不存在）
        os.makedirs('runs', exist_ok=True)

        # 清理旧目录（可选）
        if os.path.exists(writepath):
            try:
                shutil.rmtree(writepath)
            except PermissionError:
                print(f"警告：无法删除目录 {writepath}，将继续执行...")

        # 初始化 SummaryWriter
        writer = SummaryWriter(log_dir=writepath)
        print(f"TensorBoard 日志将保存至：{writepath}")  # 可选：打印路径验证

    # 构建模型和回放缓冲区...
    if not os.path.exists('model'):
        os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    if opt.Loadmodel:
        agent.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex)
    losses=0.5
    N=1000
    H=30#高度为30m
    score=4

    param_generator.calculate_all_rand_walk()
    with tqdm(total=opt.Max_train_steps, desc="Training Progress", unit="step") as pbar:
        total_steps = 0


        def should_terminate_inner_loop(inner_loop_count, N):
            return inner_loop_count >= N

        def should_evaluate(total_steps, opt):
            return total_steps % opt.eval_interval == 0

        while total_steps < opt.Max_train_steps:
            s= env.reset(seed=env_seed)
            env_seed += 1
            done = False
            inner_loop_count = 0  # 初始化内层循环计数器

            while not done:
                # 生成实时的外部参数
                external_params = param_generator.get_t_moment_params(inner_loop_count+1,H)
                # 检查外部参数是否正确生成
               # #print(f"Generated external params: {external_params}")

                # 交互和训练逻辑...
                if total_steps < opt.random_steps:
                    a = env.action_space.sample()
                else:
                    a = agent.select_action(s, deterministic=False)

                s_next, r, dw, tr= env.step(a, external_params)  # 传入外部参数
                done = (dw or tr)

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next

                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        losses=agent.train()

                if total_steps % 1000 == 0:
                    agent.exp_noise *= opt.noise_decay

                if should_evaluate(total_steps, opt):
                    score = evaluate_policy(eval_env, H,agent, param_generator, turns=1, max_steps_per_episode=N,
                                            seed=env_seed)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                        writer.add_scalar('loss',losses,global_step=total_steps)
                    print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed,
                          'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))

                # 输出当前时间步信息
                #print(f"Current step: {total_steps}, Reward: {r}, Noise: {agent.exp_noise:.4f}")

                # 更新进度条信息
                pbar.set_postfix({"Reward": f"{r:.2f}", "Noise": f"{agent.exp_noise:.4f}"})

                total_steps += 1
                pbar.update(1)  # 更新进度条

                if total_steps % opt.save_interval == 0:
                    agent.save(algo_name, BriefEnvName[opt.EnvIdex], int(total_steps / 1000),9)

                inner_loop_count += 1  # 增加内层循环计数器
                if should_terminate_inner_loop(inner_loop_count, N):
                    done = True  # 终止内层循环
                    print(f"内层循环达到 {N} 次，终止当前回合。")

    if opt.write:
        writer.close()
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()