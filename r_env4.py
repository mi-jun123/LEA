import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from ParameterGenerator import ExternalParameterGenerator


class NetworkSwitchEnv(gym.Env):
    def __init__(self, state_params_config=None):
        # 新增参数配置（单位：snr/sinr(dB), 带宽(Hz)）
        if state_params_config is None:
            state_params_config = []
            # 自组网电台状态（6个）
            adhoc_params = [
                ("rss_0", -120.0, -10.0),
                ("bitrate_0", 0.0, 1e9),
                ("rtt_0", 0.0, 200.0),
                ("cost_0", 0.0, 1.0),
                ("speed_0", 0.0, 50.0),
                ("height_0", 0.0, 1000.0)  # 自组网可能有高度（如无人机）
            ]
            state_params_config.extend(adhoc_params)
            self.num_5g=8
            # 8个5G基站状态（每个6个，共48个）
            for i in range(1, self.num_5g + 1):
                base_params = [
                    (f"rss_5g_{i}", -120.0, -10.0),
                    (f"bitrate_5g_{i}", 0.0, 1e9),
                    (f"rtt_5g_{i}", 0.0, 200.0),
                    (f"cost_5g_{i}", 0.0, 1.0),
                    (f"speed_5g_{i}", 0.0, 50.0),
                    (f"height_5g_{i}", 0.0, 1000.0)
                ]
                state_params_config.extend(base_params)
            self.total_bases=9
            # 其余代码（初始化、step、reset等）沿用原有逻辑，新增参数自动加入状态空间
            # 动作空间：0-自组网，1-5G基站1，2-5G基站2，...，8-5G基站8（共9个动作）
            self.action_space = spaces.Discrete(self.total_bases)  # 0:自组网，1-8:5G基站1-8

            # 状态空间初始化
            self.state_params = [param[0] for param in state_params_config]
            self.state_lows = np.array([param[1] for param in state_params_config], dtype=np.float32)
            self.state_highs = np.array([param[2] for param in state_params_config], dtype=np.float32)
            self.observation_space = spaces.Box(
                low=self.state_lows,
                high=self.state_highs,
                dtype=np.float32
            )
            self.state = {}
            self.num_ad=1
            self.prev_network=0
            self.default_speed = 10  # m/s
            self.default_height = 30  # m
            # 计算归一化参数
            self.state_mins = self.state_lows
            self.state_maxs = self.state_highs
            # 确保在调用 reset 之前定义 self.state_params_config
            self.state_params_config = state_params_config
            self.util_boundaries = self._init_util_boundaries()
            self.reset()





    def calculate_bitrate(self, snr, bandwidth,efficiency=0.85):
        """
        计算可用比特率（ABR）
        :param snr: 信噪比（SNR）
        :param distance: 节点间距离，单位：km
        :param bandwidth: 信道带宽，单位：Hz
        :param frequency: 信号频率，单位：MHz
        :param transmit_power: 发射功率，单位：dBm
        :param noise_power: 噪声功率，单位：dBm
        :param efficiency: 效率因子，通常小于 1
        :return: 可用比特率，单位：bps
        """

        snr_linear = 10 ** (snr / 10)
        abr = efficiency * bandwidth * np.log2(1 + snr_linear)
        return abr


    """
        执行一个动作并更新环境状态。

        参数:
        action (int): 智能体选择的动作，取值为 0（自组网）、1（5G）或 2（保持不变）。
        external_params (dict, optional): 外接的网络参数，键为参数名，值为参数值。

        返回:
        tuple: 包含以下元素的元组
            - observation (np.ndarray): 新的观测状态，包含所有状态参数。
            - reward (float): 执行动作后获得的奖励。
            - terminated (bool): 表示环境是否终止，这里始终为 False。
            - truncated (bool): 表示环境是否被截断，这里始终为 False。
            - info (dict): 包含额外信息的字典，如状态参数的变化、当前状态值。
    """
    def calculate_UB(self,eta1, b, b_min=2,b0=0.075,B0=0.49,theta=97.8):

        # 单位阶跃函数判断
        if b/1e6 >= b_min:
            step = 1
            # 计算对数部分
            denominator = b / 1e6 - b0
            if denominator == 0:
                raise ValueError("分母不能为零，rj - r0 不能等于 0")
            dist_b = theta / denominator + B0
            log_part = math.log10((255 ** 2) / dist_b)
            #print(f"log_part:{log_part}")
            # 计算最终结果
            UB = eta1 * 10 * log_part * step
        else:
            UB = 0


        return UB

    def calculate_UD(self,eta2, d, d_min=100):

        # 实现单位阶跃函数逻辑
        if d_min - d >= 0:
            unit_step = 1
        else:
            unit_step = 0

        UD=eta2*(d_min-d)/d_min* unit_step
        return UD

    def calculate_URSS(self,eta3, rss, rssmin=-90):

        # 处理单位阶跃函数
        if rss >= rssmin:
            unit_step = 1
        else:
            unit_step = 0

        URSS=eta3 * rss * unit_step
        return URSS

    def calculate_UC(self,network_type,c0,c1):

        if network_type== "adhoc_0":
            UC=1.0
        else:
            UC=c1/c0
        return  -UC

    def heaviside_step(self, x):
        """
        实现Heaviside阶跃函数，处理多种输入类型

        参数:
        x: 输入值（标量、列表或NumPy数组）

        返回:
        NumPy数组，元素为0或1
        """
        # 确保输入是NumPy数组
        x_arr = np.asarray(x)

        # 计算阶跃函数
        return np.where(x_arr >= 0, 1, 0)

    def calculate_v(self, v,k_mode, v_th=15, k1=0.1):
        """
        计算电压相关的动态方程

        参数:
        n : 神经元索引
        v : 电压值
        v_th : 电压阈值
        k1 : 基础k值
        k_mode : k值选择模式，1表示使用k1，2表示使用k1*1.5，3表示使用k1*0.5

        返回:
        计算结果
        """
        # 根据k_mode选择不同的k值
        if k_mode == 0:
            k = k1
        else:
            k = k1 * 0.1  # 示例：使用k1的1.5倍


        term1 = k * self.heaviside_step(v - v_th)* ((v - v_th) / v_th)
        term2 =self.heaviside_step(v_th - v)* ((v_th - v) / v_th)
        return term1 + term2

    def calculate_h(self, h, k_mode,h_th=120, k2=0.1 ):
        """
        计算门控变量相关的动态方程

        参数:
        n : 神经元索引
        h : 门控变量值
        h_th : 门控变量阈值
        k2 : 基础k值
        k_mode : k值选择模式，1表示使用k2，2表示使用k2*1.5，3表示使用k2*0.5

        返回:
        计算结果
        """
        # 根据k_mode选择不同的k值
        if k_mode == 0:
            k = k2
        else:
            k = k2 * 0.1  #
        #print(f"{k}")
        term1 = k * ((h - h_th) / h_th) *self.heaviside_step(h - h_th)
        term2 = ((h_th - h) / h_th) * self.heaviside_step(h_th - h)
        #print(f"term2{term2}")
        return term1 + term2

    def _init_util_boundaries(self):
        """
        从状态参数配置中推导各效用的上下界：
        - Ub: bitrate [b_min, b_max] → Ub上下界
        - Ud: rtt [0, d_max] → Ud上下界（0或eta2）
        - U_Rs: rss [rssmin, 0] → U_Rs上下界（0到-eta3*rssmin）
        - Uc: [0.1, 1]（5G成本比）
        """


        # 计算各效用理论极值
        return {
            "Ub": {
                "min": self.calculate_UB(eta1=0.5, b=2e6, b_min=2),  # b_min时的Ub
                "max": self.calculate_UB(eta1=0.5, b=1e8, b_min=2)  # b_max时的Ub
            },
            "Ud": {
                "min": 0,  # d > d_min时Ud=0
                "max": 100  # eta2=0.6（d ≤ d_min时）
            },
            "U_Rs": {
                "min": 0,  # rss < rssmin时U_Rs=0
                "max": self.calculate_URSS(eta3=0.7, rss=-90,)  # rss=rssmin时的最大值
            },
            "Uc": {
                "min": 0,
                "max": 5  # 自组网成本
            },
            "Us": {
                "min": self.calculate_v(v=0,k_mode=1),  #
                "max": self.calculate_v(v=50,k_mode=0)  # m/s
            },
            "Uh": {
                "min": self.calculate_h(h=0,k_mode=1),  # m
                "max": self.calculate_h(h=1000,k_mode=0)  # m
            }
        }

    def _init_util_boundaries2(self):
        """初始化效用边界（参数名 -> 最小/最大值）"""
        util_boundaries = {
            # 自组网参数边界（与5G基站共用相同边界）
            "rss": {"min": -120.0, "max": -10.0},  # 信号强度（RSS）
            "bitrate": {"min": 0.0, "max": 1e8},  # 比特率
            "rtt": {"min": 0.0, "max": 200.0},  # 时延（RTT）
            "cost": {"min": 0.0, "max": 1.0},  # 成本
            "speed": {"min": 0.0, "max": 50.0},  # 速度
            "height": {"min": 0.0, "max": 1000.0},  # 高度
        }
        return util_boundaries

    def _get_param_to_util_mapping(self):
        """生成参数名到效用类型的映射（基于前缀匹配）"""
        param_to_util = {
            # 自组网参数
            "rss_adhoc_0": "rss",
            "bitrate_adhoc_0": "bitrate",
            "rtt_adhoc_0": "rtt",
            "c_adhoc_0": "cost",
            "v_adhoc_0": "speed",
            "h_adhoc_0": "height",

            # 5G基站参数（1-8号）
            **{f"rss_5g_{i}": "rss" for i in range(1, 9)},
            **{f"bitrate_5g_{i}": "bitrate" for i in range(1, 9)},
            **{f"rtt_5g_{i}": "rtt" for i in range(1, 9)},
            **{f"c_5g_{i}": "cost" for i in range(1, 9)},
            **{f"v_5g_{i}": "speed" for i in range(1, 9)},
            **{f"h_5g_{i}": "height" for i in range(1, 9)},
        }
        return param_to_util

    def normalize_state(self, state_dict):
        """根据效用边界对状态进行归一化，返回数组格式"""
        # 初始化边界和映射
        util_boundaries = self._init_util_boundaries2()
        param_to_util = self._get_param_to_util_mapping()

        # 定义参数顺序（与你的配置一致）
        PARAM_ORDER = [
            # 自组网参数
            "rss_adhoc_0", "bitrate_adhoc_0", "rtt_adhoc_0", "c_adhoc_0", "v_adhoc_0", "h_adhoc_0",
            # 5G基站参数（1-8号）
            *[f"rss_5g_{i}" for i in range(1, 9)],
            *[f"bitrate_5g_{i}" for i in range(1, 9)],
            *[f"rtt_5g_{i}" for i in range(1, 9)],
            *[f"c_5g_{i}" for i in range(1, 9)],
            *[f"v_5g_{i}" for i in range(1, 9)],
            *[f"h_5g_{i}" for i in range(1, 9)],
        ]

        # 创建归一化数组
        normalized_array = np.zeros(len(PARAM_ORDER), dtype=np.float32)

        # 填充归一化值
        for idx, param in enumerate(PARAM_ORDER):
            if param not in state_dict:
                print(f"警告：状态字典缺少参数 '{param}'，使用默认值0")
                continue

            value = state_dict[param]
            if param not in param_to_util:
                print(f"警告：参数 '{param}' 无效用类型映射，使用原值")
                normalized_array[idx] = value
                continue

            util_type = param_to_util[param]
            bounds = util_boundaries[util_type]
            min_val, max_val = bounds["min"], bounds["max"]

            # 处理边界值相同的情况（避免除零错误）
            if max_val == min_val:
                normalized_array[idx] = 0.0
            else:
                normalized_array[idx] = (value - min_val) / (max_val - min_val)

        return normalized_array
    def calculate_G(self, omega_b, omega_d, omega_Rs, omega_c, omega_s, omega_h, Ub, Ud, U_Rs, Uc, Us, Uh):
        """
        计算 G(x, a_t) 函数
        :param omega_b: 比特率效用权重
        :param omega_d: 时延效用权重
        :param omega_Rs: 信号强度效用权重
        :param omega_c: 成本效用权重
        :param omega_s: 速度效用权重
        :param omega_h: 高度效用权重
        :param Ub: 比特率效用值
        :param Ud: 时延效用值
        :param U_Rs: 信号强度效用值
        :param Uc: 成本效用值
        :param Us: 速度效用值
        :param Uh: 高度效用值
        :return: G(x, a_t) 的计算结果
        """
        # 独立 Min - Max 归一化（核心修改）
        normalized = {
            "Ub": self._min_max_norm(Ub, self.util_boundaries["Ub"]),
            "Ud": self._min_max_norm(Ud, self.util_boundaries["Ud"]),
            "U_Rs": self._min_max_norm(U_Rs, self.util_boundaries["U_Rs"]),
            "Uc": self._min_max_norm(Uc, self.util_boundaries["Uc"]),
            "Us": self._min_max_norm(Us, self.util_boundaries["Us"]),
            "Uh": self._min_max_norm(Uh, self.util_boundaries["Uh"])
        }

        # 加权融合（使用归一化后的值）
        omega = np.array([omega_b, omega_d, omega_Rs, omega_c, omega_s, omega_h])  # 权重向量
        G_value = np.sum(omega * np.array([
            normalized["Ub"],
            normalized["Ud"],
            normalized["U_Rs"],
            normalized["Uc"],
            normalized["Us"],
            normalized["Uh"]
        ]))
        #C=self._min_max_norm(Uc, self.util_boundaries["Uc"]),
        #print(f"C:{C}")
        return float(G_value)
    def _min_max_norm(self, value, boundaries):
        """
        独立维度Min-Max归一化：
        norm_value = (value - boundary.min) / (boundary.max - boundary.min + eps)
        """
        eps = 1e-8
        return (value - boundaries["min"]) / (boundaries["max"] - boundaries["min"] + eps)

    def calculate_Q(self, alpha, G_value, a_t,prev_network):
        # 只维护indicator_same变量
        indicator_same = 1 if a_t == prev_network else 0
        indicator_diff = 1 - indicator_same  # 通过计算得到另一个值

        return (1 - alpha) * G_value * indicator_diff + alpha * G_value * indicator_same

    def calculate_reward(self, d_state, action,prev_network):
        # 根据action确定前缀
        if action == 0:  # 自组网
            prefix = "adhoc_0"
        else:  # 5G基站
            prefix = f"5g_{action}"
        rss = d_state.get(f"rss_{prefix}", 0)
        rtt = d_state.get(f"rtt_{prefix}", 0)
        #print(f"rtt{rtt}")
        bitrate = d_state.get(f"bitrate_{prefix}", 0)
        v = d_state.get(f"v_{prefix}", 0)
        h = d_state.get(f"h_{prefix}", 0)
        c1 = d_state["c_5g_1"]
        c0 = d_state["c_adhoc_0"]



        eta1 = 0.5
        eta2 = 0.6
        eta3 = 0.7
        Ub = self.calculate_UB(eta1, bitrate)
        Ud = self.calculate_UD(eta2, rtt)
        U_Rs = self.calculate_URSS(eta3, rss)
        Uc = self.calculate_UC(prefix, c0, c1)
        Us=self.calculate_v(v,action)
        Uh=self.calculate_h(h,action)
        omega_b = 0.44548085
        omega_d = 0.04034934
        omega_Rs = 0.28216224
        omega_c = 0.05674012
        omega_s=0.09580386
        omega_h=0.07946359
        G_value = self.calculate_G(omega_b, omega_d, omega_Rs, omega_c,omega_h, omega_s,Ub, Ud, U_Rs, Uc,Us,Uh)

        alpha = 0.6
        Q_value = self.calculate_Q(alpha, G_value, action,prev_network)
        return Q_value,rss,rtt,bitrate,v,h,c1,c0

    def update_external_states(self, external_params):
        # 遍历9个基站/节点
        params = external_params

        # 遍历所有基站/节点
        for i in range(len(params['SNR'])):
            # 获取第i个基站的参数
            snr_i = params['SNR'][i]
            rss_i = params['RSS'][i]
            rtt_i = params['RTT'][i]

            # 根据索引分配正确的网络类型和编号
            if i == 0:  # 索引0固定为自组网
                prefix = "adhoc_0"
                self.state[f"c_{prefix}"]=0.1
            else:  # 其余索引为5G基站（编号从1开始）
                prefix = f"5g_{i}"
                self.state[f"c_{prefix}"] = 1

            # 更新状态空间
            self.state[f"rss_{prefix}"] = rss_i
            self.state[f"rtt_{prefix}"] = rtt_i

            # 计算并更新比特率
            if prefix.startswith("5g"):
                # 5G使用SINR计算比特率
                bandwidth = 100e6 # 5G默认100MHz
                bitrate = self.calculate_bitrate(snr=snr_i, bandwidth=bandwidth)
                #print(f"{snr_i}")
                #print(f"5g{bandwidth}")
            else:
                # 自组网使用SNR计算比特率
                bandwidth = 20e6 # 自组网默认10MHz
                bitrate = self.calculate_bitrate(snr=snr_i, bandwidth=bandwidth)
                #print(f"{snr_i}")
                #print(f"ad{bandwidth}")

            self.state[f"bitrate_{prefix}"] = bitrate


            # 在更新状态时使用
            self.state[f"v_{prefix}"] = self.default_speed
            self.state[f"h_{prefix}"] = self.default_height

        #print(f"{self.state}")


    def step(self, action, external_params=None):
        terminated = False
        truncated = False


        # 获取初始状态并赋值给self.state
        t_state = external_params
        #print(f"t_state:{t_state}")
        if external_params is not None:
            self.update_external_states(t_state)

        # 计算奖励（传入归一化状态）
        reward , rss, rtt, bitrate, v, h, c1, c0= self.calculate_reward(self.state, action,self.prev_network)
        if self.prev_network==action:
            if reward<0.35:
                reward-=0.05
        #print(f"reward:{reward}")
        if action==0:
            c=c0
        else:
            c=c1
        #print(f"reward:{reward}")
        observation=self.normalize_state(self.state)

        #print(f"observation:{observation}")
        self.prev_network=action


        return (
            observation,
            float(reward),
            terminated,
            truncated,
        )

    def reset(self, ini_coordinate=np.array([-199.9, 199.9]),seed=None, options=None):
        # 设置随机种子，保证结果可复现
        if seed is not None:
            np.random.seed(seed)
            generator_seed = seed  # 使用传入的种子初始化生成器
        else:
            generator_seed = 42  # 若未传入种子，使用默认值

        generator = ExternalParameterGenerator(
            seed=generator_seed  # 使用统一的种子值
        )

        # 初始化环境
        generator.calculate_all_rand_walk(ini_coordinate)
        self.prev_network=generator.get_nearest_eNB(0)
        # 获取初始状态并赋值给self.state
        self.update_external_states(generator.get_t_moment_params(0,self.default_height))
        observation=self.normalize_state(self.state)
        return observation

    def close(self):
        """
        关闭环境并清理资源。

        这里没有需要清理的资源，所以方法为空。
        """
        # 清理资源（如果需要）
        pass




