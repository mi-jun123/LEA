import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import math
class NetworkSwitchEnv(gym.Env):
    def __init__(self, state_params_config=None):
        # 新增参数配置（单位：snr/sinr(dB), 带宽(Hz)）
        if state_params_config is None:
            state_params_config = [
                # 自组网参数（新增）
                ("snr_ad_hoc", -20.0, 40.0),    # 信噪比（dB）
                ("bandwidth_ad_hoc", 1e6, 1e9), # 带宽（Hz）
                # 原有自组网参数
                ("rss_ad_hoc", -120.0, -10.0),
                ("rtt_ad_hoc", 0.0, 200.0),
                ("bitrate_ad_hoc", 0.0, 1e9),
                ("c_ad_hoc", 0, 1),
                # 5G 参数（新增）
                ("sinr_5g", -20.0, 40.0),      # 信干噪比（dB）
                ("bandwidth_5g", 1e6, 1e9),    # 带宽（Hz）
                # 原有5G参数
                ("rtt_5g", 0.0, 200.0),
                ("rss_5g", -120.0, -10.0),
                ("bitrate_5g", 0.0, 1e9),
                ("c_5g", 0, 1),
                ("current_network", 0, 1)
            ]
        # 其余代码（初始化、step、reset等）沿用原有逻辑，新增参数自动加入状态空间
            # 动作空间：0 - 自组网，1 - 5G，2 - 保持不变
            self.action_space = spaces.Discrete(4)
            self.state_params = [param[0] for param in state_params_config]
            self.state_lows = np.array([param[1] for param in state_params_config], dtype=np.float32)
            self.state_highs = np.array([param[2] for param in state_params_config], dtype=np.float32)

            # 状态空间定义
            self.observation_space = spaces.Box(
                low=self.state_lows,
                high=self.state_highs,
                dtype=np.float32
            )

            # 初始化状态变量
            self.state = {}
            self.base_values = {
                'snr_ad_hoc': 9.466939688322915,
                'bandwidth_ad_hoc': 20000000.0,
                'rss_ad_hoc': -49.13512664779722,
                'rtt_ad_hoc': 20,
                'bitrate_ad_hoc': 22790074.57330997,
                'c_ad_hoc': 0.9,
                'sinr_5g': -6.8993236003105025,
                'bandwidth_5g': 100000000.0,
                'rtt_5g': 40,
                'rss_5g': -51.325971482991115,
                'bitrate_5g': 21446538.179779507,
                'c_5g': 0.1,
                'current_network': 0
            }
            # 计算归一化参数
            self.state_mins = self.state_lows
            self.state_maxs = self.state_highs
            # 确保在调用 reset 之前定义 self.state_params_config
            self.state_params_config = state_params_config
            self.util_boundaries = self._init_util_boundaries()
            self.reset()



    def calculate_bitrate(self, snr, bandwidth,distance=0, frequency=0, transmit_power=20, noise_power=-95,
                          efficiency=0.8):
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
        """
        def calculate_path_loss(distance, frequency):
           
            计算自由空间路径损耗
            :param distance: 节点间距离，单位：km
            :param frequency: 信号频率，单位：MHz
            :return: 路径损耗，单位：dB
            
            return 20 * np.log10(distance) + 20 * np.log10(frequency) + 32.45
        """
       #path_loss = calculate_path_loss(distance, frequency)
        #receive_power = transmit_power - path_loss
        snr_linear = 10 ** (snr / 10)
        abr = efficiency * bandwidth * np.log2(1 + snr_linear)
        return abr

    def normalize_state(self, state):
            """
            对状态进行归一化处理，将状态值缩放到 [0, 1] 范围。
            """
            if isinstance(state, dict):
                normalized_state = {}
                for param in self.state_params:
                    value = state[param]
                    min_val = self.state_mins[self.state_params.index(param)]
                    max_val = self.state_maxs[self.state_params.index(param)]
                    normalized_value = (value - min_val) / (max_val - min_val)
                    normalized_state[param] = normalized_value
                return normalized_state
            elif isinstance(state, np.ndarray):
                normalized_state = (state - self.state_mins) / (self.state_maxs - self.state_mins)
                return normalized_state
            else:
                raise ValueError("Unsupported state type. Expected dict or numpy.ndarray.")
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
        if b >= b_min:
            step = 1
        else:
            step = 0

        # 计算对数部分
        denominator = b/1e6 - b0
        if denominator == 0:
            raise ValueError("分母不能为零，rj - r0 不能等于 0")
        dist_b=theta/denominator+B0
        log_part = math.log10((255 ** 2) / dist_b)

        # 计算最终结果
        UB = eta1 * 10 * log_part * step
        return UB

    def calculate_UD(self,eta2, d, d_min=150):

        # 实现单位阶跃函数逻辑
        if d_min - d >= 0:
            unit_step = 1
        else:
            unit_step = 0

        UD=-eta2 *d* unit_step
        return UD

    def calculate_URSS(self,eta3, rss, rssmin=-70):

        # 处理单位阶跃函数
        if rss >= rssmin:
            unit_step = 1
        else:
            unit_step = 0

        URSS=eta3 * rss * unit_step
        return URSS

    def calculate_UC(self,network_type,c0,c1):

        if network_type== 0:
            UC=1.0
        elif network_type== 1:
            UC=c1/c0
        return  -UC

    def _init_util_boundaries(self):
        """
        从状态参数配置中推导各效用的上下界：
        - Ub: bitrate [b_min, b_max] → Ub上下界
        - Ud: rtt [0, d_max] → Ud上下界（0或eta2）
        - U_Rs: rss [rssmin, 0] → U_Rs上下界（0到-eta3*rssmin）
        - Uc: [0.1, 1]（5G成本比）
        """
        # 从state_params获取参数范围
        params = {p[0]: (p[1], p[2]) for p in self.state_params_config}

        # 计算各效用理论极值
        return {
            "Ub": {
                "min": self.calculate_UB(eta1=0.5, b=2e6, b_min=2),  # b_min时的Ub
                "max": self.calculate_UB(eta1=0.5, b=1e8, b_min=2)  # b_max时的Ub
            },
            "Ud": {
                "min": 0,  # d > d_min时Ud=0
                "max": 30  # eta2=0.6（d ≤ d_min时）
            },
            "U_Rs": {
                "min": 0,  # rss < rssmin时U_Rs=0
                "max": self.calculate_URSS(eta3=0.7, rss=-70, rssmin=-70)  # rss=rssmin时的最大值
            },
            "Uc": {
                "min": 0.1,  # 5G成本比最小值（假设c1/c0=0.1）
                "max": 1.0  # 自组网成本
            }
        }

    def calculate_G(self, omega_b, omega_d, omega_Rs, omega_c, Ub, Ud, U_Rs, Uc):
        """
        计算 G(x, a_t) 函数
        :param omega_b: 比特率效用权重
        :param omega_d: 时延效用权重
        :param omega_Rs: 信号强度效用权重
        :param omega_c: 成本效用权重
        :param Ub: 比特率效用值
        :param Ud: 时延效用值
        :param U_Rs: 信号强度效用值
        :param Uc: 成本效用值
        :return: G(x, a_t) 的计算结果
        """
        #utilities = [Ub, Ud, U_Rs, Uc]
        #

        # 独立Min-Max归一化（核心修改）
        normalized = {
            "Ub": self._min_max_norm(Ub, self.util_boundaries["Ub"]),
            "Ud": self._min_max_norm(Ud, self.util_boundaries["Ud"]),
            "U_Rs": self._min_max_norm(U_Rs, self.util_boundaries["U_Rs"]),
            "Uc": self._min_max_norm(Uc, self.util_boundaries["Uc"])
        }

        # 加权融合（使用归一化后的值）
        omega = np.array([omega_b, omega_d, omega_Rs, omega_c])  # 权重向量
        G_value = np.sum(omega * np.array([
            normalized["Ub"],
            normalized["Ud"],
            normalized["U_Rs"],
            normalized["Uc"]
        ]))

        # ... 动作敏感奖励（同前） ...
        return float(G_value)
    def _min_max_norm(self, value, boundaries):
        """
        独立维度Min-Max归一化：
        norm_value = (value - boundary.min) / (boundary.max - boundary.min + eps)
        """
        eps = 1e-8
        return (value - boundaries["min"]) / (boundaries["max"] - boundaries["min"] + eps)

    def calculate_Q(self, alpha, G_value, a_t):

        # 动作 2 和 3 是保持不变的动作
        if a_t in [2, 3]:
            indicator_same = 1
            indicator_diff = 0
        # 动作 0 和 1 是切换动作
        elif a_t in [0, 1]:
            indicator_same = 0
            indicator_diff = 1
        else:
            raise ValueError(f"Invalid action {a_t}, valid actions are 0-3")

        return (1 - alpha) * G_value * indicator_diff + alpha * G_value * indicator_same

    def calculate_reward(self, normalized_state, action):
        state1 = {}
        for param in normalized_state:
            norm_val = normalized_state[param]
            min_val = self.state_mins[self.state_params.index(param)]
            max_val = self.state_maxs[self.state_params.index(param)]
            original_val = norm_val * (max_val - min_val) + min_val  # 反归一化
            state1[param] = original_val

        if self.state["current_network"] == 0:
            rss = state1["rss_ad_hoc"]
            rtt = state1["rtt_ad_hoc"]
            bitrate = state1["bitrate_ad_hoc"]
        else:
            rss = state1["rss_5g"]
            rtt = state1["rtt_5g"]
            bitrate = state1["bitrate_5g"]

        c1 = state1["c_5g"]
        c0 = state1["c_ad_hoc"]
        eta1 = 0.5
        eta2 = 0.6
        eta3 = 0.7
        Ub = self.calculate_UB(eta1, bitrate)
        Ud = self.calculate_UD(eta2, rtt)
        U_Rs = self.calculate_URSS(eta3, rss)
        Uc = self.calculate_UC(self.state["current_network"], c0, c1)

        omega_b = 0.65
        omega_d = 0.05
        omega_Rs = 0.2
        omega_c = 0.1
        G_value = self.calculate_G(omega_b, omega_d, omega_Rs, omega_c, Ub, Ud, U_Rs, Uc)

        alpha = 0.6
        Q_value = self.calculate_Q(alpha, G_value, action)
        return Q_value

    def update_external_states(self, external_params):
        default_params = {
            "snr_ad_hoc": -20.0, "bandwidth_ad_hoc": 1e6,
            "rss_ad_hoc": -120.0, "rtt_ad_hoc": 200.0, "c_ad_hoc": 0,
            "sinr_5g": -20.0, "bandwidth_5g": 1e6,
            "rtt_5g": 200.0, "rss_5g": -120.0, "c_5g": 0,
            "bitrate_ad_hoc": 0.0, "bitrate_5g": 0.0
        }

        for param in self.state_params:
            if param in ["bitrate_ad_hoc", "bitrate_5g","current_network"]:
                continue  # 比特率单独计算
            original_val = external_params.get(param, default_params[param])
            min_val = self.state_mins[self.state_params.index(param)]
            max_val = self.state_maxs[self.state_params.index(param)]
            normalized_val = (original_val - min_val) / (max_val - min_val+1e-9)
            self.state[param] = normalized_val

            # 计算并归一化自组网比特率
        if "snr_ad_hoc" in external_params and "bandwidth_ad_hoc" in external_params:
            snr = external_params["snr_ad_hoc"]
            bw = external_params["bandwidth_ad_hoc"]
        else:
            snr = default_params["snr_ad_hoc"]
            bw = default_params["bandwidth_ad_hoc"]
        bitrate_ad_hoc_original = self.calculate_bitrate(snr=snr, bandwidth=bw)
        min_bitrate = self.state_mins[self.state_params.index("bitrate_ad_hoc")]
        max_bitrate = self.state_maxs[self.state_params.index("bitrate_ad_hoc")]
        self.state["bitrate_ad_hoc"] = (bitrate_ad_hoc_original - min_bitrate) / (max_bitrate - min_bitrate)

        # 计算并归一化5G比特率
        if "sinr_5g" in external_params and "bandwidth_5g" in external_params:
            sinr = external_params["sinr_5g"]
            bw = external_params["bandwidth_5g"]
        else:
            sinr = default_params["sinr_5g"]
            bw = default_params["bandwidth_5g"]
        bitrate_5g_original = self.calculate_bitrate(snr=sinr, bandwidth=bw)
        min_bitrate = self.state_mins[self.state_params.index("bitrate_5g")]
        max_bitrate = self.state_maxs[self.state_params.index("bitrate_5g")]
        self.state["bitrate_5g"] = (bitrate_5g_original - min_bitrate) / (max_bitrate - min_bitrate)

    def step(self, action, external_params=None):
        old_state = {k: v for k, v in self.state.items()}  # 存储旧状态（归一化值）
        terminated = False
        truncated = False
        prev_network = self.state["current_network"]

        if external_params is not None:
            self.update_external_states(external_params)

        # 执行动作（逻辑不变）
        # 修改点2：动作处理逻辑重构
        if prev_network == 1 and action == 0:  # 5G→自组网
            self.state["current_network"] = 0
        elif prev_network == 0 and action == 1:  # 自组网→5G
            self.state["current_network"] = 1
        elif prev_network == 0 and action == 2:  # 自组网保持
            pass
        elif prev_network == 1 and action == 3:  # 5G保持
            pass
        else:
            # 无效动作处理（保持当前状态）
            pass

        # 计算奖励（传入归一化状态）
        reward = self.calculate_reward(self.state, action)

        # 终止条件（逻辑不变）
        if self.state["current_network"] == 1 and self.state["sinr_5g"] < 0:
            terminated = True
            reward -= 0.01
        if (action == 0 or action == 1):  # 切换动作处理
            if action == 0 and self.state["sinr_5g"] < 0.3333:
                reward += 0.05
            if action == 1 and self.state["sinr_5g"] > 0.667:
                reward += 0.05

        observation = np.array([self.state[param] for param in self.state_params], dtype=np.float32)
        info = {f"{param}_change": self.state[param] - old_state.get(param, 0) for param in self.state_params}
        info.update(self.state)

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        for param_name, low, high in self.state_params_config:
            base_value = self.base_values[param_name]
            if param_name == "current_network":
                self.state[param_name] = np.random.choice([0, 1])
                self.state[param_name]=1
            else:
                scale = 0.1
                noise = np.random.uniform(-scale, scale) * base_value
                original_value = base_value + noise
                original_value = np.clip(original_value, low, high)
                # 归一化处理
                min_val = self.state_mins[self.state_params.index(param_name)]
                max_val = self.state_maxs[self.state_params.index(param_name)]
                self.state[param_name] = (original_value - min_val) / (max_val - min_val+1e-9)

        # 特殊处理比特率（需重新计算）
        # 自组网比特率
        snr_original = self.base_values["snr_ad_hoc"]
        bw_original = self.base_values["bandwidth_ad_hoc"]
        bitrate_ad_hoc_original = self.calculate_bitrate(snr=snr_original, bandwidth=bw_original)
        min_bitrate = self.state_mins[self.state_params.index("bitrate_ad_hoc")]
        max_bitrate = self.state_maxs[self.state_params.index("bitrate_ad_hoc")]
        self.state["bitrate_ad_hoc"] = (bitrate_ad_hoc_original - min_bitrate) / (max_bitrate - min_bitrate)

        # 5G比特率
        sinr_original = self.base_values["sinr_5g"]
        bw_original = self.base_values["bandwidth_5g"]
        bitrate_5g_original = self.calculate_bitrate(snr=sinr_original, bandwidth=bw_original)
        min_bitrate = self.state_mins[self.state_params.index("bitrate_5g")]
        max_bitrate = self.state_maxs[self.state_params.index("bitrate_5g")]
        self.state["bitrate_5g"] = (bitrate_5g_original - min_bitrate) / (max_bitrate - min_bitrate)

        observation = np.array([self.state[param] for param in self.state_params], dtype=np.float32)
        return (
            observation,
            {"message": "Environment reset with fluctuating initial state"}
        )
    def render(self, mode="human"):
        """
        渲染环境。

        参数:
        mode (str): 渲染模式，支持 "human" 和 "rgb_array"。

        返回:
        np.ndarray or None: 如果模式为 "rgb_array"，返回一个空的 RGB 数组；如果模式为 "human"，打印环境信息并返回 None。
        """
        # 实现基础的人类可读渲染
        if mode == "human":
            status_str = "Network Status | "
            for param in self.state_params:
                if param == "current_network":
                    network_name = "自组网" if self.state[param] == 0 else "5G"
                    status_str += f"{param}: {network_name} | "
                else:
                    status_str += f"{param}: {self.state[param]:.1f} | "
            print(status_str.rstrip(" | "))
            return None
        # 支持 rgb_array 模式返回空数组（符合 API 规范）
        elif mode == "rgb_array":
            return np.empty((64, 64, 3), dtype=np.uint8)
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")

    def close(self):
        """
        关闭环境并清理资源。

        这里没有需要清理的资源，所以方法为空。
        """
        # 清理资源（如果需要）
        pass




