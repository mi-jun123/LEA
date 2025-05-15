import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import math
import numpy as np
import matplotlib.pyplot as plt


class NetworkSwitchEnv(gym.Env):
    def __init__(self, state_params_config=None):


        self.num_of_eNBs = 9
        self.num_of_points_measured=6561
        eNB_coordinate = np.zeros([num_of_eNBs, 2])

        eNB_coordinate[0, :] = [0, 0]
        eNB_coordinate[1, :] = [0, 200]
        eNB_coordinate[2, :] = [0, -200]
        eNB_coordinate[3, :] = [200, 0]
        eNB_coordinate[4, :] = [-200, 0]
        eNB_coordinate[5, :] = [-200, 200]
        eNB_coordinate[6, :] = [200, 200]
        eNB_coordinate[7, :] = [200, -200]
        eNB_coordinate[8, :] = [-200, -200]
        self.MS_coordinate = np.zeros([self.num_of_points_measured, 2])#间距5m
        # 绘制基站分布
        """
        plt.scatter(eNB_coordinate[:, 0], eNB_coordinate[:, 1], marker='o', label='Base Stations')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Base Station Distribution')
        plt.grid(True)
        plt.legend()
        plt.show()
        """
        # 新增参数配置（单位：snr/sinr(dB), 带宽(Hz)）
        if state_params_config is None:
            state_params_config = []
            # 自组网电台状态（6个）
            adhoc_params = [
                ("rss_adhoc", -120.0, -10.0),
                ("bitrate_adhoc", 0.0, 1e9),
                ("rtt_adhoc", 0.0, 200.0),
                ("cost_adhoc", 0.0, 1.0),
                ("speed_adhoc", 0.0, 50.0),
                ("height_adhoc", 0.0, 1000.0)  # 自组网可能有高度（如无人机）
            ]
            state_params_config.extend(adhoc_params)

            # 8个5G基站状态（每个6个，共48个）
            for i in range(1, self.num_5g + 1):
                base_params = [
                    (f"rss_5g_{i}", -120.0, -10.0),
                    (f"bitrate_5g_{i}", 0.0, 1e9),
                    (f"rtt_5g_{i}", 0.0, 200.0),
                    (f"cost_5g_{i}", 0.0, 1.0),
                    (f"speed_5g_{i}", 0.0, 50.0),
                    (f"height_5g_{i}", 0.0, 1000.0)  # 5G基站高度固定为0（地面站）
                ]
                state_params_config.extend(base_params)

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

            # 计算归一化参数
            self.state_mins = self.state_lows
            self.state_maxs = self.state_highs
            # 确保在调用 reset 之前定义 self.state_params_config
            self.state_params_config = state_params_config
            self.util_boundaries = self._init_util_boundaries()
            self.reset()





    def calculate_bitrate(self, snr, bandwidth,distance=0, frequency=0, transmit_power=20, noise_power=-95,
                          efficiency=0.85):
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

       #path_loss = calculate_path_loss(distance, frequency)
        #receive_power = transmit_power - path_loss
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
            },
            "Us": {
                "min": 0,  #
                "max": 50.0  # m/s
            },
            "Uh": {
                "min": 0,  # m
                "max": 1000.0  # m
            }
        }

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

        omega_b = 0.44548085
        omega_d = 0.04034934
        omega_Rs = 0.28216224
        omega_c = 0.05674012
        omega_s=0.09580386
        omega_h=0.07946359
        G_value = self.calculate_G(omega_b, omega_d, omega_Rs, omega_c,omega_h, omega_s,Ub, Ud, U_Rs, Uc,Us,Uh)

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

        self.state["current_network"] = action

        # 计算奖励（传入归一化状态）
        reward = self.calculate_reward(self.state, action)


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
        try:
            # 调用随机游走计算函数
            self.calculate_all_rand_walk()
            rss=ExternalParameterGenerator.get_received_power(tx_power, h_lap, distance[0][0])
            # 提取第一个基站与第一个测量点的相关信息
            sinr = self.SINR[0][0]
            distance = self.distance[0][0]

            # 构建初始状态向量
            s = [rss, sinr, distance, x, y]

            return s
        except Exception as e:
            print(f"An error occurred while calculating the first state: {e}")
            return None
    def base_station_random_setting(self):
        for i in range(self.num_of_eNBs):
            self.tx_RSS[i] = np.random.rand() * 5 + 20
    def new_walk(self):
        a =10

        ini_coordinate = [-799.9, 799.9]
        unit = (a + 1) * 1600 / self.num_of_points_measured

        x = int(self.num_of_points_measured / (a + 1) / (a - 1))
        y = int(self.num_of_points_measured / (a + 1))
        for t in range(y):
            self.MS_coordinate[t] = np.array(ini_coordinate) + np.array([0, - unit * t])
        for cnt in range(a - 1):
            for b in range(y, x + y):
                self.MS_coordinate[b + cnt * (x + y)] = self.MS_coordinate[b + cnt * (x + y) - 1] + np.array([unit, 0])
            for b in range(x + y, x + 2 * y):
                if cnt % 2 == 0:
                    self.MS_coordinate[b + cnt * (x + y)] = self.MS_coordinate[b + cnt * (x + y) - 1] + np.array(
                        [0, unit])
                else:
                    self.MS_coordinate[b + cnt * (x + y)] = self.MS_coordinate[b + cnt * (x + y) - 1] + np.array(
                        [0, - unit])

        for t in range((a - 1) * x + a * y, self.num_of_points_measured):
            self.MS_coordinate[t] = self.MS_coordinate[t - 1] + np.array([0, -unit])


    def calculate_distance(self):
        for x in range(self.num_of_eNBs):
            for i in range(self.num_of_points_measured):
                self.distance[x][i] = math.sqrt(pow(self.MS_coordinate[i][0] - self.eNB_coordinate[x][0], 2) + pow(
                    self.MS_coordinate[i][1] - self.eNB_coordinate[x][1], 2))

    def calculate_all_rand_walk(self):

        self.base_station_random_setting()
        self.new_walk()

    def close(self):
        """
        关闭环境并清理资源。

        这里没有需要清理的资源，所以方法为空。
        """
        # 清理资源（如果需要）
        pass




