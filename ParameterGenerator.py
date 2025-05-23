import numpy as np
import math


class ExternalParameterGenerator:
    def __init__(self, freq=35e8, max_pathloss=120, seed=32):
        # 设置随机数种子
        self.seed = seed
        np.random.seed(self.seed)

        self.num_of_eNBs = 9
        self.num_of_points_measured = 6561
        self.eNB_coordinate = np.zeros([self.num_of_eNBs, 2])
        self.eNB_coordinate[0, :] = [0, 0]
        self.eNB_coordinate[1, :] = [0, 200]
        self.eNB_coordinate[2, :] = [0, -200]
        self.eNB_coordinate[3, :] = [200, 0]
        self.eNB_coordinate[4, :] = [-200, 0]
        self.eNB_coordinate[5, :] = [-200, 200]
        self.eNB_coordinate[6, :] = [200, 200]
        self.eNB_coordinate[7, :] = [200, -200]
        self.eNB_coordinate[8, :] = [-200, -200]
        self.MS_coordinate = np.zeros([self.num_of_points_measured, 2])

        # 环境参数（ITU定义的城市参数）
        self.alpha = 0.09  # 建成区比例
        self.beta = 500  # 建筑密度 (buildings/km²)
        self.gamma = 20  # 建筑高度尺度参数 (m)

        # 系统参数
        self.freq = freq  # 频率 (Hz)
        self.freq0=16e8
        self.freq1=35e8
        self.PL_max = max_pathloss  # 最大允许路径损耗 (dB)

        # Sigmoid曲线拟合参数（根据表I/表II预计算）
        self.a, self.b = self.calculate_sigmoid_params()

        # 传播组额外损耗（根据论文典型值）
        self.eta_LoS = 0.1  # LoS额外损耗 (dB)
        self.eta_NLoS = 21.0  # NLoS额外损耗 (dB)


        self.distance = np.zeros([self.num_of_eNBs, self.num_of_points_measured])
        self.ideal_RSS = np.zeros([self.num_of_eNBs, self.num_of_points_measured])
        self.SINR = np.zeros((self.num_of_eNBs, self.num_of_points_measured))
        self.RTT = np.zeros([self.num_of_eNBs, self.num_of_points_measured])
        self.SNR = np.zeros([self.num_of_eNBs, self.num_of_points_measured])

    def new_walk(self, speed=10.0, time_interval=1.0):
        # 随机游走路径生成（如果需要随机性，可以在这里设置子种子）
        ini_coordinate = np.array([-199.9, 199.9])
        x, y = 0, 0
        unit = speed * time_interval  # 每步移动距离 = 速度 × 时间间隔
        num_points = self.num_of_points_measured
        direction = 1
        for t in range(num_points):
            self.MS_coordinate[t] = ini_coordinate + np.array([x, y])
            if t % 40 == 39:  # 每 30 步改变方向
                direction *= -1
                y -= unit  # 换行
            else:
                x += direction * unit  # 水平移动

    def calculate_distance(self):
        for x in range(self.num_of_eNBs):
            for i in range(self.num_of_points_measured):
                self.distance[x][i] = math.sqrt(
                    pow(self.MS_coordinate[i][0] - self.eNB_coordinate[x][0], 2) +
                    pow(self.MS_coordinate[i][1] - self.eNB_coordinate[x][1], 2))

    def get_nearest_eNB(self, current_point):
        """获取距离当前测量点最近的基站编号和距离"""
        distances = self.distance[:, current_point]
        nearest_eNB_index = np.argmin(distances)
        nearest_distance = distances[nearest_eNB_index]
        #print(f"nearest_eNB_index{nearest_eNB_index}")
        return nearest_eNB_index
    def calculate_all_rand_walk(self):
        self.new_walk()
        self.calculate_distance()
    def calculate_sigmoid_params(self):
        """根据ITU参数拟合Sigmoid曲线参数a和b"""
        alpha_beta = self.alpha * self.beta
        gamma = self.gamma

        # 计算a的多项式拟合（表I系数）
        a = 0
        for i in range(4):
            for j in range(4 - i):
                coeff = [[9.34e-01, 2.30e-01, -20.25e-03, 1.86e-05],
                         [1.97e-02, 2.44e-03, 6.58e-06, 0],
                         [-1.24e-04, -3.34e-06, 0, 0],
                         [2.73e-07, 0, 0, 0]][j][i]
                a += coeff * (alpha_beta ** i) * (gamma ** j)

        # 计算b的多项式拟合（表II系数）
        b = 0
        for i in range(4):
            for j in range(4 - i):
                coeff = [[1.17e+00, -7.56e-02, 1.98e-03, -1.78e-05],
                         [-5.79e-03, 1.81e-04, -1.65e-06, 0],
                         [1.73e-05, -2.02e-07, 0, 0],
                         [-20.0e-08, 0, 0, 0]][j][i]
                b += coeff * (alpha_beta ** i) * (gamma ** j)

        return a, b

    def calculate_los_probability(self, elevation_angle_deg):
        """计算视距概率（Sigmoid近似）"""
        theta = math.radians(elevation_angle_deg)
        return 1 / (1 + math.exp(-self.b * (theta - math.radians(self.a))))

    def calculate_pathloss(self,a, h_lap, ground_distance, h):
        """计算平均路径损耗（结合LoS/NLoS概率）"""
        d = math.sqrt((h_lap - h) ** 2 + ground_distance ** 2)  # 斜距
        #print(f"d{d}")
        #print(f"ground_distance{ground_distance}")

        if a==0:
            freq=self.freq0
        else:
            freq=self.freq1
        #print(f"freq{freq}")
        theta = math.atan2(h_lap - h, ground_distance)  # 仰角（弧度）
        pl_LoS = 20 * math.log10(d) + 20 * math.log10(freq) + 20 * math.log10(4 * math.pi / 3e8) + self.eta_LoS
        pl_NLoS = 20 * math.log10(d) + 20 * math.log10(freq) + 20 * math.log10(4 * math.pi / 3e8) + self.eta_NLoS
        #print(f"pl_LoS{pl_LoS}")
        #print(f"pl_NLoS{pl_NLoS}")
        p_los = self.calculate_los_probability(math.degrees(theta))
        average_pl = p_los * pl_LoS + (1 - p_los) * pl_NLoS

        return average_pl

    def get_received_power(self, a,tx_power, h_lap, ground_distance, h):
        """计算接收信号功率（考虑阴影衰落和多径效应）"""
        # 设置随机数生成器的状态，确保每次运行生成相同的随机数序列
        rng_state = np.random.get_state()
        np.random.seed(self.seed)

        pl = self.calculate_pathloss(a,h_lap, ground_distance, h)

        shadow_fading = np.random.normal(0, 8)  # 阴影衰落（标准差8dB，论文典型值）
        multipath_fading = np.random.rayleigh()  # 多径衰落（瑞利分布）

        # 恢复随机数生成器状态，不影响其他随机过程
        np.random.set_state(rng_state)

        # 总损耗 = 平均路径损耗 + 阴影衰落 + 多径衰落（dB转换）
        total_loss = pl + shadow_fading + 10 * math.log10(multipath_fading)
        received_power = tx_power - total_loss

        return received_power

    def get_RSS(self, t, h_lap=50):
        """
        获取所有基站在测量点t的RSS值

        参数:
        t: 测量点索引
        h_lap: 基站高度，默认为50米

        返回:
        包含所有基站RSS值的列表
        """
        RSS_list = []  # 用于存储所有基站的RSS值

        for i in range(self.num_of_eNBs):
            # 为每个基站使用不同的子种子，确保结果可复现
            np.random.seed(self.seed)
            #print(f"i:{i}")
            #print(f"t:{t}")
            rss = self.get_received_power(i,27, h_lap, self.distance[i][t], 0)
            RSS_list.append(rss)
            #print(f"rss{RSS_list}")
            self.ideal_RSS[i,t]=rss
        return RSS_list  # 返回所有基站的RSS列表输出一维数组

    def calculate_SNR(self, t):
        """
        计算t时刻所有基站到用户的SNR（信噪比）

        参数:
        t: 时间索引或测量点索引

        返回:
        SNR矩阵 [基站数, 1]，其中基站0的SNR为NaN
        """
        # 噪声基底 (dBm)
        noise_floor = -104

        # 对所有基站（排除基站0）计算SNR
        for i in range(self.num_of_eNBs):
            # 使用固定种子确保可复现
            np.random.seed(self.seed + t * self.num_of_eNBs + i)

            # 获取信号功率
            signal_power = self.ideal_RSS[i, t]

            # 计算热噪声功率 (dBm)
            thermal_noise = noise_floor

            # 计算额外的随机噪声（dB）
            random_noise = np.random.normal(0, 2)  # 正态分布，标准差2dB

            # 计算SNR (dB)
            snr = signal_power - (thermal_noise + random_noise)

            # 存储结果
            self.SNR[i, t] = snr
            if i==0:
                self.SNR[i, t]= self.SNR[i, t]-8


        return self.SNR[:,t]

    def rttc(self, t):
        # 使用固定种子确保RTT生成可复现
        np.random.seed(self.seed + t)
        rtt = np.random.rand(self.num_of_eNBs) * 5 + 60
        self.RTT[:, t] = rtt
        return self.RTT[:,t]

    def get_t_moment_params(self, t,h):
        """
        获取t时刻的SNR、RSS和RTT

        参数:
        t: 时间索引或测量点索引

        返回:
        包含t时刻SNR、RSS和RTT的字典
        """
        rss = self.get_RSS(t,h)
        snr = self.calculate_SNR(t)
        rtt = self.rttc(t)
        snr_prob = 0.05
        # 确保RSS是NumPy数组后再进行运算
        rss = np.array(rss)  # 将列表转换为NumPy数组
        np.random.seed(self.seed+t+h)
        #print(f"已设置随机种子: {self.seed}")
        if np.random.random() < snr_prob:
            snr[1:] = snr[1:] - 10
            #print("SNR数组已处理（保持第一位不变，其余减10）")
            rss[1:] = rss[1:] - 20
            #print("RSS数组已处理（保持第一位不变，其余减20）")


        return {
            'SNR': snr,
            'RSS': rss,
            'RTT': rtt
        }

