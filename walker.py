import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


class RandomWalker:
    def __init__(self, num_of_points_measured=6561, speed=1.0, time_interval=1.0):
        self.num_of_points_measured = num_of_points_measured
        self.speed = speed
        self.time_interval = time_interval
        self.MS_coordinate = np.zeros((num_of_points_measured, 2))  # 存储移动轨迹坐标

        # 新增：基站坐标设置
        self.num_of_eNBs = 9  # 9个基站
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

        # 动画控制变量
        self.paused = False
        self.ani = None

    def new_walk(self):
        """生成随机游走路径并可视化，包含基站位置"""
        ini_coordinate = np.array([-199.9, 199.9])
        x, y = 0, 0
        unit = self.speed * self.time_interval
        direction = 1  # 1为向右，-1为向左

        # 生成坐标轨迹
        for t in range(self.num_of_points_measured):
            self.MS_coordinate[t] = ini_coordinate + np.array([x, y])

            # 每40步改变方向并换行
            if t % 40 == 39:
                direction *= -1
                y -= unit  # 向下移动一行
            else:
                x += direction * unit  # 水平移动

        # 调用可视化函数
        self.visualize_path()

    def visualize_path(self):
        """可视化移动轨迹和基站位置"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-250, 250)
        ax.set_ylim(-250, 250)
        ax.set_xlabel('X 坐标 (米)')
        ax.set_ylabel('Y 坐标 (米)')
        ax.set_title('移动轨迹与基站位置可视化')
        ax.grid(True, linestyle='--', alpha=0.5)

        # 绘制所有基站位置
        for i in range(self.num_of_eNBs):
            ax.plot(self.eNB_coordinate[i, 0], self.eNB_coordinate[i, 1], 'gs',
                    markersize=10, label=f'基站{i}' if i == 0 else "")

        # 添加基站图例
        ax.legend(loc='upper right', title='基站位置')

        # 保存初始坐标
        ini_coordinate = self.MS_coordinate[0]

        # 初始化轨迹和当前位置标记
        line, = ax.plot([], [], 'r-', lw=1.5, label='移动轨迹')
        point, = ax.plot([], [], 'ro', markersize=8, label='当前位置')
        start_point, = ax.plot(ini_coordinate[0], ini_coordinate[1], 'go',
                               markersize=8, label='起点')
        end_point, = ax.plot([], [], 'yo', markersize=8, label='终点')

        # 显示控制提示和距离信息
        control_text = ax.text(0.02, 0.95, "点击图表暂停/继续", transform=ax.transAxes,
                               fontsize=10, bbox=dict(facecolor='w', alpha=0.5))
        distance_text = ax.text(0.02, 0.90, "", transform=ax.transAxes,
                                fontsize=9, bbox=dict(facecolor='w', alpha=0.5))

        def calculate_distance(point1, point2):
            """计算两点间距离"""
            return np.sqrt(np.sum((point1 - point2) ** 2))

        def animate(i):
            """动画更新函数"""
            if not self.paused:
                x = self.MS_coordinate[:i + 1, 0].tolist()
                y = self.MS_coordinate[:i + 1, 1].tolist()
                line.set_data(x, y)
                point.set_data([x[-1]], [y[-1]])

                # 计算当前位置到各基站的距离
                distances = [calculate_distance(self.MS_coordinate[i], self.eNB_coordinate[j])
                             for j in range(self.num_of_eNBs)]
                nearest_eNB = np.argmin(distances)
                min_distance = distances[nearest_eNB]

                # 更新距离显示
                distance_text.set_text(f"最近基站: {nearest_eNB}, 距离: {min_distance:.1f}米")

                # 最后一帧显示终点
                if i == self.num_of_points_measured - 1:
                    end_point.set_data([x[-1]], [y[-1]])

            return line, point, start_point, end_point, distance_text

        def toggle_pause(event):
            """点击事件处理：切换暂停状态"""
            self.paused = not self.paused
            if self.paused:
                self.ani.event_source.stop()
                control_text.set_text("点击图表继续")
            else:
                self.ani.event_source.start()
                control_text.set_text("点击图表暂停")
            fig.canvas.draw_idle()

        # 连接点击事件
        fig.canvas.mpl_connect('button_press_event', toggle_pause)

        # 创建动画
        self.ani = animation.FuncAnimation(
            fig, animate, frames=self.num_of_points_measured, interval=1, blit=False
        )

        plt.tight_layout()
        plt.show()


# 示例运行
if __name__ == "__main__":
    walker = RandomWalker(num_of_points_measured=6561, speed=10.0, time_interval=1)
    walker.new_walk()