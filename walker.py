import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 设置中文字体支持（根据系统情况调整）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 确保负号正确显示


class RandomWalker:
    def __init__(self, num_of_points_measured=100, speed=1.0, time_interval=1.0):
        self.num_of_points_measured = num_of_points_measured
        self.speed = speed
        self.time_interval = time_interval
        self.MS_coordinate = np.zeros((num_of_points_measured, 2))  # 存储坐标轨迹

        # 动画控制变量
        self.paused = False  # 初始状态：未暂停
        self.ani = None  # 存储动画对象

    def new_walk(self):
        """生成随机游走路径并可视化"""
        ini_coordinate = np.array([-199.9, 199.9])
        x, y = 0, 0
        unit = self.speed * self.time_interval  # 每步移动距离
        direction = 1  # 初始方向：1为向右，-1为向左

        # 生成坐标轨迹
        for t in range(self.num_of_points_measured):
            self.MS_coordinate[t] = ini_coordinate + np.array([x, y])

            # 每30步改变方向并换行
            if t % 30 == 29:
                direction *= -1
                y -= unit  # 向下移动一行
            else:
                x += direction * unit  # 水平移动

        # 调用可视化函数
        self.visualize_path()

    def visualize_path(self):
        """可视化移动轨迹"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_title('随机游走路径可视化')
        ax.grid(True, linestyle='--', alpha=0.7)

        # 保存初始坐标供内部函数使用
        ini_coordinate = self.MS_coordinate[0]

        # 初始化轨迹和当前位置标记
        line, = ax.plot([], [], 'b-', lw=2, label='轨迹')
        point, = ax.plot([], [], 'ro', markersize=8, label='当前位置')
        start_point, = ax.plot(ini_coordinate[0], ini_coordinate[1], 'go', markersize=8, label='起点')
        end_point, = ax.plot([], [], 'yo', markersize=8, label='终点')

        # 设置图例
        ax.legend()

        # 显示控制提示
        control_text = ax.text(0.02, 0.02, "点击图表暂停/继续", transform=ax.transAxes,
                               fontsize=10, bbox=dict(facecolor='w', alpha=0.5))

        def animate(i):
            """动画更新函数"""
            if not self.paused:  # 仅在未暂停时更新
                x = self.MS_coordinate[:i + 1, 0].tolist()
                y = self.MS_coordinate[:i + 1, 1].tolist()
                line.set_data(x, y)
                point.set_data([x[-1]], [y[-1]])

                # 在动画的最后一帧显示终点
                if i == self.num_of_points_measured - 1:
                    end_point.set_data([x[-1]], [y[-1]])

            return line, point, start_point, end_point

        def toggle_pause(event):
            """点击事件处理函数：切换暂停/继续状态"""
            self.paused = not self.paused
            if self.paused:
                self.ani.event_source.stop()
                control_text.set_text("点击图表继续")
            else:
                self.ani.event_source.start()
                control_text.set_text("点击图表暂停")
            fig.canvas.draw_idle()  # 更新画布

        # 连接点击事件
        fig.canvas.mpl_connect('button_press_event', toggle_pause)

        # 创建动画并保存到 self.ani
        self.ani = animation.FuncAnimation(
            fig, animate, frames=self.num_of_points_measured, interval=100, blit=False
        )

        plt.tight_layout()
        plt.show()


# 示例运行
if __name__ == "__main__":
    walker = RandomWalker(num_of_points_measured=200, speed=10.0, time_interval=1)
    walker.new_walk()