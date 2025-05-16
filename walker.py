import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class RandomWalker:
    def __init__(self, num_of_points_measured=100, speed=1.0, time_interval=1.0):
        self.num_of_points_measured = num_of_points_measured
        self.speed = speed
        self.time_interval = time_interval
        self.MS_coordinate = np.zeros((num_of_points_measured, 2))  # 存储坐标轨迹

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
        line, = ax.plot([], [], 'b-', lw=2, label='移动轨迹')
        point, = ax.plot([], [], 'ro', markersize=8, label='当前位置')
        ax.plot(ini_coordinate[0], ini_coordinate[1], 'go', markersize=8, label='起点')

        def animate(i):
            """动画更新函数"""
            x = self.MS_coordinate[:i + 1, 0].tolist()  # 所有已走过的x坐标
            y = self.MS_coordinate[:i + 1, 1].tolist()  # 所有已走过的y坐标

            line.set_data(x, y)  # 更新轨迹线

            # 方法1：将当前位置作为单点序列
            point.set_data([x[-1]], [y[-1]])

            # 方法2：分别设置x和y数据（等效）
            # point.set_xdata([x[-1]])
            # point.set_ydata([y[-1]])

            return line, point

        # 创建动画
        ani = animation.FuncAnimation(
            fig, animate, frames=self.num_of_points_measured, interval=100, blit=True
        )

        # 添加图例和显示
        ax.legend()
        plt.tight_layout()
        plt.show()
# 示例运行
if __name__ == "__main__":
    walker = RandomWalker(num_of_points_measured=1000, speed=10.0, time_interval=1)
    walker.new_walk()