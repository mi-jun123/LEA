import numpy as np
import matplotlib.pyplot as plt

# 假设 self.num_of_eNBs = 16
num_of_eNBs = 9
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

# 假设已经按照之前的建议生成了测量点坐标 self.MS_coordinate
# 这里只是示例，实际需要根据你的具体生成方法来
MS_coordinate = np.zeros([6561, 2])  # 间距5m

# 绘制基站分布
plt.scatter(eNB_coordinate[:, 0], eNB_coordinate[:, 1], marker='o', label='Base Stations')

# 绘制测量点
plt.scatter(MS_coordinate[:, 0], MS_coordinate[:, 1], marker='x', label='Measurement Points')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Base Station and Measurement Points Distribution')
plt.grid(True)
plt.legend()
plt.show()