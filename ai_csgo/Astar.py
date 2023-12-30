import math
from PIL import Image
import numpy as np
import heapq
import matplotlib.pyplot as plt

# 加载图像
image = Image.open('binary_map.jpg')
# 转化为灰度图像
#image = image.convert('L')
# 转化为numpy数组
image_data = np.array(image)
# 由于原图是二值化的，我们假设所有非0值（比如255）为可通行，同时假设0为不可通行
binary_map = np.where(image_data == 0, 0, 1)

class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.obstacle_value = 0  # 黑色部分的值，根据您的图像可能会有所不同

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_y][new_x] != self.obstacle_value:
                    neighbors.append((new_x, new_y))
        return neighbors

    def distance_to_obstacle(self, node):
        x, y = node
        mask = np.ones((7, 7))  # 以当前节点为中心的7x7掩码
        mask_x, mask_y = mask.shape
        min_distance = float('inf')

        for i in range(-3, 4):
            for j in range(-3, 4):
                if 0 <= x + i < self.width and 0 <= y + j < self.height:
                    if self.grid[y + j][x + i] == self.obstacle_value and mask[j + 3][i + 3] == 1:
                        # 如果发现黑色部分，计算到它的距离
                        distance = max(abs(i), abs(j))
                        min_distance = min(min_distance, distance)

        return min_distance

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path

    def find_path(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))  # (f, node)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}

        while open_set:
            _, current = heapq.heappop(open_set)  # We don't need the f value, it's only used for sorting
            if current == end:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                if self.distance_to_obstacle(neighbor) >= 3:  # 检查距离限制
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found


# 示例用法
grid = [[1]*500 for _ in range(500)]  # Example grid, you should replace this with your actual grid
astar = AStar(grid)
astar = AStar(binary_map)  # 不再使用全白地图
start = (414, 112)
end = (94,421)
path = astar.find_path(start, end)
print(path)

count = 0
for position in path:
    if count % 10 == 0:
        print(position)
    count += 1

# 绘制地图
fig, ax = plt.subplots()
ax.imshow(binary_map, cmap='Greys', interpolation='nearest')
"""
# 绘制起点和终点
ax.plot(start[0], start[1], 'o', color='green', markersize=10, label='Start')
ax.plot(end[0], end[1], 'o', color='red', markersize=10, label='End')

# 绘制路径
if path:
    X, Y = zip(*path)
    ax.plot(X, Y, linestyle='-', color='blue', marker='o', markersize=4, label='Path')

# 显示图例
ax.legend()

# 显示地图
plt.show()
"""
# 绘制每十次移动的位置
count = 0
for position in path:
    if count % 30 == 0:
        ax.plot(position[0], position[1], 'o', color='blue', markersize=5)
    count += 1

# 绘制起点和终点
ax.plot(start[0], start[1], 'o', color='green', markersize=10, label='Start')
ax.plot(end[0], end[1], 'o', color='red', markersize=10, label='End')

# 显示地图
plt.show()