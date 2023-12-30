import cv2
import numpy as np
import time
from grabscreen import grab_screen ,grab_screen_mss#截屏
while True:
    start_time_image=time.time()
    img0=grab_screen_mss(region=(30, 100, 310, 360))
# 读取两张图像
img1 = cv2.imread('image.png')  # 大地图
img2 = cv2.imread('image2.png')  # 小地图

# 将图像大小进行缩放
img1 = cv2.resize(img1, (1000, 1000))
img2 = cv2.resize(img2, (100, 100))

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 找到图像中的关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)  # 大地图
kp2, des2 = sift.detectAndCompute(img2, None)  # 小地图

# 初始化暴力匹配器
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 应用比率测试
good = []
for m, n in matches:
    if m.distance < 0.55 * n.distance:
        good.append(m)

# 提取匹配点
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# 计算单应性矩阵
M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)  # 注意这里是小地图到大地图的转换

# 当前位置在小地图上的坐标，假设为小地图中心点
current_position_small_map = np.array([[50, 50, 1]]).T  # 3x1坐标向量，因为小地图尺寸为100x100
# 使用单应性矩阵将当前位置的坐标转换到大地图上
current_position_large_map = np.dot(M, current_position_small_map)
# 归一化处理，将齐次坐标转换为2D坐标
current_position_large_map = (current_position_large_map / current_position_large_map[2])[:2]

# 添加箭头以表示朝向
arrow_length = 30
center = (int(current_position_large_map[0]), int(current_position_large_map[1]))
direction_offset = (0, -arrow_length)
end_point = (center[0] + direction_offset[0], center[1] + direction_offset[1])
img_with_arrow = cv2.arrowedLine(img1, center, end_point, (0, 255, 0), 2)

# 在大地图上标记当前位置
img_with_marker = cv2.circle(img_with_arrow, center, 3, (0, 0, 255), -1)

# 显示带有标记和箭头的图像
print(current_position_large_map)
cv2.imshow('Map with Current Position and Orientation', img_with_marker)
cv2.waitKey(0)
cv2.destroyAllWindows()