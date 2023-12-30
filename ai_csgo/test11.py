from pynput.mouse import Button, Controller
import time
import pyautogui
from pynput.mouse import Controller
import win32gui
import win32api
import pynput #控制键盘鼠标的输入
from pynput import mouse
mouse = pynput.mouse.Controller()
from pynput import keyboard
kb = keyboard.Controller()
time.sleep(5)
# 创建一个键盘

# 按下 a 键
kb.press('w')
# 松开 a 键
time.sleep(4)
kb.release('w')
time.sleep(4)
# 屏幕左上角坐标为(0, 0) 右下角为(屏幕宽度, 屏幕高度)
print(f"当前鼠标位置: {mouse.position}")  # 当前鼠标位置: (881, 467)

# 目标位置
target_x, target_y = 2000, 720
#
# # 最终将鼠标移动到目标位置
mouse.position = (target_x, target_y)

# 给mouse.position赋值等于移动鼠标，这里相当于移动到(100, 100)的位置
# 如果坐标小于0，那么等于0。如果超出屏幕范围，那么等于最大范围
# mouse.position = (10, 500)  # 此方法等价于mouse.move(100, 100)
print(f"当前鼠标位置: {mouse.position}")  # 当前鼠标位置: (100, 100)


# 按下左键,同理Button.right是右键
mouse.press(Button.left)
# 松开左键
mouse.release(Button.left)
# 上面两行连在一起等于一次单击。如果上面两行紧接着再重复一次，那么整体会实现双击的效果
# 因为两次单击是连续执行的，没有等待时间。如果中间来一个time.sleep几秒，那么就变成两次单击了


# 当然鼠标点击我们有更合适的办法，使用click函数
# 该函数接收两个参数：点击鼠标的哪个键、以及点击次数
# 这里连续点击两次，等于双击
mouse.click(Button.right, 2)
