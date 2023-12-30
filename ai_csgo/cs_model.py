from threading import Thread #这里当时打算多线程？ 我忘了是不是我写的了
import torch
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import check_img_size,Profile,non_max_suppression,scale_boxes,xyxy2xywh
from grabscreen import grab_screen ,grab_screen_mss#截屏
from utils.augmentations import letterbox
from mouse_control import lock
from models.experimental import attempt_load
import cv2
import win32gui
import win32con #win32 调用的windows的接口
import numpy as np
import pynput #控制键盘鼠标的输入
import time
from kalmantrace import kalman
from filterpy.kalman import KalmanFilter
from utils import *
import matplotlib.pyplot as plt
#utils models都是yolo的 别管

x,y=(2560,1440) #屏幕大小
re_x,re_y=(2560,1440)
# 初始化图和坐标轴


init_x = np.array([[0], [0], [0], [0]])  # initial position and velocity
init_P = np.array([[1000, 0, 0, 0],
                   [0, 1000, 0, 0],
                   [0, 0, 1000, 0],
                   [0, 0, 0, 1000]])
init_Q = 1.0
init_R = 10.0

trajectory = []
for i in range(6):
    trajectory.append((0,0))
det_3 = [0,10,10,0,0]

show_win=True #是否展示csgo-detect屏幕

device = "cuda:0" if torch.cuda.is_available() else 'cpu' #加速方式 yolo超参数之一
half = device != 'cpu' #yolo超参数之一
kk = 0
weights = r'E:\yolov5-master\runs\train\exp6\weights\best.pt' #yolo超参数之一 训练权重
imgsz=640 #yolo超参数之一 预处理后图片大小
data='data/mydata.yaml' #yolo超参数之一 label
conf_thres=0.35 #yolo超参数之一
iou_thres=0.05
flag=0
mouse=pynput.mouse.Controller()
keyboard=pynput.keyboard.Controller()
stride=32
pt=True
names={0: 'ct', 1: 'ct_head', 2: 't', 3: 't_head'} #label

lock_mode=False #是否开锁
def on_move(x, y):
    pass

def on_click(x, y, button, pressed):
    global lock_mode
    if pressed and button == button.x2:
        lock_mode= not lock_mode
        print('lock_mode','on'if lock_mode else 'off')

def on_scroll(x, y, dx, dy):
    pass

def radar_plt(x_center,dist):
    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    area = np.pi * (15 * np.random.rand(N)) ** 2  # 0 to 15 point radii

    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()

# Collect events until released
#listener = pynput.mouse.Listener(
    #on_move=on_move,
    #on_click=on_click,
   # on_scroll=on_scroll)
#listener.start()
#这里鼠标监听已经ban了 要锁自己开
#我写点注释吧


while True:
    start_time_image=time.time()
    img0=grab_screen_mss(region=(0, 0, x, y))
    #img0=cv2.resize(img0,(re_x,re_y))
    # Load model
    if not flag:
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=half)
        #stride, names, pt = model.stride, model.names, model.pt
        #print(names)
        imgsz = check_img_size(imgsz, s=stride)
        flag=1
    '''
    if flag:
        device =  select_device(device)
        model = attempt_load(weights,device=device)
        imgsz = check_img_size(imgsz)
    '''
    #
    img = letterbox(img0, imgsz, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.


    if len(img.shape) == 3:
        img = img[None]  # img = img.unsqueeze(0)
    end_time_image=time.time()
    # print("get image time",(end_time_image-start_time_image)*1000,'ms')
    torch.cuda.synchronize()
    start_time_pred=time.time()
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    torch.cuda.synchronize()
    end_time_pred=time.time()
    # print("pred time is",(end_time_pred-start_time_pred)*1000,"ms")
    #print(pred)
    aims=[]
    # # Process predictions
    flag_3_modified = False #为什么不丢到最上面去


    for i, det in enumerate(pred):
        flag_3_modified = False
        s=''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            for *xyxy, conf, cls in reversed(det):

                #bbox(tag,x_Ceter,y_Center,x_widt   h,y_width)
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh)
                aim=('%g ' * len(line)).rstrip()% line
                print(aim)
                aim = aim.split(" ")
                aims.append(aim)

        if len(aims):
            #if(len(aims)>=2):
            if lock_mode:
                start_time_lock=time.time()
                lock(aims, mouse, re_x, re_y)
                end_time_lock=time.time()
                #print("lock time",(end_time_lock-start_time_lock)*1000,"ms")
            for i,det in enumerate(aims):
                tag,x_center,y_center,width,height=det#supposed to be 'aim'
                x_center,y_center=re_x*float(x_center),re_y*float(y_center)
                width,height=re_x*float(width),re_y*float(height)
                top_left=(int(x_center-width/2), int(y_center-height/2))
                bottom_right=(int(x_center+width/2), int(y_center+height/2))
                color=(0,255,0) #RGB
                cv2.rectangle(img0,top_left,bottom_right,color,4)
                if tag == '2':
                    dist = 6063 / height - 4.97
                    #print(dist)
                max_trajectory_length = 10  # 设置你希望的最大轨迹点数量
                if tag == '3': #kalman filter
                    flag_3_modified = True
                    init_x, init_P = kalman(init_x, init_P, det, re_x, re_y)  # optimize:free trajectory
                    #init_x, init_P, init_Q, init_R = kalman(init_x, init_P, det_3, re_x, re_y, process_noise=init_Q,measurement_noise=init_R, alpha=0.1)
                    if len(trajectory) >= max_trajectory_length:
                        trajectory = []  # 移除早期的轨迹点
                        for _ in range(6):
                            trajectory.append((0, 0))
                    trajectory.append((int(init_x[0, 0]), int(init_x[2, 0])))
                    #print(det)
                    det_3 = [0, init_x[0, 0] / re_x, init_x[2, 0] / re_y, 0, 0]
                    #print(det_3)
                    print(trajectory[-1])

                    for i in range(7):
                        cv2.circle(img0, trajectory[-(7 - i)], 3, (0, 0, 255), 3)

                    # 清空trajectory列表以释放内存
                    #trajectory.clear()
                    #cv2.circle(img0, (int(init_x[0, 0]), int(init_x[2, 0])),3, color, 3)
                    #print((init_x[0, 0], init_x[2, 0]))
        if flag_3_modified == False:
            init_x, init_P = kalman(init_x, init_P, det_3, re_x, re_y)  # optimize:free trajectory
            #init_x, init_P, init_Q, init_R = kalman(init_x, init_P, det_3, re_x, re_y, process_noise=init_Q,measurement_noise=init_R, alpha=0.1)
            trajectory.append((int(init_x[0, 0]), int(init_x[2, 0])))
            det_3 = [0,init_x[0, 0]/re_x,init_x[2, 0]/re_y,0,0]
            #print(det_3)
            #print(trajectory[-1])
            for i in range(7):
                cv2.circle(img0, trajectory[-(7 - i)], 3, (0, 0, 255), 3)

    if show_win:
        cv2.namedWindow('csgo-detect', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('csgo-detect', re_x // 4, re_y // 4)
        cv2.imshow('csgo-detect',img0)

        hwnd=win32gui.FindWindow(None,'csgo-detect')
        CVRECT=cv2.getWindowImageRect('csgo-detect')
        win32gui.SetWindowPos(hwnd,win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    if cv2.waitKey(1) & 0xFF==ord('q'): #input q to shutdown
        cv2.destroyAllWindows()
        break