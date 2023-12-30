import numpy as np
import matplotlib.pyplot as plt
import cv2
def kalman(prev_x, prev_P,aims,re_x,re_y):
    # initial parameters - same as before
    dt = 1.0
    A = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    R = 10 * np.eye(2)
    Q = np.eye(4)
    B = 0
    u = 0
    I = np.eye(4)

    x = prev_x
    P = prev_P
    list=[]
    # measurement update
    _,x_center,y_center,_,_=aims
    x_center,y_center=re_x*float(x_center),re_y*float(y_center)
    list=[x_center,y_center]
    Z = np.array([list])
    Y = Z.T - H.dot(x)
    S = H.dot(P).dot(H.T) + R
    K = P.dot(H.T).dot(np.linalg.inv(S))
    x = x + K.dot(Y)
    P = (I - K.dot(H)).dot(P)

    # prediction
    x = A.dot(x) + B * u
    P = A.dot(P).dot(A.T) + Q

    return x, P

# def kalman(prev_x, prev_P, aims, re_x, re_y, process_noise=1.0, measurement_noise=10.0, alpha=0.1):
#
#     dt = 1.0
#     # 状态转移矩阵
#     A = np.array([[1, dt, 0, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, dt],
#                   [0, 0, 0, 1]])
#
#     # 观测矩阵
#     H = np.array([[1, 0, 0, 0],
#                   [0, 0, 1, 0]])
#
#     # 过程噪声协方差矩阵
#     Q = process_noise * np.eye(4)
#
#     # 测量噪声协方差矩阵
#     R = measurement_noise * np.eye(2)
#
#     # 控制输入矩阵
#     B = 0
#
#     # 控制输入
#     u = 0
#
#     # 单位矩阵
#     I = np.eye(4)
#
#     x = prev_x
#     P = prev_P
#
#     # 测量更新
#     _, x_center, y_center, _, _ = aims
#     x_center, y_center = re_x * float(x_center), re_y * float(y_center)
#     measurement = np.array([x_center, y_center]).reshape(2, 1)
#     Y = measurement - H.dot(x)
#     S = H.dot(P).dot(H.T) + R
#     K = P.dot(H.T).dot(np.linalg.inv(S))
#     x = x + K.dot(Y)
#     P = (I - K.dot(H)).dot(P)
#
#     # 自适应噪声调整
#     residual = Y.T.dot(Y)
#     R = (1 - alpha) * R + alpha * residual * np.eye(2)
#
#     # 预测
#     x = A.dot(x) + B * u
#     P = A.dot(P).dot(A.T) + Q
#
#     return x, P, Q, R