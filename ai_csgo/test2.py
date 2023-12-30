import numpy as np
import matplotlib.pyplot as plt

#%matplotlib auto #Jupyter notebook的弹出窗绘图语句，在Pycharm和Spyder中国必须删除该句

for i in range(10):
    plt.figure() #绘制一个figure，标号基于前一个figure以自然数命名
    plt.figure(i) #以指定数据作为标号绘制一个figure
    plt.plot(np.random.randn(10,10)) #在当前figure中绘图，如果没有figure则自己新建一个再绘图
    plt.pause(0.02) #暂停时间
    plt.cla() #将当前figure中绘图区的内容清除
    plt.close() #将当前figure关闭
