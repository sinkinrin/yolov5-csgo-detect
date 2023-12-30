import matplotlib.pyplot as plt

def radar_plt(x_center,dist):
    plt.xlabel('x')
    plt.ylabel('distance')
    plt.plot(x_center,dist)
    plt.pause(0.02)
    plt.cla()

