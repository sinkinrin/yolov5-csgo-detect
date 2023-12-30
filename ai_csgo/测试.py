import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
import mainwindow
import subprocess
import threading
import time


class MainWindow(QMainWindow, mainwindow.Ui_MainWindow):  # 多重继承QMainWindow和Ui_MainWindow
    def __init__(self):
        super(MainWindow, self).__init__()  # 先调用父类QMainWindow的初始化方法
        self.setupUi(self)  # 再调用setupUi方法
        self.pushButton.clicked.connect(self.run_script_1)
        self.pushButton_2.clicked.connect(self.run_script_2)
        self.pushButton_3.setEnabled(False)  # 初始时禁用停止按钮
        self.pushButton_3.clicked.connect(self.stop_script)
        self.process_list = []
        self.show()

    def run_script_in_thread1(self):
        # 替换 'your_script.py' 为你要运行的Python文件的路径
        process = subprocess.Popen(['python', 'cs_model.py'])
        self.process_list.append(process)
        process.wait()  # 等待进程完成，即阻塞线程直到进程结束

    def run_script_1(self):
        self.button_stop.setEnabled(True)  # 启动脚本后启用停止按钮
        thread = threading.Thread(target=self.run_script_in_thread1)
        thread.start()

    def run_script_in_thread2(self):
        # 替换 'your_script.py' 为你要运行的Python文件的路径
        process = subprocess.Popen(['python', 'sift.py'])
        self.process_list.append(process)
        process.wait()  # 等待进程完成，即阻塞线程直到进程结束

    def run_script_2(self):
        self.button_stop.setEnabled(True)  # 启动脚本后启用停止按钮
        thread = threading.Thread(target=self.run_script_in_thread2)
        thread.start()

    def stop_script(self):
        for process in self.process_list:
            if process.poll() is None:  # 检查进程是否仍在运行
                process.terminate()  # 发送SIGTERM信号终止进程
                process.wait()  # 等待进程结束


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec_())