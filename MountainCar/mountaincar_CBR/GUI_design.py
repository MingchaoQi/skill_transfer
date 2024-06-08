import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, \
    QHBoxLayout, QVBoxLayout, QGridLayout, QMainWindow, QTextEdit, QFileDialog
# 需要引入 pyqtSlot 库函数
# from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QMetaObject, pyqtSlot, QTimer, QObject, pyqtSignal, QMutex, QThread, QWaitCondition, Qt, QFile
# import random
from main import BespokeAgent
from test import CBR_TL
from test_contrast import BespokeAgent3D
from stable_baselines3.common import results_plotter
import gym

interact_mutex = QMutex()
ball_mutex = QMutex()
cost_mutex = QMutex()
global cost_record
cost_record = []
aa = []


class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        self.setWindowTitle('人机混合决策系统')
        self.resize(1900, 800)
        self.init_ui()
        QMetaObject.connectSlotsByName(self)

        self.on_run()

    def init_ui(self):

        """Window Background"""
        self.setAutoFillBackground(True)
        # p = self.palette()
        # p.setColor(self.backgroundRole(), Qt.darkYellow)
        # self.setPalette(p)
        # Layout
        self.mainLayout = QGridLayout()
        self.leftLayout = QGridLayout()  ##一些看不见的表格线，用来划分区域

        # Widget（容器）
        self.panel_rule = QWidget()
        self.panel_button = QWidget()
        self.panel_figure = QWidget()
        self.leftLayout.addWidget(self.panel_rule, 0, 0)
        self.leftLayout.addWidget(self.panel_button, 1, 0)
        self.mainLayout.addLayout(self.leftLayout, 0, 0)
        self.mainLayout.addWidget(self.panel_figure, 0, 1)

        # 模块名
        self.rule_label = QLabel('2D mountainCar 控制界面', self)
        self.rule_label.setAlignment(Qt.AlignHCenter)
        self.transfer_label = QLabel('3D mountainCar 控制界面', self)
        self.display_lable = QLabel('图像显示', self)
        self.display_lable.setAlignment(Qt.AlignHCenter)
        self.transfer_label.setAlignment(Qt.AlignHCenter)
        # self.data = QLabel(self)
        # self.data1 = QLabel(self)
        # self.data2 = QLabel(self)
        self.txt = QLineEdit(self)

        # 2D rules
        self.current_rules_line = QTextEdit(self)
        self.if_label1_1 = QLabel('位置', self)
        self.if_label1_2 = QLabel('速度', self)
        self.then_label1 = QLabel('动作', self)
        self.if_line1_1 = QLineEdit(self)
        self.if_line1_2 = QLineEdit(self)
        self.then_line1 = QLineEdit(self)
        self.mu_label1 = QLabel('均值一', self)
        self.sigma_label1 = QLabel('方差一', self)
        self.mu_line1 = QLineEdit(self)
        self.sigma_line1 = QLineEdit(self)
        self.mu_label2 = QLabel('均值二', self)
        self.sigma_label2 = QLabel('方差二', self)
        self.mu_line2 = QLineEdit(self)
        self.sigma_line2 = QLineEdit(self)
        self.rule_button = QPushButton('现有规则', self)
        self.rule_button.setObjectName('rule_button')
        self.addrule_button = QPushButton('上传规则', self)
        self.addrule_button.setObjectName('addrule_button')
        self.addpara_button = QPushButton('添加参数', self)
        self.addpara_button.setObjectName('addpara_button')
        self.train_button = QPushButton('训练', self)
        self.train_button.setObjectName('train_button')
        self.use_guide = QPushButton('使用说明', self)
        self.use_guide.setObjectName('use_guide')

        # 3D rules
        self.transfer_rules_line = QTextEdit(self)
        self.if_label2_1 = QLabel('x位置', self)
        self.if_label2_2 = QLabel('x速度', self)
        self.if_label2_3 = QLabel('y位置', self)
        self.if_label2_4 = QLabel('y速度', self)
        self.then_label2 = QLabel('动作', self)
        self.if_line2_1 = QLineEdit(self)
        self.if_line2_2 = QLineEdit(self)
        self.if_line2_3 = QLineEdit(self)
        self.if_line2_4 = QLineEdit(self)
        self.then_line2 = QLineEdit(self)
        self.mu_label1_2 = QLabel('均值一', self)
        self.sigma_label1_2 = QLabel('方差一', self)
        self.mu_line1_2 = QLineEdit(self)
        self.sigma_line1_2 = QLineEdit(self)
        self.mu_label2_2 = QLabel('均值二', self)
        self.sigma_label2_2 = QLabel('方差二', self)
        self.mu_line2_2 = QLineEdit(self)
        self.sigma_line2_2 = QLineEdit(self)
        self.ruleflect_button = QPushButton('规则映射', self)
        self.ruleflect_button.setObjectName('ruleflect_button')
        self.addrule_button_1 = QPushButton('上传规则', self)
        self.addrule_button_1.setObjectName('addrule_button_1')
        self.addpara_button_1 = QPushButton('添加参数', self)
        self.addpara_button_1.setObjectName('addpara_button_1')
        self.train_button_1 = QPushButton('训练', self)
        self.train_button_1.setObjectName('train_button_1')
        self.rulegene_button = QPushButton('生成规则', self)
        self.rulegene_button.setObjectName('rulegene_button')
        self.compare_button = QPushButton('结果对比', self)
        self.compare_button.setObjectName('compare_button')

        # Figure Display
        self.figure = plt.figure(facecolor='#FFD7C4', figsize=(7, 7))

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0.5)
        self.canves = FigureCanvas(self.figure)
        self.axes1 = self.figure.add_subplot(311)
        # """
        self.axes2 = self.figure.add_subplot(312)
        self.axes3 = self.figure.add_subplot(313)
        self.axes1.set_xlabel('iterations')
        self.axes2.set_xlabel('iterations')
        self.axes3.set_xlabel('iterations')
        self.axes1.set_ylabel('theta')
        self.axes2.set_ylabel('beta')
        self.axes3.set_ylabel('other parameter')
        # """

        self.figure_ = plt.figure(facecolor='#FFD7C4', figsize=(7, 7))

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0.5)
        self.canves_ = FigureCanvas(self.figure_)
        self.axes1_ = self.figure_.add_subplot(111)
        self.axes1_.set_xlabel('iterations')
        self.axes1_.set_ylabel('reward')

        ##--------------------------------------------------------
        lay_rule = QGridLayout(self.panel_rule)
        lay_rules = QGridLayout()
        lay_rule_left = QGridLayout()
        lay_rule_top = QGridLayout()
        lay_rule_bottom = QGridLayout()
        ###
        lay_rule_left.addWidget(self.current_rules_line, 0, 0)
        lay_rule_left.addWidget(self.rule_button, 1, 0)
        ###
        lay_rules.addWidget(self.if_label1_1, 0, 0)
        lay_rules.addWidget(self.if_line1_1, 0, 1)
        lay_rules.addWidget(self.if_label1_2, 0, 2)
        lay_rules.addWidget(self.if_line1_2, 0, 3)
        lay_rules.addWidget(self.then_label1, 1, 0)
        lay_rules.addWidget(self.then_line1, 1, 1)
        lay_rules.addWidget(self.addrule_button, 2, 1)
        lay_rules.addWidget(self.use_guide, 2, 3)
        ###
        lay_rules.addWidget(self.mu_label1, 3, 0)
        lay_rules.addWidget(self.mu_line1, 3, 1)
        lay_rules.addWidget(self.sigma_label1, 4, 0)
        lay_rules.addWidget(self.sigma_line1, 4, 1)
        lay_rules.addWidget(self.mu_label2, 3, 2)
        lay_rules.addWidget(self.mu_line2, 3, 3)
        lay_rules.addWidget(self.sigma_label2, 4, 2)
        lay_rules.addWidget(self.sigma_line2, 4, 3)
        lay_rules.addWidget(self.addpara_button, 5, 0)
        lay_rules.addWidget(self.train_button, 5, 3)

        lay_rule_top.addWidget(self.rule_label, 0, 0)
        ###
        lay_rule.addLayout(lay_rule_top, 0, 0)
        lay_rule_bottom.addLayout(lay_rules, 0, 1)
        lay_rule_bottom.addLayout(lay_rule_left, 0, 0)
        lay_rule.addLayout(lay_rule_bottom, 1, 0)

        ##-------------------------------------------
        lay_transfer = QGridLayout(self.panel_button)
        lay_transfers = QGridLayout()
        lay_transfer_left = QGridLayout()
        lay_transfer_top = QGridLayout()
        lay_transfer_bottom = QGridLayout()

        lay_transfer_left.addWidget(self.transfer_rules_line, 0, 0)
        lay_transfer_left.addWidget(self.ruleflect_button, 1, 0)

        lay_transfers.addWidget(self.if_label2_1, 0, 0)
        lay_transfers.addWidget(self.if_line2_1, 0, 1)
        lay_transfers.addWidget(self.if_label2_2, 0, 2)
        lay_transfers.addWidget(self.if_line2_2, 0, 3)
        lay_transfers.addWidget(self.if_label2_3, 1, 0)
        lay_transfers.addWidget(self.if_line2_3, 1, 1)
        lay_transfers.addWidget(self.if_label2_4, 1, 2)
        lay_transfers.addWidget(self.if_line2_4, 1, 3)
        lay_transfers.addWidget(self.then_label2, 2, 0)
        lay_transfers.addWidget(self.then_line2, 2, 1)
        lay_transfers.addWidget(self.addrule_button_1, 3, 1)

        lay_transfers.addWidget(self.mu_label1_2, 4, 0)
        lay_transfers.addWidget(self.mu_line1_2, 4, 1)
        lay_transfers.addWidget(self.sigma_label1_2, 5, 0)
        lay_transfers.addWidget(self.sigma_line1_2, 5, 1)
        lay_transfers.addWidget(self.mu_label2_2, 4, 2)
        lay_transfers.addWidget(self.mu_line2_2, 4, 3)
        lay_transfers.addWidget(self.sigma_label2_2, 5, 2)
        lay_transfers.addWidget(self.sigma_line2_2, 5, 3)
        lay_transfers.addWidget(self.addpara_button_1, 6, 0)
        lay_transfers.addWidget(self.rulegene_button, 6, 1)
        lay_transfers.addWidget(self.train_button_1, 6, 3)

        lay_transfer_top.addWidget(self.transfer_label, 0, 0)

        lay_transfer.addLayout(lay_transfer_top, 0, 0)
        lay_transfer_bottom.addLayout(lay_transfers, 0, 1)
        lay_transfer_bottom.addLayout(lay_transfer_left, 0, 0)
        lay_transfer.addLayout(lay_transfer_bottom, 1, 0)

        ##-------------------------------------------------
        lay_figure = QGridLayout(self.panel_figure)
        lay_figures = QGridLayout()
        lay_figure.addWidget(self.display_lable, 0, 0)
        lay_figures.addWidget(self.canves, 0, 0)
        lay_figures.addWidget(self.canves_, 0, 1)
        lay_figures.addWidget(self.txt, 1, 0)
        lay_figures.addWidget(self.compare_button, 1, 1)
        lay_figure.addLayout(lay_figures, 1, 0)

        """Stylesheet"""
        self.panel_rule.setStyleSheet("background-color:rgb(200,255,255);")
        """Stylesheet"""
        self.panel_button.setStyleSheet("background-color:rgb(200,200,255);")
        """Stylesheet"""
        # self.panel_data.setStyleSheet("background-color:rgb(200,200,200);")
        """Stylesheet"""
        self.panel_figure.setStyleSheet("background-color:rgb(255,180,200);")
        """Initiating  mainLayout """

        self.window = QWidget()
        self.window.setLayout(self.mainLayout)
        self.setCentralWidget(self.window)

    @pyqtSlot()
    def on_rule_button_clicked(self):
        print('显示原规则')
        self.txt.setText('显示原规则')
        self.txt.setStyleSheet('color:red')
        self.current_rules_line.setText('[if x>-0.5 and 0<v<theta, then a=0];\n'
                                        '[if x>-0.5 and v>=theta, then a=2];\n'
                                        '[if x<-0.5 and v>0, then a=2];\n'
                                        '[if x<beta and -theta<v<0, then a=2];\n'
                                        '[if x<beta and v<=-theta, then a=0];\n'
                                        '[if x>beta and v<0, then a=0]')  # 显示内容替换为全局变量

    @pyqtSlot()
    def on_addrule_button_clicked(self):
        self.txt.setText('增加一条规则')
        self.txt.setStyleSheet('color:red')
        self.current_rules_line.append(
            'new rule:[' + self.if_line1_1.text() + self.if_line1_2.text() + self.then_line1.text() + ']')
        print('增加一条规则')

    @pyqtSlot()
    def on_addpara_button_clicked(self):
        try:
            mu[0] = float(self.mu_line1.text())
            sigmma[0] = float(self.sigma_line1.text())
            mu_beta[0] = float(self.mu_line2.text())
            sigmma_beta[0] = float(self.sigma_line2.text())
            print('你添加的参数是：\n'
                  'mu=%f\n'
                  'sigma=%f\n'
                  'mu_beta=%f\n'
                  'sigmma_beta=%f' % (mu[0], sigmma[0], mu_beta[0], sigmma_beta[0]))
            self.txt.setText('参数添加完成')
            self.txt.setStyleSheet('color:red')
            print('参数添加完成')
        except:
            self.txt.setText('请规范添加参数，数字类型为实数')
            self.txt.setStyleSheet('color:red')
            print('请规范添加参数，数字类型为实数')

    @pyqtSlot()
    def on_ruleflect_button_clicked(self):
        self.transfer_rules_line.setText('if x>theta and a_init<2, then a=a_init + 3\n'
                                         'else a=2')  # 显示内容替换为全局变量
        self.txt.setText('规则映射')
        self.txt.setStyleSheet('color:red')
        print('规则映射')

    @pyqtSlot()
    def on_addpara_button_1_clicked(self):  ##改成新环境中的参数变量
        try:
            theta[0] = float(self.mu_line1_2.text())
            sigma_theta[0] = float(self.sigma_line1_2.text())
            self.txt.setText('参数添加完成')
            self.txt.setStyleSheet('color:red')
            print('参数添加完成')
        except:
            self.txt.setText('请规范添加参数，数字类型为实数')
            self.txt.setStyleSheet('color:red')
            print('请规范添加参数，数字类型为实数')

    @pyqtSlot()
    def on_addrule_button_1_clicked(self):
        self.txt.setText('增加一条规则')
        self.txt.setStyleSheet('color:red')
        self.transfer_rules_line.append(
            'new rule:[' + self.if_line2_1.text() + self.if_line2_2.text() + self.if_line2_3.text() + self.if_line2_4.text() + self.then_line2.text() + ']')
        print('增加一条规则')

    @pyqtSlot()
    def on_rulegene_button_clicked(self):
        self.transfer_rules_line.setText('if y!=beta, then agent go towards the beta\n'
                                         'if x>theta and a_init<2, then a=a_init + 3\n'
                                         'else a=2')  # 显示内容替换为全局变量
        self.txt.setText('规则生成完成')
        self.txt.setStyleSheet('color:red')
        print('增加一条规则')

    def on_run(self):  ###
        # 创建工作线程
        self.thread = QThread()
        self.worker = Worker(BespokeAgent(gym.make('MountainCar-v0')))  # move init learn to here
        self.worker.moveToThread(self.thread)
        self.flag = 0  ##一个信号量，用于判断进行哪一各panel的程序
        # #
        # # # 连接信号量
        self.thread.started.connect(self.worker.run_2D)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.display.connect(self.Plot_trajectory)
        self.worker.draw_cost.connect(self.Plot_cost)
        self.worker.display_new.connect(self.Plot_trajectory_new)
        self.worker.draw_cost_new.connect(self.Plot_cost_new)
        self.worker.draw_data.connect(self.show_data)
        self.worker.show_newrule.connect(self.show_newrule)

        self.train_button.clicked.connect(self.doWake)
        self.train_button_1.clicked.connect(self.doaWake)
        self.rulegene_button.clicked.connect(self.doaWake)
        self.compare_button.clicked.connect(self.Plot_cost_rl)
        self.use_guide.clicked.connect(self.read_files)
        # self.graphWidget.clear()
        # 运行工作线程
        self.thread.start()

        return

    def show_data(self):  ##显示出生成的新规则的参数
        try:
            self.mu_line2_2.setText(str(beta[0]))
            self.sigma_line2_2.setText(str(sigma_beta[0]))
        except:
            self.txt.setText('参数显示有误')
            self.txt.setStyleSheet('color:red')
            print('参数显示有误')

    def show_newrule(self):
        self.transfer_rules_line.setText(
            '[if flag==1 and beta < -0.5 and abs(position_y) < abs(beta), then action = 3]\n'
            '[if flag==1 and abs(position_y) > abs(beta) and abs(velocity_y) < 0.005, then flag=0]\n'
            '[if flag==1 and abs(position_y) > abs(beta) and abs(velocity_y) < 0.00, then action=2]\n'
            '[if flag==0 and position_x >= theta and velocity_y >= 0 and (position_y > -0.5) and (abs(velocity_y) < 0.018433), then action=3]\n'
            '[if flag==0 and position_x >= theta and velocity_y >= 0 and (position_y < -0.5), then action=4]\n'
            '[if flag==0 and position_x >= theta and velocity_y < 0 and (position_y < -0.63454) and (abs(velocity_y) < 0.018433), then action=4]\n'
            '[if flag==0 and position_x >= theta and velocity_y < 0 and (position_y > -0.63454), then action=3]\n'
            '[if flag==0 and position_x < theta and velocity_x >= 0 and (position_x > -0.5) and (abs(velocity_x) < 0.018433), then action=0]\n'
            '[if flag==0 and position_x < theta and velocity_x >= 0 and (position_x < -0.5), then action=1]\n'
            '[if flag==0 and position_x < theta and velocity_x < 0 and (position_x < -0.63454) and (abs(velocity_x) < 0.018433), then action=1]\n'
            '[if flag==0 and position_x < theta and velocity_x < 0 and (position_x > -0.63454), then action=0]')  # 显示内容替换为全局变量
        self.txt.setText('新规则整合完成')
        self.txt.setStyleSheet('color:red')
        print('增加一条规则')

    def Plot_trajectory(self):  ###画出2D场景下的参数曲线
        self.axes1.clear()
        self.axes2.clear()
        self.axes3.clear()
        self.axes1.plot(mu, 'r-')
        self.axes2.plot(mu_beta, 'k-')
        self.axes1.set_xlabel('iterations')
        self.axes1.set_ylabel('theta')
        self.axes2.set_xlabel('iterations')
        self.axes2.set_ylabel('beta')
        self.axes3.set_xlabel('iterations')
        self.canves.draw()

    def Plot_trajectory_new(self):  ###画出3D场景下的参数曲线
        self.axes1.clear()
        self.axes2.clear()
        self.axes3.clear()
        self.axes1.plot(theta, 'r-')
        self.axes2.plot(beta, 'k-')
        self.axes1.set_xlabel('iterations')
        self.axes1.set_ylabel('theta_new')
        self.axes2.set_xlabel('iterations')
        self.axes2.set_ylabel('beta_new')
        self.axes3.set_xlabel('iterations')
        self.canves.draw()

    def Plot_cost(self):  ###画出2D场景下的奖励值
        # self.txt.setText('输入已完成')
        # self.txt.setStyleSheet('color:green')
        self.axes1_.clear()
        self.axes1_.plot(R, 'r-', label='Our method')
        self.axes1_.set_xlabel('iterations')
        self.axes1_.set_ylabel('reward')
        self.canves_.draw()

    def Plot_cost_new(self):  ###画出3D场景下的奖励值
        # self.txt.setText('输入已完成')
        # self.txt.setStyleSheet('color:green')
        self.axes1_.clear()
        self.axes1_.plot(R_transfer, 'r-', label='Our method')
        self.axes1_.set_xlabel('iterations')
        self.axes1_.set_ylabel('reward')
        self.canves_.draw()

    def Plot_cost_rl(self):  ##画其他方法得到的奖励值图
        self.axes1_.clear()
        try:
            R1 = np.append(R_transfer, R_transfer[-1] * np.ones(1420 - len(R_transfer)))
            self.axes1_.plot(R1, 'b-', label='Our method')
            log_dir = "monitor"
            results_plotter.plot_results([log_dir], 1e6, results_plotter.X_EPISODES, "DQN contrast CBR+PGPE")
            self.canves_.draw()
        except:
            self.txt.setText('原曲线未找到')
            self.txt.setStyleSheet('color:red')
            print('原曲线未找到')

    def read_files(self):
        # self指向自身，"Open File"为文件名，"./"为当前路径，最后为文件类型筛选器
        fname, ftype = QFileDialog.getOpenFileName(self, "mountainCar技能转移与推理GUI使用手册.txt", "./",
                                                   "All Files(*);;Wav(*.wav);;Txt (*.txt)")  # 如果添加一个内容则需要加两个分号
        # 该方法返回一个tuple,里面有两个内容，第一个是路径， 第二个是要打开文件的类型，所以用两个变量去接受
        # 如果用户主动关闭文件对话框，则返回值为空
        if fname[0]:  # 判断路径非空
            f = QFile(fname[0])  # 创建文件对象，不创建文件对象也不报错 也可以读文件和写文件
            # open()会自动返回一个文件对象
            f = open(fname[0], "r")  # 打开路径所对应的文件， "r"以只读的方式 也是默认的方式
            with f:
                data = f.read()
                # print(data)
                self.textEdit.setText(data)
            f.close()

    def doWait(self):
        self.worker.pause()

    def doWake(self):
        try:
            if mu[0] == 0 or mu_beta[0] == 0:
                self.txt.setText('添加参数性质')
                self.txt.setStyleSheet('color:red')
                print('添加参数性质')
            else:
                self.flag = 1
                self.worker.resume()
        except:
            self.txt.setText('请先添加参数性质或进行原策略训练')
            self.txt.setStyleSheet('color:red')
            print('请先添加参数性质或进行原策略训练')

    def doaWake(self):
        try:
            if theta[0] == 0 or sigma_theta[0] == 0 or self.flag == 0:
                if self.flag == 1:
                    self.txt.setText('添加参数性质')
                    self.txt.setStyleSheet('color:red')
                    print('添加参数性质')
                else:
                    self.txt.setText('先进行原策略的训练')
                    self.txt.setStyleSheet('color:red')
                    print('先进行原策略的训练')
            else:
                self.worker.resume()
        except:
            self.txt.setText('请先添加参数性质或进行原策略训练')
            self.txt.setStyleSheet('color:red')
            print('请先添加参数性质或进行原策略训练')


class Worker(QObject):
    # 通信信号
    finished = pyqtSignal()
    data = pyqtSignal()
    display = pyqtSignal()
    display_new = pyqtSignal()
    draw_cost = pyqtSignal()
    draw_cost_new = pyqtSignal()
    draw_data = pyqtSignal()
    show_newrule = pyqtSignal()

    # 通过共享数据
    def __init__(self, ln) -> None:
        super(Worker, self).__init__()
        self.learn = ln
        # pause the thread
        self._isPause = False
        self._value = 0
        self.cond = QWaitCondition()
        self.mutex = QMutex()
        self.save_cost = []

    def run_2D(self):
        global mu, sigmma, mu_beta, sigmma_beta, R, theta, sigma_theta, beta, sigma_beta, R_transfer
        interact_mutex.lock()
        alpha = 0.95  # studying rate
        N = 20  # parameters number
        M = 1  # eposides number
        mu = list(range(51))
        sigmma = list(range(51))
        mu_beta = list(range(51))
        sigmma_beta = list(range(51))
        R = list()

        self._isPause = True

        for t in range(50):  # range number
            self.mutex.lock()
            if self._isPause:
                self.cond.wait(self.mutex)
            self.mutex.unlock()
            u, g, a, b = self.learn.policy(mu[t], sigmma[t], mu_beta[t], sigmma_beta[t], N, M, alpha)
            mu[t + 1] = u
            sigmma[t + 1] = g
            mu_beta[t + 1] = a
            sigmma_beta[t + 1] = b
            print("mu = ", mu[t + 1])
            print("sigmma = ", sigmma[t + 1])
            print("mu_beta = ", mu_beta[t + 1])
            print("sigmma_beta = ", sigmma_beta[t + 1])
            print("delte_mu=", mu[t + 1] - mu[t])
            print("delte_beta=", mu_beta[t + 1] - mu_beta[t])
            r = self.learn.play_ones(mu[t], mu_beta[t])
            R.append(r)
            print("reward_toword=", R[t])
            if sigmma[t + 1] < 0.00005:
                break
        self.display.emit()
        self.draw_cost.emit()

        ##进入技能转移阶段
        env = gym.make("MountainCar-v0")
        agent = CBR_TL()
        try:
            case = np.load('casebase_PGPE.npy')
        except:
            case = agent.case_generation(env)
            np.save('casebase_PGPE.npy', arr=case)
        env.close()

        env = gym.make('MountainCar3D-v0')
        N1 = 10
        N2 = 20
        theta = np.zeros(N1 + 1)
        sigma_theta = np.zeros(N1 + 1)
        R_transfer = np.zeros(0)
        num_range = 10
        num_episode = 1
        alpha = 0.95

        self.pause()  ##等待点击“开始训练”按钮
        for t in range(len(theta) - 1):
            self.mutex.lock()
            if self._isPause:
                self.cond.wait(self.mutex)
            self.mutex.unlock()
            r, observation_init = agent.play_ones(env, theta[t], case)
            R_transfer = np.append(R_transfer, r)
            u, g = agent.policy(env, case, theta[t], sigma_theta[t], num_range, num_episode, alpha)
            theta[t + 1] = u
            sigma_theta[t + 1] = g
            print("mu = ", theta[t + 1])
            print("sigma = ", sigma_theta[t + 1])
            print("reward_toward=", R_transfer[t])
        self.draw_cost_new.emit()

        # 微调智能体初始位置，收集奖励情况
        R1 = np.zeros(0)
        agent.flag = 1  # 表示微调的是y方向的初值
        # agent.flag = 0
        state_init = np.zeros((N1, 4))  # 收集初始状态点
        self.pause()  ##等待点击“生成规则”按钮
        for t in range(N1):
            self.mutex.lock()
            if self._isPause:
                self.cond.wait(self.mutex)
            self.mutex.unlock()
            r, observation_init = agent.play_ones(env, theta[-1], case, index=True)
            R1 = np.append(R1, r)
            print("reward_toward=", R1[t])
            state_init[t] = observation_init
        r_max_index = np.argmax(R1)

        theta_1 = np.zeros(N2 + 1)
        sigma_theta_1 = np.zeros(N2 + 1)
        beta = np.zeros(N2 + 1)
        sigma_beta = np.zeros(N2 + 1)
        if agent.flag == 0:
            x_target = state_init[r_max_index, 2]
            # y_target = -0.54857755
            x1 = (-0.5 + x_target) / 2
            if x1 < -0.5:
                agent.flag_v = 0  # 表示要向负方向运动
            else:
                agent.flag_v = 1  # 表示要向正方向运动

            theta_1[0] = theta[-1]  # 原参数从优化后的结果开始优化
            sigma_theta_1[0] = 0.01  # 探索方差的大小可以再斟酌
            beta[0] = x1
            sigma_beta[0] = 0.01
        if agent.flag == 1:
            y_target = state_init[r_max_index, 2]
            # y_target = -0.3
            y1 = (-0.5 + y_target) / 2
            if y1 < -0.5:
                agent.flag_v = 0  # 表示要向负方向运动
            else:
                agent.flag_v = 1  # 表示要向正方向运动

            theta_1[0] = theta[-1]  # 原参数从优化后的结果开始优化
            sigma_theta_1[0] = 0.1  # 探索方差的大小可以再斟酌
            beta[0] = y1
            sigma_beta[0] = abs(-0.5 - y_target) / 4
        R2 = np.zeros(0)
        num_range = 20
        R_transfer = np.append(R_transfer, R1)
        self.draw_data.emit()
        self.draw_cost_new.emit()

        self.pause()  ##等待点击“开始训练”按钮
        for t in range(len(theta_1) - 1):
            self.mutex.lock()
            if self._isPause:
                self.cond.wait(self.mutex)
            self.mutex.unlock()
            r, _ = agent.play_twice(env, theta_1[t], beta[t], case)
            R2 = np.append(R2, r)
            u, g, a, b = agent.policy_twice(env, case, theta_1[t], sigma_theta_1[t], beta[t], sigma_beta[t],
                                            num_range, num_episode, alpha)
            theta_1[t + 1] = u
            sigma_theta_1[t + 1] = g
            beta[t + 1] = a
            sigma_beta[t + 1] = b
            print("theta = ", theta_1[t + 1])
            print("sigma_theta = ", sigma_theta_1[t + 1])
            print("beta = ", beta[t + 1])
            print("sigma_beta = ", sigma_beta[t + 1])
            print("reward_toward=", R2[t])

        R_transfer = np.append(R_transfer, R2)
        theta = np.append(theta, theta_1)
        sigma_theta = np.append(sigma_theta, sigma_theta_1)
        self.display_new.emit()
        self.draw_cost_new.emit()

        ##使用组合后的新规则继续训练
        env = gym.make('MountainCar3D-v0')
        agent = BespokeAgent3D(env)
        alpha = 0.95  # studying rate
        N = 20  # parameters number
        M = 1  # eposides number
        N3 = 60
        theta_new = np.zeros(N3 + 1)
        sigma_theta_new = np.zeros(N3 + 1)
        beta_new = np.zeros(N3 + 1)
        sigma_beta_new = np.zeros(N3 + 1)
        R3 = np.zeros(0)

        theta_new[0] = theta[-1]
        sigma_theta_new[0] = 0.1
        beta_new[0] = beta[-1]
        sigma_beta_new[0] = 0.05

        self.show_newrule.emit()
        self.pause()  ##等待点击“训练”按钮
        for t in range(N3):  # range number
            self.mutex.lock()
            if self._isPause:
                self.cond.wait(self.mutex)
            self.mutex.unlock()
            u, g, a, b = agent.policy(theta_new[t], sigma_theta_new[t], beta_new[t], sigma_beta_new[t], N, M, alpha)
            theta_new[t + 1] = u
            sigma_theta_new[t + 1] = g
            beta_new[t + 1] = a
            sigma_beta_new[t + 1] = b
            print("mu = ", theta_new[t + 1])
            print("sigmma = ", sigma_theta_new[t + 1])
            print("mu_beta = ", beta_new[t + 1])
            print("sigmma_beta = ", sigma_beta_new[t + 1])
            print("delte_mu=", theta_new[t + 1] - theta_new[t])
            print("delte_beta=", beta_new[t + 1] - beta_new[t])
            r = agent.play_ones(theta_new[t], beta_new[t])
            R3 = np.append(R3, r)
            print("reward_toword=", R3[t])
            if sigma_theta_new[t + 1] < 0.00005:
                break
        R_transfer = np.append(R_transfer, R3)
        theta = np.append(theta, theta_new)
        sigma_theta = np.append(sigma_theta, sigma_theta_new)
        beta = np.append(beta, beta_new)
        sigma_beta = np.append(sigma_beta, sigma_beta_new)
        self.display_new.emit()
        self.draw_cost_new.emit()

        self.finished.emit()
        interact_mutex.unlock()

    def pause(self):
        self._isPause = True

    def resume(self):
        self._isPause = False
        self.cond.wakeAll()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = GUI()
    demo.show()
    sys.exit(app.exec_())
