import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def smooth(data, weight=0.9):
    """
    用于平滑曲线，类似于Tensorboard中的smooth曲线
    """
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 指数加权移动平均（Exponential Moving Average，EMA），用于对数据进行平滑处理
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

if __name__ == '__main__':
    # fig, ax = plt.subplots(1, 1)  # a figure with a 1x1 grid of Axes
    #
    # # 设置上方和右方无框
    # # ax.spines['top'].set_visible(False)  # 不显示图表框的上边框
    # # ax.spines['right'].set_visible(False)
    #
    len_mean = pd.read_csv("/home/qi/下载/test.csv")
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # # 设置折线颜色，折线标签
    # # 使用平滑处理
    # # ax.plot(len_mean['episode'], smooth(len_mean[''], weight=0.6), color="blue", label="SAC")
    # ax.plot(len_mean['episode'], len_mean['reward'], color="blue", label="SAC")
    # # ax.plot(data, color="red", label='data')
    # # 不使用平滑处理
    # # ax1.plot(len_mean['Step'], len_mean['Value'], color="red",label='all_data')
    #
    # # s设置标签位置，lower upper left right，上下和左右组合
    # plt.legend(loc='lower right')
    #
    # ax.set_xlabel("episode")
    # ax.set_ylabel("reward")
    # ax.set_title("Directly learn the reward curve for the cabinet opening task")
    # # plt.axhline(y=1000, color='red', ls="--", label="最大理论奖励")
    # plt.show()
    # # 保存图片，也可以是其他格式，如pdf
    # fig.savefig(fname='./直接学习开抽屉任务的平滑奖励曲线' + '.png', format='png')
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.title("开抽屉到开门直接策略迁移奖励曲线")
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(len_mean['reward'], label='Rewards', color="blue")
    # plt.plot(smooth(rewards), label='平滑奖励曲线', color="red")
    # plt.ylim([1000, max(rewards)])
    plt.legend()
    plt.show()
