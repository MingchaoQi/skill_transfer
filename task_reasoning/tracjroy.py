import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt


class trac_plan:

    def __init__(self, d_init, d_final, t_final, freq):
        self.d_init = d_init
        self.d_final = d_final
        self.t_final = t_final
        self.freq = freq
        self.n = int(t_final * freq)  # 一定要是一个正整数
        self.t = np.zeros(self.n + 1)
        self.d = np.zeros((self.n + 1, len(self.d_init)))
        self.d_dot = np.zeros((self.n + 1, len(self.d_init)))
        self.d_ddot = np.zeros((self.n + 1, len(self.d_init)))

    def virtual_trac(self):
        for i in range(self.n + 1):
            self.t[i] = i * self.t_final / self.n
            self.d[i] = self.d_init + (self.d_final - self.d_init) * (
                10 * (self.t[i] / self.t_final)**3 - 15 *
                (self.t[i] / self.t_final)**4 + 6 *
                (self.t[i] / self.t_final)**5)
            self.d_dot[i] = (self.d_final - self.d_init) * (
                30 * (self.t[i] / self.t_final)**2 - 60 *
                (self.t[i] / self.t_final)**3 + 30 *
                (self.t[i] / self.t_final)**4) * (1 / self.t_final)
            self.d_ddot[i] = (self.d_final - self.d_init) * (
                60 * (self.t[i] / self.t_final) - 180 *
                (self.t[i] / self.t_final)**2 + 120 *
                (self.t[i] / self.t_final)**3) * (1 / self.t_final**2)

    def virtual_trac_soft(self):
        for i in range(self.n + 1):
            self.t[i] = i * self.t_final / self.n
            self.d[i] = self.d_init + (self.d_final - self.d_init) * (
                10 * (self.t[i] / self.t_final)**3 - 15 *
                (self.t[i] / self.t_final)**4 + 6 *
                (self.t[i] / self.t_final)**5)
            self.d_dot[i] = (self.d_final - self.d_init) * (
                30 * (self.t[i] / self.t_final)**2 - 60 *
                (self.t[i] / self.t_final)**3 + 30 *
                (self.t[i] / self.t_final)**4)
            self.d_ddot[i] = (self.d_final - self.d_init) * (
                60 * (self.t[i] / self.t_final) - 180 *
                (self.t[i] / self.t_final)**2 + 120 *
                (self.t[i] / self.t_final)**3)

    def R(self, k_x, k_y, k_z, theta):
        s = np.sin(theta)
        c = np.cos(theta)
        v = 1 - np.cos(theta)
        T1 = [(k_x * k_x) * v + c, k_x * k_y * v - k_z * s,
              k_x * k_z * v + k_y * s, 0]
        T2 = [
            k_x * k_y * v + k_z * s, k_y * k_y * v + c,
            k_y * k_z * v - k_x * s, 0
        ]
        T3 = [
            k_x * k_z * v - k_y * s, k_y * k_z * v + k_x * s,
            (k_z * k_z) * v + c, 0
        ]
        T4 = [0, 0, 0, 1]

        return np.mat([T1, T2, T3, T4], dtype=np.float64)

    # 旋转矩阵 欧拉角
    def rotate_mat(self, axis, radian):
        rot_matrix = linalg.expm(
            np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        rot1 = np.append(rot_matrix[0], 0)
        rot2 = np.append(rot_matrix[1], 0)
        rot3 = np.append(rot_matrix[2], 0)
        rot4 = [0, 0, 0, 1]

        return np.mat([rot1, rot2, rot3, rot4], dtype=np.float64)

    def T(self, dx, dy, dz):
        T_2_1 = [1, 0, 0, dx]
        T_2_2 = [0, 1, 0, dy]
        T_2_3 = [0, 0, 1, dz]
        T_2_4 = [0, 0, 0, 1]

        return np.mat([T_2_1, T_2_2, T_2_3, T_2_4], dtype=np.float64)

    def resize(self, d_init_rec, d_fin_rec):  # omiga是轨迹的所放量
        l1 = self.d[-1] - self.d[0]  # 原轨迹的方向向量
        l2 = d_fin_rec - d_init_rec  # 新轨迹的方向向量

        self.t_ref = np.zeros_like(self.t)
        self.d_ref = np.zeros_like(self.d)
        self.d_dot_ref = np.zeros_like(self.d_dot)
        self.d_ddot_ref = np.zeros_like(self.d_ddot)
        self.d_ref[0] = self.d[0]

        ##轨迹缩放
        omiga = abs(l2) / abs(l1)
        omiga_t = omiga.max()
        for i in range(len(self.t)):
            self.t_ref[i] = omiga_t * self.t[i]
            for j in range(len(self.d[0])):
                self.d_dot_ref[i][j] = omiga[j] * self.d_dot[i][j] / omiga_t
                self.d_ddot_ref[i][j] = omiga[j] * self.d_ddot[i][j] / (omiga_t
                                                                        **2)
            if i == len(self.t) - 1:
                break
            else:
                for j in range(len(self.d[0])):
                    self.d_ref[i + 1][j] = self.d_ref[i][j] + omiga[j] * (
                        self.d[i + 1][j] - self.d[i][j])

        ##轨迹插补
        self.t_res = np.arange(0, self.t_ref[-1], 1.0 / self.freq)
        self.t_res = np.append(self.t_res, self.t_ref[-1])
        self.d_res = np.zeros((len(self.t_res), len(self.d[0])))
        self.d_dot_res = np.zeros((len(self.t_res), len(self.d[0])))
        self.d_ddot_res = np.zeros((len(self.t_res), len(self.d[0])))
        for j in range(len(self.d[0])):
            x = self.d_ref[:, j]
            z = np.polyfit(self.t_ref, x,
                           5)  # 使用五次多项式进行拟合，拟合精度还有待提高，对周期函数的拟合无能为力
            y = np.poly1d(z)
            x1 = y(self.t_res)
            self.d_res[:, j] = x1
            self.d_dot_res[:, j] = np.append(0, np.diff(self.d_res[:, j]))
            self.d_ddot_res[:, j] = np.append(0, np.diff(self.d_dot_res[:, j]))

        ##轨迹对齐
        self.t_new = self.t_res
        self.d_new = self.d_res
        self.d_dot_new = self.d_dot_res
        self.d_ddot_new = self.d_ddot_res

        l1 = self.d_res[-1] - self.d_res[0]  #由于缩放后轨迹的方向向量改变，需要重新计算
        dl = d_init_rec - self.d_res[0]  # 新老轨迹之间的平移量
        k = np.cross(l1, l2)  # 叉乘求法向量，也是旋转轴
        k_norm = k / np.linalg.norm(k)  # 归一化
        if np.linalg.norm(l1) != 0 and np.linalg.norm(l2) != 0:
            vector_dot_product = np.dot(l1, l2)
            theta = np.arccos(
                vector_dot_product /
                (np.linalg.norm(l1) * np.linalg.norm(l2)))  # 求解新老方向向量夹角

            self.d_new = np.column_stack(
                (self.d_res, np.ones([len(self.t_res), 1])))  # 增加一列1变成增广矩阵
            rot = self.rotate_mat(k_norm, theta)  # 或者用self.R代替，两者计算结果是一样的
            W = self.T(self.d_res[0, 0] + dl[0], self.d_res[0, 1] + dl[1],
                       self.d_res[0, 2] + dl[2]) @ rot @ self.T(
                           -self.d_res[0, 0], -self.d_res[0, 1],
                           -self.d_res[0, 2])  #计算坐标转换矩阵，由两个平移矩阵和一个旋转矩阵相乘构成
            M = np.dot(W, self.d_new.T)
            """
            #这是另一种计算转换后坐标的方式
            # M = np.dot(self.T(-self.d_res[0, 0], -self.d_res[0, 1], -self.d_res[0, 2]), self.d_new.T)
            # rot = self.rotate_mat(k_norm, theta)
            # M = np.dot(rot, M)
            # M = np.dot(self.T(self.d_res[0, 0]+dl[0], self.d_res[0, 1]+dl[1], self.d_res[0, 2]+dl[2]), M)
            """
            self.d_new = np.array(M.T)
            self.d_new = self.d_new[:, 0:len(self.d[0])]
            for j in range(len(self.d_init)):
                self.d_dot_new[:, j] = np.append(0, np.diff(self.d_new[:, j]))
                self.d_ddot_new[:,
                                j] = np.append(0, np.diff(self.d_dot_new[:,
                                                                         j]))
        else:
            print("Zero magnitude vector!")
            return 0


if __name__ == "__main__":
    d_0 = np.array([0.13, -0.10, 0.0])
    d_f = np.array([0.83, 2.5, -1.55])
    t_f = 3.0
    plan = trac_plan(d_0, d_f, t_f, freq=20)
    plan.virtual_trac()
    l = len(plan.d)

    ##画图展示
    z = np.linspace(0, 9, 1000)  # 在1~13之间等间隔取1000个点
    x = 5 * np.cos(z / 7)
    y = 5 * np.sin(z / 7)
    plan.d = np.zeros((1000, 3))
    plan.d[:, 0] = x
    plan.d[:, 1] = y
    plan.d[:, 2] = z
    plan.d_dot = np.zeros((1000, 3))
    plan.d_dot[:, 0] = np.append(0, np.diff(x))
    plan.d_dot[:, 1] = np.append(0, np.diff(y))
    plan.d_dot[:, 2] = np.append(0, np.diff(z))
    plan.d_ddot = np.zeros((1000, 3))
    plan.d_ddot[:, 0] = np.append(0, np.diff(plan.d_dot[:, 0]))
    plan.d_ddot[:, 1] = np.append(0, np.diff(plan.d_dot[:, 1]))
    plan.d_ddot[:, 2] = np.append(0, np.diff(plan.d_dot[:, 2]))
    plan.t = z

    d_new_0 = np.array([2, 2.3, 1.9])
    d_new_f = np.array([13.5, 9.3, 3.1])
    plan.resize(d_new_0, d_new_f)

    d_x = plan.d[:, 0]
    d_y = plan.d[:, 1]
    d_z = plan.d[:, 2]
    d_new_x = plan.d_new[:, 0]
    d_new_y = plan.d_new[:, 1]
    d_new_z = plan.d_new[:, 2]
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(d_x, d_y, d_z, 'blue')
    ax1.plot3D(d_new_x, d_new_y, d_new_z, 'red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.figure()
    plt.plot(d_x, d_y, 'blue', label=r'old trajectory')
    plt.plot(d_new_x, d_new_y, 'red', label=r'new trajectory')
    plt.legend(loc=2, fontsize='large')
    plt.xticks(fontsize=18, fontproperties='Times New Roman')
    plt.yticks(fontsize=18, fontproperties='Times New Roman')
    plt.xlabel('X', fontsize=20, fontdict={'family': 'Times New Roman'})
    plt.ylabel('Y', fontsize=20, fontdict={'family': 'Times New Roman'})
    plt.figure()
    plt.plot(d_x, d_z, 'blue', label=r'old trajectory')
    plt.plot(d_new_x, d_new_z, 'red', label=r'new trajectory')
    plt.legend(loc=1, fontsize='large')
    plt.xticks(fontsize=18, fontproperties='Times New Roman')
    plt.yticks(fontsize=18, fontproperties='Times New Roman')
    plt.xlabel('X', fontsize=20, fontdict={'family': 'Times New Roman'})
    plt.ylabel('Z', fontsize=20, fontdict={'family': 'Times New Roman'})
    plt.show()
