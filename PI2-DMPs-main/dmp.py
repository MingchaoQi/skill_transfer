import numpy as np
import matplotlib.pyplot as plt
import math


class OriginSystemTerm:
    """
    ydd=alpha_y*(beta_y(goal-y)-yd) + force(basic function)
    yd+=tau*ydd
    y+=tau*yd
    tau>1代表加速仿真，tau<1代表减速仿真
    """

    def __init__(self, alpha_y=45, beta_y=16, alpha_g=12, start=0, goal=1, tau=1.0):
        self.alpha_y = alpha_y
        self.beta_y = beta_y
        self.alpha_g = alpha_g
        self.tau = tau

        self.ydd = 0
        self.yd = 0
        self.y = start

        self.g = goal
        self.dG = goal - start  # g-y0项,对f项有一个空间缩放

    def prepare_step(self, f):
        self.ydd = self.alpha_y * (self.beta_y * (self.g - self.y) - self.yd) + f

    def run_step(self, dt):
        self.y += self.tau * self.yd * dt
        self.yd += self.tau * self.ydd * dt


class NonlinearTerm:
    """
    phi_i(x)=exp(-0.5*(x-c_i)**2*D)
    xd=-alpha_x*x (canonical dynamical system)

    x+=tau*xd
    """

    def __init__(self, start_time=0.0, end_time=1.0, n_bfs=10, alpha_x=6, tau=1.0):
        self.start = start_time
        self.end = end_time
        self.n_bfs = n_bfs
        self.alpha_x = alpha_x
        self.tau = tau
        self.mean_time = np.linspace(start_time, end_time, n_bfs + 2)[1:-1]  #
        self.mean_canonical = np.exp(-alpha_x / tau * self.mean_time)  # 正态非线性的均值，取一阶正则系统x在某时刻的取值
        self.weight = np.random.random(n_bfs)
        self.sx2 = np.ones(n_bfs)  # fit the tracjectory
        self.sxtd = np.ones(n_bfs)  # fit the tracjectory
        self.D = (np.diff(self.mean_canonical) * 0.55) ** 2
        self.D = 1 / np.hstack((self.D, self.D[-1]))  # 基函数的方差，控制分布覆盖作用范围（基函数的宽度）

        self.x = 1  # 起点为1
        self.xd = 0

    def set_weight(self, weight):
        self.weight = weight

    def prepare_step(self):  # canonical dynamical system
        self.xd = self.tau * (-self.alpha_x * self.x)

    def run_step(self, dt):
        self.x += self.xd * self.tau * dt

    def get_psi(self):  #
        psi = np.exp(-0.5 * (self.x - self.mean_canonical) ** 2 * self.D)
        return psi

    def calc_f(self, delta_y):  # non-linear dynamic character
        f = self.weight.dot(self.calc_g(delta_y))
        return f

    def calc_g(self, delta_y):
        psi = self.get_psi()
        g = psi / np.sum(psi + 1e-10) * self.x * delta_y
        return g

    def show(self):  # 试验性画出基函数分布，观察其在时间的分布
        t = np.linspace(self.start, self.end, 100)
        x = np.exp(-self.alpha_x / self.tau * t)
        y = []
        for i in range(self.n_bfs):
            yi = np.exp(-(x - self.mean_canonical[i]) ** 2 / 2 * self.D[i])
            y.append(yi)
        y = np.array(y).T
        plt.plot(t, y)
        plt.show()


class DMPs:
    def __init__(self, start=np.array([0]), goal=np.array([1]), start_time=0.0, end_time=2.0, start_dmp_time=0.0, end_dmp_time=1.0, n_bfs=10,
                 tau=1.0, dt=0.01, n_dmps=1):
        self.tau = tau
        self.start_time = start_time
        self.end_time = end_time
        self.start_dmp_time = start_dmp_time
        self.end_dmp_time = end_dmp_time
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.start = start
        self.goal = goal
        self.sys = list(range(n_dmps))
        self.f_term = NonlinearTerm(start_time=start_dmp_time, end_time=end_dmp_time, tau=tau, n_bfs=n_bfs)
        self.dt = dt
        self.t = np.zeros([self.n_dmps])
        self.length = int((end_time - start_time) / dt)  #
        self.y = np.zeros([self.n_dmps]) #
        self.yd = np.zeros([self.n_dmps])  #
        self.ydd = np.zeros([self.n_dmps])  #
        self.x = np.zeros([self.n_dmps])  #
        self.psi = np.zeros([n_dmps, self.length, n_bfs])  #
        self.weight = np.zeros([n_dmps, n_bfs])  # 基础权重，对一个rollout是一样的
        self.f = 0  # 非线性项外力的值
        for sys_index in range(self.n_dmps):
            self.sys[sys_index] = OriginSystemTerm(start=self.start[sys_index], goal=self.goal[sys_index], tau=self.tau)

    def run_step(self, has_dmp, sys_index):
        if has_dmp:
            self.f = self.f_term.calc_f(self.sys[sys_index].dG)
            self.f_term.prepare_step()  # resert x,xd,xdd
            self.f_term.run_step(self.dt)
            self.sys[sys_index].prepare_step(self.f)  # resert y,yd,ydd
            self.sys[sys_index].run_step(self.dt)
        else:
            self.f = 0.0  # 相当于PD控制逐渐收敛到goal
            self.sys[sys_index].prepare_step(self.f)  # resert y,yd,ydd
            self.sys[sys_index].run_step(self.dt)
        self.y[sys_index] = self.sys[sys_index].y
        self.yd[sys_index] = self.sys[sys_index].yd
        self.ydd[sys_index] = self.sys[sys_index].ydd
        self.x[sys_index] = self.f_term.x
        # self.psi = self.f_term.get_psi()
        # self.weight = self.f_term.weight
        self.t[sys_index] = self.t[sys_index] + self.dt

    def run_trajectory(self):
        length = int((self.end_time - self.start_time) / self.dt)
        y = np.zeros([self.n_dmps, length])
        yd = np.zeros([self.n_dmps, length])
        ydd = np.zeros([self.n_dmps, length])
        t = np.zeros([self.n_dmps, length])
        for sys_index in range(self.n_dmps):
            self.f_term = NonlinearTerm(start_time=self.start_dmp_time, end_time=self.end_dmp_time, tau=self.tau, n_bfs=self.n_bfs)
            self.f_term.weight = self.weight[sys_index]
            for i in range(length):
                if self.start_dmp_time <= self.t[sys_index] < self.end_dmp_time:
                    self.run_step(has_dmp=True, sys_index=sys_index)
                else:
                    self.run_step(has_dmp=False, sys_index=sys_index)
                y[sys_index][i] = self.y[sys_index]
                yd[sys_index][i] = self.yd[sys_index]
                ydd[sys_index][i] = self.ydd[sys_index]
                t[sys_index][i] = self.t[sys_index]

        return y, yd, ydd, t

    def run_fit_trajectory(self, target: np.ndarray, target_d: np.ndarray, target_dd: np.ndarray):
        """
        target行向量
        """
        for sys_index in range(self.n_dmps):
            y0 = target[sys_index][0]
            g = target[sys_index][-1]

            X = np.zeros(target[sys_index].shape)
            G = np.zeros(target[sys_index].shape)
            x = 1

            for i in range(len(target[sys_index])):
                X[i] = x
                G[i] = g
                xd = -self.f_term.alpha_x * x
                x += xd * self.tau * self.dt

            self.sys[sys_index].dG = g - y0
            F_target = (target_dd[sys_index] / (self.tau ** 2) - self.sys[sys_index].alpha_y * (
                    self.sys[sys_index].beta_y * (G - target[sys_index]) - target_d[sys_index] / self.tau))  # f target
            PSI = np.exp(
                -0.5 * ((X.reshape((-1, 1)).repeat(self.f_term.n_bfs, axis=1) - self.f_term.mean_canonical.reshape(1,
                                                                                                                   -1)
                         .repeat(target[sys_index].shape, axis=0)) ** 2) * (
                    self.f_term.D.reshape(1, -1).repeat(target[sys_index].shape, axis=0)))
            X *= self.sys[sys_index].dG
            self.f_term.sx2 = ((X * X).reshape((-1, 1)).repeat(self.f_term.n_bfs, axis=1) * PSI).sum(
                axis=0)  # 拟合高斯基函数权重（局部加权回归方法）
            self.f_term.sxtd = ((X * F_target).reshape((-1, 1)).repeat(self.f_term.n_bfs, axis=1) * PSI).sum(axis=0)
            self.weight[sys_index] = self.f_term.sxtd / (self.f_term.sx2 + 1e-10)
            # self.weight[sys_index] = self.f_term.weight  # fit the weight

    def calc_g(self):
        g = np.zeros([self.n_dmps, self.n_bfs])
        for sys_index in range(self.n_dmps):
            g[sys_index] = self.f_term.calc_g(self.sys[sys_index].dG)
        return g

    def set_weight(self, weight, sys_index):
        self.weight = weight
        self.f_term.set_weight(weight)


if __name__ == "__main__":
    t = np.arange(0, 1.0, 0.01)
    x_trac = np.zeros([2, len(t)])
    x_trac[0] = 2 * np.cos(2 * t)
    x_trac[1] = 2 * np.sin(2 * t)
    xd = np.zeros([2, len(t)])
    xd[0] = np.append(0, np.diff(x_trac[0]))
    xd[1] = np.append(0, np.diff(x_trac[1]))
    xdd = np.zeros([2, len(t)])
    xdd[0] = np.append(0, np.diff(xd[0]))
    xdd[1] = np.append(0, np.diff(xd[1]))
    dmps = DMPs(start=x_trac[:,0], goal=x_trac[:,-1], tau=1, n_bfs=500, end_dmp_time=1.0, end_time=1.2, n_dmps=2)
    dmps.run_fit_trajectory(x_trac, xd, xdd)
    weight = dmps.weight
    y, yd, ydd, t = dmps.run_trajectory()
    print(weight)
    plt.plot(t[0], y[0])
    x_t = np.arange(0, 1.0, 0.01)
    plt.plot(x_t, x_trac[0])
    # dmps.f_term.show()

    t = np.arange(0, 1.0, 0.01)
    x1 = np.sqrt(2 * t * 1 - t*t)
    t2 = np.arange(0, 1.5, 0.01)
    x2 = 1.5 * np.sqrt(2*2/3*t2 - (2.0/3)**2 * t2 *t2)
    x3 = np.sqrt(2 * t2 * 2 - t2*t2)
    plt.plot(t, x1, 'b')
    plt.plot(t2, x2, 'r')
    plt.plot(t2, x3, 'g')
    plt.show()