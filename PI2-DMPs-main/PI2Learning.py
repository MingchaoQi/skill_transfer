import numpy as np
from dmp import DMPs, NonlinearTerm
import functools
import matplotlib.pyplot as plt


class Trajectory:
    """
    可以修改，使其包含多维DMP，还需要对应的生成奖励
    """

    def __init__(self, n_dmps=1, start_time=0.0, end_time=2.0, start=np.array([0]), goal=np.array([1]),
                 start_dmp_time=0.0, end_dmp_time=1.0, n_bfs=10, tau=1,
                 dt=0.01):
        self.dmps_systems = DMPs(start_time=start_time, end_time=end_time, start_dmp_time=start_dmp_time,
                                 end_dmp_time=end_dmp_time, n_bfs=n_bfs, tau=tau, start=start, goal=goal, n_dmps=n_dmps)

        self.n_bfs = n_bfs
        self.length = int((end_time - start_time) / dt)  #
        self.n_dmps = n_dmps
        self.start_dmp_index = int((start_dmp_time - start_time) / dt)
        self.end_dmp_index = int((end_dmp_time - start_time) / dt)
        self.y = np.zeros([n_dmps, self.length])  #
        self.yd = np.zeros([n_dmps, self.length])  #
        self.ydd = np.zeros([n_dmps, self.length])  #
        self.t = np.zeros([n_dmps, self.length])  #
        self.x = np.zeros([n_dmps, self.length])  #
        self.mean_canonical = self.dmps_systems.f_term.mean_canonical  #
        self.weight = np.zeros([n_dmps, n_bfs])  # 基础权重，对一个rollout是一样的
        self.eps = np.zeros([n_dmps, self.length, n_bfs])  # 实际权重为基础权重+eps(every timestep)
        self.psi = np.zeros([n_dmps, self.length, n_bfs])  #
        self.g_term = np.zeros([n_dmps, self.length, n_bfs])  # [g(t)]_j
        self.r_t = np.zeros([self.n_dmps, self.length])  #
        self.r_end = np.zeros(n_dmps)  # 终止奖励（代价）

    def log_step(self, tick, sys_index):  # every timestep's position
        self.y[sys_index, tick] = self.dmps_systems.y[sys_index]
        self.yd[sys_index, tick] = self.dmps_systems.yd[sys_index]
        self.ydd[sys_index, tick] = self.dmps_systems.ydd[sys_index]
        self.t[sys_index, tick] = self.dmps_systems.t[sys_index]
        self.x[sys_index, tick] = self.dmps_systems.f_term.x
        self.psi[sys_index, tick] = self.dmps_systems.f_term.get_psi()
        self.g_term[sys_index, tick] = self.dmps_systems.calc_g()[sys_index]

    def run_step(self, tick, sys_index):
        self.dmps_systems.f_term.weight = self.weight[sys_index]
        self.dmps_systems.run_step(has_dmp=(self.start_dmp_index <= tick < self.end_dmp_index), sys_index=sys_index)

    def calc_cost(self):
        Q = 100
        R = 1
        for sys_index in range(self.n_dmps):
            for i in range(self.start_dmp_index, self.end_dmp_index):
                Meps = self.g_term[sys_index, i].dot(self.eps[sys_index, i]) * self.g_term[sys_index, i] / (
                        np.linalg.norm(self.g_term[sys_index, i]) ** 2)  # M(t)*eps(t)
                norm_term = 0.5 * R * np.linalg.norm(self.weight[sys_index] + Meps) ** 2  # 正则项
                self.r_t[sys_index][i] += 0.5 * self.ydd[sys_index, i] ** 2 * Q + norm_term
                if i == 30 and self.y[sys_index, i] != 0.1:  # 在0.3秒是经过0.1值
                    self.r_t[sys_index][i] += 1e10 * (self.y[sys_index, i] - 0.1) ** 2
            self.r_end[sys_index] += 0.5 * (self.yd[sys_index, self.end_dmp_index - 1]) ** 2 * 1000 + 0.5 * (
                    self.y[sys_index, self.end_dmp_index - 1] - 1) ** 2 * 1000
        return self.r_t, self.r_end

    def calc_cost_traj(self, x, xd):
        Q = 1000
        R = 10
        for sys_index in range(self.n_dmps):
            for i in range(self.start_dmp_index, self.end_dmp_index):
                Meps = self.g_term[sys_index, i].dot(self.eps[sys_index, i]) * self.g_term[sys_index, i] / (
                        np.linalg.norm(self.g_term[sys_index, i]) ** 2)  # M(t)*eps(t)
                # norm_term = 0.5 * R * np.linalg.norm(self.weight[sys_index] + Meps) ** 2  # 正则项
                norm_term = 0.0
                self.r_t[sys_index][i] += 0.5 * (self.y[sys_index, i] - x[sys_index, i]) ** 2 * Q + 0.5 * (
                        self.yd[sys_index, i] - xd[sys_index, i]) ** 2 * R + norm_term
            self.r_end[sys_index] += 0.5 * (self.yd[sys_index, self.end_dmp_index - 1] - xd[
                sys_index, self.end_dmp_index - 1]) ** 2 * R + 0.5 * (
                                             self.y[sys_index, self.end_dmp_index - 1] - x[
                                         sys_index, self.end_dmp_index - 1]) ** 2 * Q
        return self.r_t, self.r_end


def cmp(self: Trajectory, other: Trajectory):
    if self.r_t[0].sum() + self.r_end[0] > other.r_t[0].sum() + other.r_end[0]:
        return 1
    elif self.r_t[0].sum() + self.r_end[0] == other.r_t[0].sum() + other.r_end[0]:
        return 0
    else:
        return -1


class ReplayBuffer:
    def __init__(self, size=10, n_reuse=5):  # 对于样例任务，n_reuse会导致buffer前几个锁定且更新占优，就会使dtheta变成非零常数
        self.buffer = []
        self.size = size
        self.n_reuse = n_reuse

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index) -> Trajectory:
        return self.buffer[index]

    def append(self, traj: Trajectory):
        self.buffer.append(traj)
        if len(self.buffer) >= self.size:
            return 1  # 需要pop
        else:
            return 0

    def pop(self):
        for i in range(self.size - self.n_reuse):
            self.buffer.pop()

    def sort(self):
        self.buffer.sort(key=functools.cmp_to_key(cmp))


class PI2LearningPer:
    def __init__(self, n_dmps=1, num_basic_function=500, n_updates=600, start_pos=np.array([0]), goal_pos=np.array([1]),
                 trace=np.array([0]), trace_d=np.array([0]), start_time=0.0, end_time=1.0,
                 start_dmp_time=0.0, end_dmp_time=1.0, std=40, dt=0.01):
        """
        只将噪声添加到强度最大的基函数
        """
        self.n_dmps = n_dmps  # 维数
        self.n_bfs = num_basic_function
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.start_time = start_time
        self.end_time = end_time
        self.start_dmp_time = start_dmp_time
        self.end_dmp_time = end_dmp_time
        self.start_dmp_index = int((start_dmp_time - start_time) / dt)
        self.end_dmp_index = int((end_dmp_time - start_time) / dt)
        self.std = std
        self.repetitions = 30
        self.updates = n_updates  ##
        self.n_reuse = 0
        self.dt = dt
        self.trace = trace
        self.trace_d = trace_d

        self.weight = np.zeros([n_dmps, self.n_bfs])
        self.length = int((self.end_time - self.start_time) / self.dt)
        self.dmp_length = int((self.end_dmp_time - self.start_dmp_time) / self.dt)  # 仿真离散长度
        self.R = np.zeros([self.dmp_length, self.repetitions])
        self.S = np.zeros([self.dmp_length, self.repetitions])
        self.P = np.zeros([self.n_dmps, self.dmp_length, self.repetitions])
        self.buffer = ReplayBuffer(size=self.repetitions, n_reuse=self.n_reuse)

    def run(self):
        for i in range(self.updates):
            if i % 10 == 0:
                traj_eval = self.rollout(0)  # noise = 0
                print(traj_eval.r_t.sum(1) + traj_eval.r_end)  ###
            if i == self.updates - 1:  # finish the update
                traj_eval = self.rollout(0)
                self.traj = traj_eval
                print(self.weight)
                # print(traj_eval.y[0, 30])
                # plt.plot(traj_eval.t[0], traj_eval.y[0])
                # plt.plot(traj_eval.t[1], traj_eval.y[1])
                plt.show()
            noise_gain = max((self.updates - i) / self.updates, 0.1)

            while 1:
                flag = self.buffer.append(self.rollout(noise_gain))  # 循环size=10次
                if flag:
                    break

            self.pi2_update(10)  # update the weight
            self.buffer.sort()
            self.buffer.pop()

    def rollout(self, noise_gain):  # run one rollout
        std_eps = noise_gain * self.std
        traj = Trajectory(n_dmps=self.n_dmps, dt=self.dt, n_bfs=self.n_bfs, start=self.start_pos, goal=self.goal_pos,
                          end_dmp_time=self.end_dmp_time, end_time=self.end_time)
        last_index = -1  # time是同时的，所以会同时切换
        EPS = np.zeros([self.n_dmps, self.n_bfs])
        for sys_index in range(self.n_dmps):
            traj.dmps_systems.f_term = NonlinearTerm(start_time=self.start_dmp_time, end_time=self.end_dmp_time,
                                                     tau=1, n_bfs=self.n_bfs)
            for t in range(self.length):
                traj.log_step(t, sys_index)  # every timestep's position
                index = traj.psi[sys_index, t].argmax()  # choose the max psi###
                if index != last_index:  # 切换了activate
                    EPS = np.zeros([self.n_dmps, self.n_bfs])
                    last_index = index
                    eps = np.random.normal(loc=0.0, scale=std_eps)  # 仅扰动当前时间activate的base function
                    EPS[sys_index, index] = eps
                    traj.dmps_systems.weight[sys_index] = self.weight[sys_index] + EPS[sys_index]
                    traj.weight = traj.dmps_systems.weight
                    traj.eps[sys_index, t, :] = EPS[sys_index]  # noise!=0
                else:
                    traj.eps[sys_index, t, :] = EPS[sys_index]  # noise=0
                traj.run_step(t, sys_index)  # run a roolout
        # traj.calc_cost()  # calculate the cost
        traj.calc_cost_traj(self.trace, self.trace_d)
        return traj

    def pi2_update(self, h=10):
        for sys_index in range(self.n_dmps):
            self.R = np.zeros([self.dmp_length, self.repetitions])
            self.S = np.zeros([self.dmp_length, self.repetitions])
            expS = self.S
            for m in range(self.repetitions):
                self.R[:, m] = self.buffer[m].r_t[sys_index][self.start_dmp_index:self.end_dmp_index]
                self.R[:, m][-1] += self.buffer[m].r_end[sys_index]  # 末奖励
            self.S = np.flip(np.flip(self.R).cumsum(0))  # theta+Meps项包含在奖励中
            maxS = self.S.max(1).reshape(-1, 1)
            minS = self.S.min(1).reshape(-1, 1)
            # for m in range(self.repetitions):
            #     expS[:,m] = np.exp(-h * (self.S[:,m] - minS) / (maxS - minS))
            expS = np.exp(-h * (self.S - minS) / (maxS - minS))
            self.P[sys_index] = expS / expS.sum(1).reshape(-1, 1)
        PMeps = np.zeros([self.n_dmps, self.repetitions, self.dmp_length, self.n_bfs])
        dtheta_new = np.zeros([self.n_dmps, self.n_bfs])
        for sys_index in range(self.n_dmps):
            for m in range(self.repetitions):
                traj = self.buffer[m]
                gTeps = (traj.g_term[sys_index, self.start_dmp_index:self.end_dmp_index] * traj.eps[sys_index,
                                                                                           self.start_dmp_index:self.end_dmp_index]).sum(
                    1)
                gTg = (traj.g_term[sys_index, self.start_dmp_index:self.end_dmp_index] ** 2).sum(1)
                PMeps[sys_index, m] = traj.g_term[sys_index, self.start_dmp_index:self.end_dmp_index] * (
                    (self.P[sys_index][:, m] * gTeps / (gTg + 1e-10)).reshape(-1, 1))
        dtheta = PMeps.sum(1)
        traj = self.buffer[0]
        for sys_index in range(self.n_dmps):
            N = np.linspace(self.dmp_length, 1, self.dmp_length)
            W = N.reshape(-1, 1) * traj.psi[sys_index, self.start_dmp_index:self.end_dmp_index]
            W = W / W.sum(0)
            dtheta_new[sys_index] = (W * dtheta[sys_index]).sum(0)
        # for sys_index in range(self.n_dmps):
        #     dtheta = (W * dtheta[sys_index]).sum(0)
        self.weight += dtheta_new
        # print('dtheta=', dtheta_new)

    def set_weight(self, weight):
        self.weight = weight


if __name__ == "__main__":
    t = np.arange(0, 1.0, 0.01)
    x_trac = np.zeros([3, len(t)])
    x_trac[0] = 2 * np.cos(2 * t)
    x_trac[1] = 2 * np.sin(2 * t)
    x_trac[2] = np.exp(2*t)
    xd = np.zeros([3, len(t)])
    xd[0] = np.append(0, np.diff(x_trac[0]))
    xd[1] = np.append(0, np.diff(x_trac[1]))
    xd[2] = np.append(0, np.diff(x_trac[2]))
    xdd = np.zeros([3, len(t)])
    xdd[0] = np.append(0, np.diff(xd[0]))
    xdd[1] = np.append(0, np.diff(xd[1]))
    xdd[2] = np.append(0, np.diff(xd[2]))

    dmps = DMPs(start=x_trac[:,0], goal=x_trac[:,-1], tau=1, n_bfs=500, end_dmp_time=1.0, end_time=1.2, n_dmps=len(x_trac))
    dmps.run_fit_trajectory(x_trac, xd, xdd)
    weight = dmps.weight
    y, yd, ydd, t1 = dmps.run_trajectory()

    learn = PI2LearningPer(n_dmps=len(x_trac), start_pos=x_trac[:,0], goal_pos=x_trac[:,-1], trace=x_trac, trace_d=xd,
                           num_basic_function=500, end_time=1.2)
    learn.set_weight(weight)
    learn.run()

    plt.plot(t, x_trac[0], 'blue')
    plt.plot(learn.traj.t[0], learn.traj.y[0], 'red')
    plt.figure()
    plt.plot(t, x_trac[1], 'blue')
    plt.plot(learn.traj.t[1], learn.traj.y[1], 'red')
    plt.figure()
    plt.plot(t, x_trac[2], 'blue')
    plt.plot(learn.traj.t[2], learn.traj.y[2], 'red')
    plt.figure()

    # traj = learn.rollout(0)
    # plt.plot(traj.t[0], traj.y[0])
