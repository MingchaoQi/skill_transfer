import numpy as np
import math

class controller:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def PD_control(self, last_pos, cur_pos, goal_pos, goal_vel):  # PD控制生成控制指令
        cur_vel = (cur_pos - last_pos) * 20
        a = self.kp * (goal_pos - cur_pos) + self.kd * (goal_vel - cur_vel)
        return a

    def vel_control(self, cur_pos, goal_pos, freq):
        a = self.kp * (goal_pos - cur_pos)

        return a

