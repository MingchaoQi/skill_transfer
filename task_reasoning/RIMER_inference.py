import random
import numpy as np
import matplotlib.pyplot as plt
import math
'''
对于如何添加同一环境状态的新的参数值尚不明确
规则只考虑全为“且”的情况
'''

def obs_build(observation, new_state): #更新环境状态库，添加新的环境状态与状态值集合
    observation.update(new_state)

    return observation

def traj_build(traj_base, new_traj): #更新输出轨迹库，添加新的轨迹名称与轨迹点集合
    traj_base.update(new_traj)

    return traj_base

def rule_build(state, obs, in_importance, theta, beta, traj): #构建新规则，包括每个环境状态的相对重要程度、输出的每个结论的初始置信度、此条规则的相对激活权重
    if len(state) == len(obs) and len(beta) == len(traj):
        rule = {
            'state': state,
            'attribute_weights': in_importance,
            'relative_weight': theta,
            'output_weights': beta
        }
    else:
        a = np.zeros(len(obs) - len(state))
        b = np.zeros(len(traj) - len(beta))
        state1 =np.append(state, a)
        in_importance1 = np.append(in_importance, a)
        beta1 = np.append(beta, b)
        rule = {
            'state': state1,
            'attribute_weights': in_importance1,
            'relative_weight': theta,
            'output_weights': beta1
        }

    return rule

def input_recongition(A, transfrom): #辨识输入的环境与现有环境状态分量的每个参数的相似度，当前环境的状态必须要在现有参数的范围内（不包括端点）
    O1 = np.zeros(len(transfrom['material']))
    O2 = np.zeros(len(transfrom['object']))
    O3 = np.zeros(len(transfrom['manipulator']))
    O4 = np.zeros(len(transfrom['relative_pos']))
    O = np.array([O1, O2, O3, O4])

    for i in range(len(A)):
        a = A[i]
        a_min = math.floor(a)
        a_max = math.ceil(a)
        O_min = (a_max - a) / 1
        O_max = (a - a_min) / 1
        O[i][a_min - 1] = O_min
        O[i][a_max - 1] = O_max

    return O

def rule_inference(rule_base, sim_base): #输入现有环境状态分量的每个参数的相似度，经过计算得到每个输出结论的相对置信度
    alpha = np.ones(len(rule_base))
    w = np.ones(len(rule_base))
    beta = np.zeros((len(rule_base), len(rule_base[0]['output_weights'])))

    for k in range(len(rule_base)):
        rule = rule_base[k]
        attribute = rule['state']
        delta = rule['attribute_weights']
        theta = rule['relative_weight']
        beta[k] = rule['output_weights']
        alpha[k] = 1
        sum_tau, b = 0, 0
        for i in range(len(attribute)):
            if attribute[i] == 0:
                alpha[k] = alpha[k] * 1
            else:
                alpha[k] = alpha[k] * (sim_base[i][attribute[i] - 1] ** delta[i])
                sum_tau += np.sum(sim_base[i])
                b = b+1
        sum_tau = sum_tau / b
        w[k] = theta * alpha[k]
        beta[k] = beta[k] * sum_tau
    sum_w = np.sum(w)
    if sum_w != 0:
        w = w / sum_w

    m = np.ones((len(rule_base), len(rule_base[0]['output_weights'])))
    m_D = np.ones(len(rule_base))
    m_D_ = np.ones(len(rule_base))
    m_D__ = np.ones(len(rule_base))
    for k in range(len(rule_base)):
        m[k] = w[k] * beta[k]
        m_D[k] = 1 - np.sum(m[k])
        m_D_[k] = 1 - w[k]
        m_D__[k] = w[k] * (1 - np.sum(m[k]))

    m_I = np.ones((len(rule_base), len(rule_base[0]['output_weights'])))
    m_D_I = np.ones(len(rule_base))
    m_D_I_ = np.ones(len(rule_base))
    m_D_I__ = np.ones(len(rule_base))
    m_I[0] = m[0]
    m_D_I[0] = m_D[0]
    m_D_I_[0] = m_D_[0]
    m_D_I__[0] = m_D__[0]
    for k in range(len(rule_base) - 1):
        c = 0
        for j in range(len(rule_base[0]['output_weights'])):
            c = c + m_I[k][j] * (np.sum(m[k+1]) - m[k+1][j])
        K = 1.0 / (1 - c)
        for j in range(len(rule_base[0]['output_weights'])):
            m_I[k+1][j] = K * (m_I[k][j] * m[k+1][j] + m_I[k][j] * m_D[k+1] + m_D_I[k] * m[k+1][j])
        m_D_I_[k+1] = K * (m_D_I_[k] * m_D_[k+1])
        m_D_I__[k+1] = K * (m_D_I__[k] * m_D__[k+1] + m_D_I__[k] * m_D_[k+1] + m_D_I_[k] * m_D__[k+1])
        m_D_I[k+1] = m_D_I_[k+1] + m_D_I__[k+1]

    if m_D_I_[-1] != 1:
        Beta = m_I[-1] / (1 - m_D_I_[-1])
        Beta_D = m_D_I__[-1] / (1 - m_D_I_[-1])
    else:
        Beta, Beta_D = None, None

    return Beta, Beta_D

if __name__ == '__main__':
    ##构建观察到的环境状态库
    obs1 = np.array([1,2,3,4,5])
    obs2 = np.array([1,2,3,4,5,6])
    obs3 = np.array([1,2,3,4])
    obs4 = np.array([1,2,3])
    obs = {
        'material':obs1.copy(),
        'object': obs2.copy(),
        'manipulator': obs3.copy(),
        'relative_pos': obs4.copy(),
    }
    ##构建输出的结论（轨迹）
    traj1 = np.array([[1,1,1], [2,2,2], [3,3,3]])
    traj2 = np.array([[0,0,0], [1,1,1], [4,4,4]])
    traj3 = np.array([[1,1,1]])
    traj = {
        'traj_free': 'minimum_jerk_traj',
        'traj1': traj1.copy(),
        'traj2': traj2.copy(),
        'traj3': traj3.copy(),
    }
    ##构建一条规则
    state = np.array([1,1,2,3])
    in_importance = np.array([1,1,0.8,0.9]) #相当于delta_i
    theta = 1
    beta = np.array([0, 0, 0.9, 0.1])
    rule_1 = rule_build(state, obs, in_importance, theta, beta, traj)
    ##构建第二条规则
    state = np.array([2, 5, 2, 1])
    in_importance = np.array([0.8, 1, 0.9, 0.9])  # 相当于delta_i（必须有至少一个为1）
    theta = 1
    beta = np.array([0.9, 0, 0.1, 0])
    rule_2 = rule_build(state, obs, in_importance, theta, beta, traj)
    ##构建第三条规则
    state = np.array([3, 1, 2, 3])
    in_importance = np.array([1, 1, 0.9, 0.9])  # 相当于delta_i
    theta = 1
    beta = np.array([0, 0.8, 0, 0.2])
    rule_3 = rule_build(state, obs, in_importance, theta, beta, traj)
    ##输入一组数据，输出与环境状态的相似度
    input = np.array([1.1, 1.05, 1.9, 2.99])
    output = input_recongition(input, obs) #相当于输出的是alpha_(i,j)
    ##规则进行组合
    Rule = [rule_1, rule_2, rule_3]
    Beta, Beta_D = rule_inference(Rule, output)



