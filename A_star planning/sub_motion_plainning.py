from py2neo import Node, Relationship, Graph, Path, Subgraph
from py2neo import NodeMatcher, RelationshipMatcher
import numpy as np
import math
import scipy.linalg as linalg

# 从知识图谱中检索关节的自由度及相关信息
skill_graph = Graph("bolt://localhost:7687", auth=('neo4j', 'Qi121220'))

relationship_matcher = RelationshipMatcher(skill_graph)
r = relationship_matcher.match(None, r_type='DOF and relative position').all()

connection_type = dict()
joint_type = dict()
joint_pos = dict()
joint_axis = dict()
joint_range = dict()
joint_damping = dict()
connection_type["cabinet"] = r[0].get('connection_type')
joint_type["cabinet"] = r[0].get('joint_type')
joint_pos["cabinet"] = r[0].get('joint_pos')
joint_axis["cabinet"] = r[0].get('joint_axis')
joint_range["cabinet"] = r[0].get('joint_range')
joint_damping["cabinet"] = r[0].get('joint_damping')

connection_type["door"] = r[1].get('connection_type')
joint_type["door"] = r[1].get('joint_type')
joint_pos["door"] = r[1].get('joint_pos')
joint_axis["door"] = r[1].get('joint_axis')
joint_range["door"] = r[1].get('joint_range')
joint_damping["door"] = r[1].get('joint_damping')

def rotate_mat(axis, radian):
    """
    利用旋转矩阵并结合知识图谱中提取的相关信息，计算门把手的最终运动位置
    旋转矩阵 欧拉角
    """

    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix

# 分别是x,y和z轴以及旋转轴
axis_x, axis_y, axis_z = [1,0,0], [0,1,0], [0, 0, 1]
# 使用列表形式存放数据，可避免数据转换的麻烦
rand_axis = [0,0,1]
#旋转角度为joint_range
yaw = joint_range["door"][1]
#返回旋转矩阵
rot_matrix = rotate_mat(rand_axis, yaw)
print(rot_matrix)
# 计算点绕着轴运动后的点(变换到门坐标系)
door_joint_pos = env.door_joint_pos
x = env._slide_handle_xpos - np.array([door_joint_pos[0],door_joint_pos[1],0])
# 旋转后的坐标，需要再次变换到世界坐标系
x1 = np.dot(rot_matrix,x)
x1 = x1 + np.array([door_joint_pos[0],door_joint_pos[1],0])



