from py2neo import Node, Relationship, Graph

skill_graph = Graph("bolt://localhost:7687", auth=('neo4j', 'Qi121220'))

task_1 = Node('task', name='open_drawer')
skill_graph.create(task_1)

action_1 = Node('sub_task', name='move to the target', order='0')
action_2 = Node('sub_task', name='pick', order='1')
action_3 = Node('sub_task', name='move', order='2')
action_4 = Node('sub_task', name='drop', order='3')
skill_graph.create(action_1)
skill_graph.create(action_2)
skill_graph.create(action_3)
skill_graph.create(action_4)

relation_1_1 = Relationship(task_1, 'subtask', action_1)
relation_1_2 = Relationship(task_1, 'subtask', action_2)
relation_1_3 = Relationship(task_1, 'subtask', action_3)
relation_1_4 = Relationship(task_1, 'subtask', action_4)
skill_graph.create(relation_1_1)
skill_graph.create(relation_1_2)
skill_graph.create(relation_1_3)
skill_graph.create(relation_1_4)

action1_require = Node('require', name='gripper', state='open', subgraph='action1_require_subgraph')
skill_graph.create(action1_require)

action1_obtain_1 = Node('obtain', name='gripper', state='open', subgraph='action1_obtain_subgraph')
action1_obtain_2 = Node('obtain', name='cabinet handle', subgraph='action1_obtain_subgraph')
relationship_action1_obtain = Relationship(action1_obtain_1, 'around', action1_obtain_2,
                                           subgraph='action1_obtain_subgraph')
skill_graph.create(action1_obtain_1)
skill_graph.create(action1_obtain_2)
skill_graph.create(relationship_action1_obtain)

action2_require_1 = Node('require', name='gripper', state='open', subgraph='action2_require_subgraph')
action2_require_2 = Node('require', name='cabinet handle', subgraph='action2_require_subgraph')
relationship_action2_require = Relationship(action2_require_1, 'around', action2_require_2,
                                            subgraph='action2_require_subgraph')
skill_graph.create(action2_require_1)
skill_graph.create(action2_require_2)
skill_graph.create(relationship_action2_require)

action2_obtain_1 = Node('obtain', name='gripper', state='closed', subgraph='action2_obtain_subgraph')
action2_obtain_2 = Node('obtain', name='cabinet handle', subgraph='action2_obtain_subgraph')
relationship_action2_obtain = Relationship(action2_obtain_1, 'around', action2_obtain_2,
                                           subgraph='action2_obtain_subgraph')
skill_graph.create(action2_obtain_1)
skill_graph.create(action2_obtain_2)
skill_graph.create(relationship_action2_obtain)

action3_require_1 = Node('require', name='gripper', state='closed', subgraph='action3_require_subgraph')
action3_require_2 = Node('require', name='cabinet handle', joint_state='-0.1', subgraph='action3_require_subgraph')
relationship_action3_require = Relationship(action3_require_1, 'around', action3_require_2,
                                            subgraph='action3_require_subgraph')
skill_graph.create(action3_require_1)
skill_graph.create(action3_require_2)
skill_graph.create(relationship_action3_require)

action3_obtain_1 = Node('obtain', name='gripper', state='closed', subgraph='action3_obtain_subgraph')
action3_obtain_2 = Node('obtain', name='cabinet handle', joint_state='0', subgraph='action3_obtain_subgraph')
relationship_action3_obtain = Relationship(action3_obtain_1, 'around', action3_obtain_2,
                                           subgraph='action3_obtain_subgraph')
skill_graph.create(action3_obtain_1)
skill_graph.create(action3_obtain_2)
skill_graph.create(relationship_action3_obtain)

action4_require_1 = Node('require', name='gripper', state='closed', subgraph='action4_require_subgraph')
action4_require_2 = Node('require', name='cabinet handle', subgraph='action4_require_subgraph')
relationship_action4_require = Relationship(action4_require_1, 'around', action4_require_2,
                                            subgraph='action4_require_subgraph')
skill_graph.create(action4_require_1)
skill_graph.create(action4_require_2)
skill_graph.create(relationship_action4_require)

action4_obtain_1 = Node('obtain', name='gripper', state='open', subgraph='action4_obtain_subgraph')
action4_obtain_2 = Node('obtain', name='cabinet handle', subgraph='action4_obtain_subgraph')
relationship_action4_obtain = Relationship(action4_obtain_1, 'around', action4_obtain_2,
                                           subgraph='action4_obtain_subgraph')
skill_graph.create(action4_obtain_1)
skill_graph.create(action4_obtain_2)
skill_graph.create(relationship_action4_obtain)

object_1 = Node('object', name='cabinet', pos='0.3,0.4,0', quat='1,0,0,0')
object_1_1 = Node('sub_object', name='base', mass='0.35')
object_1_2 = Node('sub_sub_object', name='drawer link', mass='7.85398', diaginertia="0.923301 0.764585 0.168533")
object_1_3 = Node('sub_sub_sub_object', name='cabinet handle', mass='0.18', size='0.02')
skill_graph.create(object_1)
skill_graph.create(object_1_1)
skill_graph.create(object_1_2)
skill_graph.create(object_1_3)

relationship2_1 = Relationship(object_1_1, 'relative_position', object_1, connection_type='attach', pos='0 0 0 0',
                               quat='1 0 0 0,', inertial_pos='0 0 0 0', inertial_quat='1 0 0 0')
relationship2_2 = Relationship(object_1_2, 'DOF and relative position', object_1_1, connection_type='joint',
                               joint_type='slide',
                               joint_range='-0.1 0', joint_axis='[0,1,0]', joint_pos='0 0 0',
                               joint_damping='100.0', pos='0 -0.01 0.076', quat='1 0 0 0', inertial_pos='0 0 0.35',
                               inertial_quat='0.5 0.5 0.5 0.5')
relationship2_3 = Relationship(object_1_3, 'relative position', object_1_2, pos='0 -0.16 0.04', quat='1 0 0 0',
                               inertial_pos='0 0 0 0', inertial_quat='1 0 0 0')
skill_graph.create(relationship2_1)
skill_graph.create(relationship2_2)
skill_graph.create(relationship2_3)

object_2 = Node('object', name='door lock', pos='', quat='1 0 0 -1')
object_2_1 = Node('sub_object', name='frame', mass='7.85398')
object_2_2 = Node('sub_sub_object', name='door', mass='2.43455', diaginertia='0.0913751 0.0521615 0.043714')
object_2_3 = Node('sub_sub_sub_object', name='latch', mass='0.1', diaginertia='0.0483771 0.0410001 0.0111013')
object_2_4 = Node('sub_sub_sub_sub_object', name='door handle', size='0.02')
skill_graph.create(object_2)
skill_graph.create(object_2_1)
skill_graph.create(object_2_2)
skill_graph.create(object_2_3)
skill_graph.create(object_2_4)

relationship3_1 = Relationship(object_2_1, 'relative position', object_2, pos='0 0.22 0', quat='0.707388 0 0 -0.706825',
                               inertial_pos='0.3 0 0', inertial_quat='0.5 0.5 0.5 0.5')
relationship3_2 = Relationship(object_2_2, 'DOF and relative position', object_2_1, connection_type='joint',
                               joint_type='hinge',
                               joint_range=[0.0,0.4], joint_axis=[0,0,1], joint_pos=[0.255,0,0],
                               joint_damping=[1], pos=[0.3,0,0], quat=[1,0,0,0], inertial_pos=[0.0296816,-0.00152345,0],
                               inertial_quat=[0.701072,0,0,0.713091])
relationship3_3 = Relationship(object_2_3, 'relative position', object_2_2, pos='-0.175 0 -0.025', quat='1 0 0 0',
                               inertial_pos='-0.017762 0.0138544 0',
                               inertial_quat='0.365653 0.605347 -0.36522 0.605365')
relationship3_4 = Relationship(object_2_4, 'relative position', object_2_3, pos='0.125 -0.10 0', quat='1 0 0 0',
                               inertial_pos='0 0 0 0', inertial_quat='1 0 0 0')

skill_graph.create(relationship3_1)
skill_graph.create(relationship3_2)
skill_graph.create(relationship3_3)
skill_graph.create(relationship3_4)
