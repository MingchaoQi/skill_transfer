from py2neo import Node, Relationship, Graph

skill_graph = Graph("bolt://localhost:7687", auth=('neo4j', 'Qi121220'))

object_1 = Node('object', name='cabinet', pos='0.3,0.4,0', quat='1,0,0,0')
object_1_1 = Node('sub_object', name='base', mass='0.35')
object_1_2 = Node('sub_sub_object', name='drawer link', mass='7.85398', diaginertia="0.923301 0.764585 0.168533")
object_1_3 = Node('sub_sun_sub_object', name='cabinet handle', mass='0.18', size='0.02')
skill_graph.create(object_1)
skill_graph.create(object_1_1)
skill_graph.create(object_1_2)
skill_graph.create(object_1_3)

relationship2_1 = Relationship(object_1_1, 'relative_position', object_1, connection_type='attach', pos='0 0 0 0',
                               quat='1 0 0 0,', inertial_pos='0 0 0 0', inertial_quat='1 0 0 0')
relationship2_2 = Relationship(object_1_2, 'DOF and relative position', object_1_1, connection_type='joint',
                               joint_type='slide',
                               joint_range='-0.1 0', joint_axis='0 1 0', joint_pos='0 0 0',
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
                               joint_range="0.0 0.4", joint_axis="0 0 1", joint_pos='0.255 0 0',
                               joint_damping='1', pos='0.3 0 0', quat='1 0 0 0', inertial_pos='0.0296816 -0.00152345 0',
                               inertial_quat='0.701072 0 0 0.713091')
relationship3_3 = Relationship(object_2_3, 'relative position', object_2_2, pos='-0.175 0 -0.025', quat='1 0 0 0',
                               inertial_pos='-0.017762 0.0138544 0',
                               inertial_quat='0.365653 0.605347 -0.36522 0.605365')
relationship3_4 = Relationship(object_2_4, 'relative position', object_2_3, pos='0.125 -0.10 0', quat='1 0 0 0',
                               inertial_pos='0 0 0 0', inertial_quat='1 0 0 0')



skill_graph.create(relationship3_1)
skill_graph.create(relationship3_2)
skill_graph.create(relationship3_3)
skill_graph.create(relationship3_4)
