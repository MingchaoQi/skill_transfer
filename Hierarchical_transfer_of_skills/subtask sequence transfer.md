# Task-level subtask sequence transfer
Subtask sequences planned in similar task scenarios are always akin, such as in the tasks of opening doors and drawers. To enable the transfermof task-level subtask sequences across similar scenarios, it is essential for the agent to have a comprehensive understanding of the transfer tasks and scenarios. 

In this context, we introduce LLM, leveraging its powerful reasoning and generalization capabilities to achieve the aforementioned goals. This paper is based on the contextual learning and thoughtchain prompts in prompt-based learning, constructing a new four-stage prompt word paradigm.

**First, we provide the LLM with a comprehensive description of the complex planning task and the state graph descriptions from the skill library.**

```
I will give you a manipulation task description and I need you help me to finish the task <br> planning. The task description consists of three parts: The main task, the reference task, and the scene description. 

The main task and the reference task have the same structure, the reference task's planning has been known while the main task's planning is not known. Each task consists of several sub-tasks, and each sub-task consists of three parts: "action", "require_states" and "obtain_states". "require_states" are initial states before"action" and  "obtain_states" are final states after "action". By adopting this structure, task planning can reflect changes in the scene. 

The scene description and task planning are knowledge graphs written by py2neo and the original XML format, which include all needed knowledge to be used in task planning, such as objects and their parts to be manipulated, and the task planning should use the object in the scene description.

The aim of task planning is that when the reference task and the scene description have been known, and I give a new task named the main task, you can give me a reasonable task planning result.

Do you understand? If yes, re-say what I mean detailedly.
```
**Next, we enable the LLM to comprehend the entire knowledge graph of the operations skill library. We input the skill library into the LLM as Neo4j-based code.**

```
## The Python code about knowledge graph of scene description
object_1 = Node('object', name='cabinet')
object_1_1 = Node('sub_object', name='base', mass='0.35')
object_1_2 = Node('sub_sub_object', name='drawer_link', mass='7.85398', diaginertia="0.923301 0.764585 0.168533")
object_1_3 = Node('sub_sub_sub_object', name='cabinet_handle', mass='0.18')
skill_graph.create(object_1)
skill_graph.create(object_1_1)
skill_graph.create(object_1_2)
skill_graph.create(object_1_3)

relationship2_1 = Relationship(object_1_1, 'part', object_1)
relationship3_1 = Relationship(object_1_2, 'part', object_1)
relationship4_1 = Relationship(object_1_3, 'part', object_1)
relationship2_2 = Relationship(object_1_2, 'relative position', object_1_1, connection_type='joint',
                               joint_type='slide',
                               joint_range='-0.1 0', joint_axis='0 1 0', joint_pos='0 0 0',
                               joint_damping='100.0', pos='0 -0.01 0.076', quat='1 0 0 0', inertial_pos='0 0 0.35',
                               inertial_quat='0.5 0.5 0.5 0.5')
relationship2_3 = Relationship(object_1_3, 'relative position', object_1_2, connection_type='attach', pos='0 -0.16 0.04', quat='1 0 0 0',
                               inertial_pos='0 0 0 0', inertial_quat='1 0 0 0')
skill_graph.create(relationship2_1)
skill_graph.create(relationship3_1)
skill_graph.create(relationship4_1)
skill_graph.create(relationship2_2)
skill_graph.create(relationship2_3)

object_2 = Node('object', name='door_lock')
object_2_1 = Node('sub_object', name='frame', mass='7.85398')
object_2_2 = Node('sub_sub_object', name='door', mass='2.43455', diaginertia='0.0913751 0.0521615 0.043714')
object_2_3 = Node('sub_sub_sub_object', name='latch', mass='0.1', diaginertia='0.0483771 0.0410001 0.0111013')
skill_graph.create(object_2)
skill_graph.create(object_2_1)
skill_graph.create(object_2_2)
skill_graph.create(object_2_3)

relationship2_1 = Relationship(object_2_1, 'part', object_2)
relationship3_1 = Relationship(object_2_2, 'part', object_2)
relationship4_1 = Relationship(object_2_3, 'part', object_2)
relationship3_2 = Relationship(object_2_2, 'relative position', object_2_1, connection_type='joint',
                               joint_type='hinge',
                               joint_range="0.0 0.4", joint_axis="0 0 1", joint_pos='0.255 0 0',
                               joint_damping='1', pos='0.3 0 0', quat='1 0 0 0', inertial_pos='0.0296816 -0.00152345 0',
                               inertial_quat='0.701072 0 0 0.713091')
relationship3_3 = Relationship(object_2_3, 'relative position', object_2_2, connection_type='attach', pos='-0.175 0 -0.025', quat='1 0 0 0',
                               inertial_pos='-0.017762 0.0138544 0',
                               inertial_quat='0.365653 0.605347 -0.36522 0.605365')
skill_graph.create(relationship2_1)
skill_graph.create(relationship3_1)
skill_graph.create(relationship4_1)
skill_graph.create(relationship3_2)
skill_graph.create(relationship3_3)
```
**In the third step, we guide the LLM to deeply understand the spatial semantic information in the scene graph of the skill library.**

```
Before starting the task planning, I would like to give you more information and rules about the task planning description. 
As for the relationships, we assume that there exists five connection types: attach, joint, in, on, and around. The default connection type is "attach". In the task planning, we use the "object -> connection types -> object" to describe this relationship. We use the form: "object(property)" to present the object and its properties, and as for relationships, we use the form "connection type(property)" to present the relationship and its property.
for example:
base(mass='0.35')->joint(joint_damping='100.0')->drawer link(mass='7.85')

If you understand, use the form "object(property) -> connection types(property) -> object(property)" to present the objects/sub_objects and their properties.
```
**Finally, we provide reference planning for existing complex tasks and subtask sequence migration requirements for new scenes.**

```
Great! Now I will give you the reference task and its planning.
reference task: "open the cabinet", the target object is "cabinet".
task planning:
{"task": open  the cabinet}
{"sub_task":1, "action": "move", "require_states": gripper(open), "obtain_states":gripper(open)-> around -> cabinet handle}
{"sub_task":2, "action":"pick", "require_states": gripper(open) -> around -> cabinet handle, "obtain_states": gripper(closed) -> around -> cabinet handle}
{"sub_task":3, "action":"move", "require_states": gripper(closed) -> around -> Cabinet Handle && base -> joint(joint_range=0) -> drawer link, "obtain_states":gripper(closed) -> around -> Cabinet Handle and base -> joint(joint_range=-0.1) -> drawer link}
{"sub_task":4, "action":"drop", "require_states": gripper(closed), "obtain_states": gripper(open)}
Notes: The "gripper" is the manipulator, which can be seen as an object. It has two states: open and closed. sub-task 1~4 is the order of the sub-task.
In task planning, the "action" can be chosen from "move", "pick" and "drop", and the sub_objects that exist in the task planning should be part of the target object.
The notes should be followed in the task planning.
My question is: the main task is "open the door lock", and the target object is "door lock". Give me the task planning of the main task based on the knowledge of the reference task and scene description.
```