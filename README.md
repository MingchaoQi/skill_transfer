<h1 align="center">
Semantic-Geometric-Physical-Driven Robot Manipulation Skill Transfer
via Skill Library and Tactile Representation<br>
</h1>

Mingchao Qi, Yuanjin Li, Xing Liu<sup>* </sup>, Yizhai Zhang, Pangfeng Huang

**Paper Access:** [📝PDF](https://arxiv.org/pdf/2411.11714) | [arXiv](https://arxiv.org/abs/2411.11714)

**Hardware Support:** 
+ Robot: Kuka iiwa14 
+ Tactile Sensor: GelSight Mini



# 📚 Overview

![](./files/structure.png)
![](./files/experiment.png)

# 🛠️ Installation
This project is implemented based on **Robosuite** in simulation and on the **KUKA iiwa14** and **GelSight Mini** in the real-world environment. Therefore, it is necessary to install [ROS2 Humble](https://github.com/ros2)、[gelsightinc/gsrobotics](https://github.com/gelsightinc/gsrobotics) and [lbr_fri_ros2_stack](https://github.com/lbr-stack/lbr_fri_ros2_stack) prior to running this project to operate the robotic arm and tactile sensor.

Clone this repo and install prerequisites:

```bash
# Clone this repo
git clone https://github.com/MingchaoQi/skill_transfer.git
cd skill_transfer
    
# Create a Conda environment
conda create -n skill_transfer python=3.10
conda activate skill_transfer
    
# Install robosuite and robosuite-task-zoo
cd envs/robosuite-task-zoo
pip install -e .
    
# Install other prequisites
pip install -r requirements.txt
```

# 🧑🏻‍💻 Deployment on Real-Robots
After downloading the files, you need to **compile** and **source** `tac3d` as a ROS2 package. If there are additional dependencies required to run with this package, please adjust the relevant XML files accordingly. Your computer must have a complete **ROS2 Humble** environment installed.
```bash
cd ~/<your_working_layer>
```
```bash
colcon build
```

```bash
source ./install/setup.py
```

After preparing the robotic arm's gripper equipped with the GelSight Mini sensor, you can activate various nodes to monitor and publish the object's pose.

- Activate the robotic arm pose subscription node (based on `LBR`).
```python3
ros2 run tac3d listener
```

- Activate the node that publishes the object's world coordinates (waiting for GelSight initialization is required).
```python3
ros2 run tac3d orientation_publisher
```

- Activate the pose alignment command publishing node.
```python3
ros2 run tac3d pose_planning_node
```

# 👍 Citation
If you find our work helpful, please cite us:

```
@article{qi2024semantic,
  title={Semantic-Geometric-Physical-Driven Robot Manipulation Skill Transfer via Skill Library and Tactile Representation},
  author={Qi, Mingchao and Li, Yuanjin and Liu, Xing and Liu, Zhengxiong and Huang, Panfeng},
  journal={arXiv preprint arXiv:2411.11714},
  year={2024}
}
```

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.
