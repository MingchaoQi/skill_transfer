一、使用强化学习算法实现2D mountaincar小车自适应翻沟
1、DQN方法。详见DQN.py。
2、rule based PGPE方法。方法详见“PGPE强化学习mountaincar总结.docx”；代码可见main.py。
3、DQN+PPR方法。简而言之就是在DQN的训练过程中加入了人为动作策略指导，这里采用的是和rule based PGPE中一样的规则式动作策略。PPR的具体方法可见“Persistent Rule-based Interactive Reinforcement Learning”。

二、3D mountaincar小车环境的配置
1、编写“MountainCar3DEnv”,在“gym/envs/classic_control”文件夹中添加“mountain_car_3D.py”。此文件是在"mountain_car.py"的基础上修改的，也附上了0.21版gym中的"mountain_car.py"文件，使用二维平面表示三维环境，如果gym版本更新可自行在源代码中修改。
2、对应在“envs/_init_.py”和"classic_control/_init_.py"中添加相应的环境，仿照原有语句进行添加即可。
3、注意在“mountain_car_3D.py”中，添加了函数“reset_manual”，在"mountain_car.py"中也做相应的修改。初次调用此函数会有问题，具体问题忘了，但是不复杂，相应解决就好。

三、CBR实现2D到3D环境技能的迁移
1、demo中的流程是使用rule based PGPE方法，具体过程详见“CBR+PGPE仿真报告”。目前来看这个方法还是存在一些问题的，不容易说清楚，但是案例库的建立和相似度的比较还是有参考价值的。
2、代码见“test.py”，也可跟着“mountaincar交互界面控制手册”的步骤使用更完善和直观的“GUI_design.py”。
3、"test_contrast.py"是使用rule based PGPE直接解决3D mountaincar问题。