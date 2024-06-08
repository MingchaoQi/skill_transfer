# Genetic Algorithms for Mapping Evolution (GAME)
Code for the Genetic Algorithms for Mapping Evolution (GAME) project. Submitted as part of a research project for the Evolutionary Computation and Swarm Intelligence course at Johns Hopkins University during Fall 2022.

The paper has been accepted to the IEEE Congress on Evolutionary Computation (CEC) 2023. 

The paper can be found at this link: TBD.

# Abstract
Recently, there has been a focus on using transfer learning to reduce the sample complexity in reinforcement learning. One component that enables transfer is an intertask mapping that relates a pair of tasks. Automatic methods attempt to learn task relationships either by evaluating all possible mappings in a brute force manner, or by using techniques such as neural networks to represent the mapping. However, brute force methods do not scale well in problems since there is an exponential number of possible mappings, and automatic methods that use complex representations generate mappings that are not always interpretable. In this paper, we describe
a population-based algorithm that generates intertask mappings in a tractable amount of time. The idea is to use an explicit representation of an intertask mapping, and to combine an evolutionary algorithm with an offline evaluation scheme to search for the optimal mapping. Experiments on two transfer learning problems show that our approach is capable of finding highly-fit mappings and searching a space that is infeasible for a brute force approach. Furthermore, agents that learn using the mappings found by our approach are able to reach a performance target faster than agents that learn without transfer.

# Repository layout
**GAME**: contains the main source files.
* **agents**: contains code for the agents used in the reinforcement learning environments. We use Sarsa lambda with tile coding function approximation as our agents.
* **bin**: contains the main GAME algorithms, e.g. intertask mapping learning, reinforcement learning experiments, and transfer learning algorithms.
  * **brute_force.py**: code for the brute force intertask mapping learning algorithm.
  * **ea.py**: code for GAME, the evolutionary algorithms used to evolve intertask mappings.
  * **hill_climber.py**: code for GAME-RMHC, the hill climbing variant of GAME.
  * **intertask_mappings.py**: code to represent intertask mappings and functions that utilize intertask mappings to construct a relation between two reinforcement learning tasks.
  * **mountain_car_experiments.py**: code to set up and run tests in the Mountain Car reinforcement learning environment.
  * **value_fnc_transfer.py**: code for the Value Function Transfer algorithm used to transfer training weights from the source Mountain Car task to the target Mountain Car task.
* **envs**: contains code for the reinforcement learning environments. Mountain Car was built as an [OpenAI Gym](https://github.com/openai/gym) environment.
* **utils**: utility code, e.g. config, training data scrapers and loaders.

**notebooks** and **scripts** contains notebooks and scripts that were used to run experiments and save results. These are not organized and represent a lot of whiteboarding.
