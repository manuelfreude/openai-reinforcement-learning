# Open ai reinforcement learning experimental code
In this repository, code is included for experimenting with the open ai environments. Several environments will be tackled over time and different reinforcement learning algorithms. 


## Temporal difference learning for Taxi-v2 environment

Within the Udacity trainings, several methods were developed to solve the Taxi-v2 environment: temporal difference, sarsa and sarsamax. In general, the coded sarsamax model provded the best rewards for the environment, none of the algorithms reached the 9.7 solution state. To date, the best score in the leaderboard was 9.423. The provided code did not always provide the same results in each iteration, yet the sarsamax achieved a higher score at least once. 

## Deep deterministic policy gradient actor-critic for Pendulum-v0 environment with Keras
Keras DDPG source code for the "DDPG_Pendulum.py" is a slightly modified https://github.com/pemami4911/deep-rl. Details about the background of the agent can be found here: http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html. The agent is working in the environment, but not coming close to keeping the pendulum balanced. 

The "DDPG_Pendulum_v2.py" includes code with modified neural network architectures compared to the above mentioned version, batch normalization layers were added. The performance improved and the agent seemed to some degree understand that the pendulum needs to be balanced on top for best results, yet it did not succeed in doing so and further iterations of the code are needed. 

## Deep deterministic policy gradient actor-critic for Pendulum-v0 environment with TensorFlow

The TensorFlow code is obtained from https://github.com/pemami4911/deep-rl and slightly modified along the number of max episodes. The code's performance is quite well, even though an error occurred when rendering was tried, so the performance visualization could not be created, potentially an error due to the laptop used. Detailed reading is provided in this blog post http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html. 

## Further environments and code are tbd
