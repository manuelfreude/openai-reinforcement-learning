# Open ai reinforcement learning experimental code
In this repository, code is included for experimenting with the open ai environments. Several environments will be tackled over time and different reinforcement learning algorithms.


## Temporal difference learning for Taxi-v2 environment

Within the Udacity trainings, several methods were developed to solve the Taxi-v2 environment: temporal difference, sarsa and sarsamax. In general, the coded sarsamax model provded the best rewards for the environment, none of the algorithms reached the 9.7 solution state. To date, the best score in the leaderboard was 9.423. The provided code did not always provide the same results in each iteration, yet the TD learning and sarsamax achieved higher scores at least once when iterating 40.000 times.

The results for episodes 500-5000 are shown below, sarsamax shows the steepest learning curve while TD starts out nicely.

![Taxi-v2 best average rewards](https://github.com/manuelfreude/openai-reinforcement-learning/blob/master/Taxi-v2_rewards.png)

## Deep deterministic policy gradient actor-critic for Pendulum-v0 environment with Keras
Keras DDPG source code for the "DDPG_Pendulum_Keras.py" is a slightly modified https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69. The code includes neural network architectures, batch normalization layers were added. The performance improved and the agent seemed to some degree understand that the pendulum needs to be balanced on top for best results, yet it did not succeed in doing so. Rewards look as follows for 200 and 7.500 episodes:

![keras 200 episodes rewards](https://github.com/manuelfreude/openai-reinforcement-learning/blob/master/keras_reward_200.png)

![keras 7.500 episodes rewards](https://github.com/manuelfreude/openai-reinforcement-learning/blob/master/keras_reward_7500.png)

No learning curve can be seen, the agent keeps running the pendulum around the clock.

## Deep deterministic policy gradient actor-critic for Pendulum-v0 environment with TensorFlow

The TensorFlow code is obtained from https://github.com/pemami4911/deep-rl and slightly modified. Detailed reading is provided in this blog post http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html. The code's performance is quite well, rendering implemented and data summarized for tensorboard (https://www.datacamp.com/community/tutorials/tensorboard-tutorial). For about 75 episodes, the rewards indicate a clear learning curve with a peak reward at -2 (episode 72) as follows:

![ddpg tf rewards](https://github.com/manuelfreude/openai-reinforcement-learning/blob/master/ddpg_tf_rewards.png)

## Further environments and code will follow
