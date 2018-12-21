import gym
import numpy as np
from collections import deque
import sys
import math
from collections import defaultdict
import csv


env = gym.make('Taxi-v2')
env.reset()
env.observation_space.n
env.render()

# temporal difference learning

def td_prediction(env, num_episodes, alpha, gamma, window):
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # Q
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    file_output = 'taxi_td.csv'
    labels = ['num_episodes', 'Best Avg Reward TD']

    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)

        # loop over episodes
        for episode in range(1, num_episodes+1):
            done = False
            G, reward = 0,0
            state = env.reset()
            samp_reward = 0
            while done != True:
                    # choose action A
                    action = np.argmax(Q[state]) #1
                    # take action A, observe R, S'
                    state2, reward, done, info = env.step(action) #2
                    # perform updates
                    Q[state,action] += alpha * (reward + np.max(gamma*(Q[state2])) - Q[state,action]) #3
                    # S <- S'
                    state = state2
                    samp_reward += reward
                    if done:
                        # save final sampled reward
                        samp_rewards.append(samp_reward)
                        break
            if (episode >= 100):
                    # get average reward from last 100 episodes
                    avg_reward = np.mean(samp_rewards)
                    # append to deque
                    avg_rewards.append(avg_reward)
                    # update best average reward
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
            # monitor progress
            print("\rEpisode {}/{} || Best average reward {}".format(episode, num_episodes, best_avg_reward), end="")
            sys.stdout.flush()
            # check if task is solved (according to OpenAI Gym)
            if best_avg_reward >= 9.7:
                print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
                break
            if episode == num_episodes: print('\n')
            writer.writerow([num_episodes, best_avg_reward])
        return avg_rewards, best_avg_reward

# sarsa code

def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
    """ updates the action-value function estimate using the most recent time step """
    return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

def epsilon_greedy_probs(env, Q_s, episode, eps=None):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    epsilon = 0.5 / episode
    if eps is not None:
        epsilon = eps
    policy_s = np.ones(env.action_space.n) * epsilon / (env.action_space.n)
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / (env.action_space.n))
    return policy_s

import matplotlib.pyplot as plt

def sarsa(env, num_episodes, alpha, gamma, window):
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # Q
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    # initialize performance monitor
    plot_every = 100
    # loop over episodes

    file_output = 'taxi_sarsa.csv'
    labels = ['num_episodes', 'Best Avg Reward Sarsa']

    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)

        for episode in range(1, num_episodes+1):
            done = False
            G, reward = 0,0
            state = env.reset()
            samp_reward = 0
            # get epsilon-greedy action probabilities
            policy_s = epsilon_greedy_probs(env, Q[state], episode)
            # pick action A
            action = np.random.choice(np.arange(env.action_space.n), p=policy_s)
            # limit number of time steps per episode
            for t_step in np.arange(300):
                # take action A, observe R, S'
                state2, reward, done, info = env.step(action)
                # add reward to score
                G += reward
                samp_reward += reward
                if not done:
                    # get epsilon-greedy action probabilities
                    policy_s = epsilon_greedy_probs(env, Q[state2], episode)
                    # pick next action A'
                    next_action = np.random.choice(np.arange(env.action_space.n), p=policy_s)
                    # update TD estimate of Q
                    Q[state][action] = update_Q(Q[state][action], Q[state2][next_action],
                                                reward, alpha, gamma)
                    # S <- S'
                    state = state2
                    # A <- A'
                    action = next_action
                if done:
                    # update TD estimate of Q
                    Q[state][action] = update_Q(Q[state][action], 0, reward, alpha, gamma)
                    # append score
                    samp_rewards.append(samp_reward)
                    break
            if (episode >= window):
                    # get average reward from last window episodes
                    avg_reward = np.mean(samp_rewards)
                    # append to deque
                    avg_rewards.append(avg_reward)
                    # update best average reward
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
            # monitor progress
            print("\rEpisode {}/{} || Best average reward {}".format(episode, num_episodes, best_avg_reward), end="")
            sys.stdout.flush()
            # check if task is solved (according to OpenAI Gym)
            if best_avg_reward >= 9.7:
                print('\nEnvironment solved in {} episodes.'.format(episode), end="")
                break
            if episode == num_episodes: print('\n')
            writer.writerow([num_episodes, best_avg_reward])
        return avg_rewards, best_avg_reward


# sarsamax code

import matplotlib.pyplot as plt

def sarsamax(env, num_episodes, alpha, gamma, window):
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # Q
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    # initialize performance monitor
    plot_every = 100
    # loop over episodes

    file_output = 'taxi_sarsamax.csv'
    labels = ['num_episodes', 'Best Avg Reward Sarsamax']

    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)

        for episode in range(1, num_episodes+1):
            done = False
            G, reward = 0,0
            state = env.reset()
            samp_reward = 0
            # get epsilon-greedy action probabilities
            policy_s = epsilon_greedy_probs(env, Q[state], episode)
            # pick action A
            action = np.random.choice(np.arange(env.action_space.n), p=policy_s)
            # limit number of time steps per episode
            for t_step in np.arange(300):
                # take action A, observe R, S'
                state2, reward, done, info = env.step(action)
                # add reward to score
                G += reward
                samp_reward += reward
                if not done:
                    # get epsilon-greedy action probabilities
                    policy_s = epsilon_greedy_probs(env, Q[state2], episode)
                    # pick next action A'
                    next_action = np.random.choice(np.arange(env.action_space.n), p=policy_s)
                    # update TD estimate of Q
                    Q[state][action] = update_Q(Q[state][action], np.max(Q[state2][next_action]),
                                                reward, alpha, gamma)
                    # S <- S'
                    state = state2
                    # A <- A'
                    action = next_action
                if done:
                    # update TD estimate of Q
                    Q[state][action] = update_Q(Q[state][action], 0, reward, alpha, gamma)
                    # append score
                    samp_rewards.append(samp_reward)
                    break
            if (episode >= window):
                    # get average reward from last window episodes
                    avg_reward = np.mean(samp_rewards)
                    # append to deque
                    avg_rewards.append(avg_reward)
                    # update best average reward
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
            # monitor progress
            print("\rEpisode {}/{} || Best average reward {}".format(episode, num_episodes, best_avg_reward), end="")
            sys.stdout.flush()
            # check if task is solved (according to OpenAI Gym)
            if best_avg_reward >= 9.7:
                print('\nEnvironment solved in {} episodes.'.format(episode), end="")
                break
            if episode == num_episodes: print('\n')
            writer.writerow([num_episodes, best_avg_reward])
        return avg_rewards, best_avg_reward


num_episodes = 10000 #best 40.000
alpha = 0.618 #best 0.618
gamma = 0.7 #best 0.7
window = 100 #best

td_prediction(env, num_episodes, alpha, gamma, window)      #best score 9.483


num_episodes = 10000 #best 40.000
alpha = 0.618 #best 0.618
gamma = 1.0 #best 1.0
window = 100

sarsa(env, num_episodes, alpha, gamma, window)              #best score 9.331


num_episodes = 10000 #best 40.000
alpha = 0.3 #best 0.3
gamma = 1.0 #best 1.0
window = 100

sarsamax(env, num_episodes, alpha, gamma, window)           #best score 9.367
