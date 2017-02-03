import gym
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pylab
import logging
import os
import sys

# the code is modified based on the framework provided by Carlos Aguayo
# https://gym.openai.com/algorithms/alg_0eUHoAktRVWWM7ZoDBWQ9w


class QLearner(object):
    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 epsilon=0.9,
                 epsilon_decay_rate=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.state = 0
        self.action = 0
        self.qtable = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

    def set_initial_state(self, state):
        """
        @summary: Sets the initial state and returns an action
        @param state: The initial state
        @returns: The selected action
        """
        self.state = state
        self.action = self.qtable[state].argsort()[-1]
        return self.action

    def move(self, state_prime, reward):
        """
        @summary: Moves to the given state with given reward and returns action
        @param state_prime: The new state
        @param reward: The reward
        @returns: The selected action
        """
        alpha = self.alpha
        gamma = self.gamma
        state = self.state
        action = self.action
        qtable = self.qtable

        choose_random_action = (1 - self.epsilon) <= np.random.uniform(0, 1)

        if choose_random_action:
            action_prime = random.randint(0, self.num_actions - 1)
        else:
            action_prime = self.qtable[state_prime].argsort()[-1]

        self.epsilon *= self.epsilon_decay_rate

        qtable[state, action] = (1 - alpha) * qtable[state, action] + alpha * (reward + gamma * qtable[state_prime, action_prime])

        self.state = state_prime
        self.action = action_prime

        return self.action


def cart_pole_with_qlearning(num_episode):
    env = gym.make('CartPole-v0')
    experiment_filename = './cartpole-experiment-1'
    env.monitor.start(experiment_filename, force=True)


    max_number_of_steps = 200
    number_of_features = env.observation_space.shape[0]


    cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
    pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
    cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
    angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]

    def build_state(features):
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def to_bin(value, bins):
        return np.digitize(x=[value], bins=bins)[0]


    reward_log = []
    learner = QLearner(num_states=10 ** number_of_features,
                       num_actions=env.action_space.n,
                       alpha=0.3,
                       gamma=0.99,
                       epsilon=1,
                       epsilon_decay_rate=0.99)
    


    for episode in xrange(num_episode):
        observation = env.reset()
        cart_position,  cart_velocity, pole_angle, angle_rate_of_change = observation
        state = build_state([to_bin(cart_position, cart_position_bins),                             
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])
        action = learner.set_initial_state(state)

        episode_reward = 0
        for step in xrange(max_number_of_steps - 1):
            observation, reward, done, info = env.step(action)

            cart_position, cart_velocity, pole_angle,  angle_rate_of_change = observation

            state_prime = build_state([to_bin(cart_position, cart_position_bins),
                                       to_bin(cart_velocity, cart_velocity_bins),
                                       to_bin(pole_angle, pole_angle_bins),
                                       to_bin(angle_rate_of_change, angle_rate_bins)])

            episode_reward += reward
            if done:
               reward = -200

            action = learner.move(state_prime, reward)

            if done:

                break

        reward_log.append(episode_reward)
    

    env.monitor.close()
    logger.info("Now trying to upload results to the scoreboard.")
    gym.upload(experiment_filename,api_key='sk_A1BpYdknRCFWZDAIFWSew')
    return reward_log


if __name__ == "__main__":
    random.seed(0)


    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    logger.setLevel(logging.INFO)



num_episode = 5000
reward_log = cart_pole_with_qlearning(num_episode)
plt.subplot(2,1,1)
plt.plot( np.arange(num_episode), reward_log ) 
#plt.axis([0, num_episode, 0, 210])
plt.title('reward of each episode')

reward_log = np.array(reward_log)
ave_reward = []

for i in range(num_episode/100):
    reward_list = reward_log[i*num_episode/100:(i+1)*num_episode/100]
    average = np.average(reward_list)

    print average
    ave_reward.append(average)

plt.subplot(2,1,2)
plt.plot( np.arange(num_episode/100)+1, ave_reward )
#plt.axis([1, num_episode/100, 0, 210])
plt.title('average reward over each 100 episode')
pylab.show()




