import os
import gym
import math
import keras
import logging
import argparse
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

from gym import wrappers
from tendo import colorer
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (Dense, Input, Dropout, Lambda)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO: Try decaying exploration factor
# TODO: measure impact of replay factor resetting


# We aim to solve the balancing pole
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

# Environment parameters
A = env.action_space.n
D = env.observation_space.shape

def get_model(hidden_size):
    """ Defines an MLP Q-network takes an observation from the environment
        and outputs a Q value per action.
        In this case there are two actions, i.e: force applyied to the cart (-1, +1)
    """
    inp = Input(shape=D)
    x = Dense(hidden_size, activation="relu")(inp)
    # x = Dropout(p=0.25)(x)
    q = Dense(A, activation="linear")(x)
    model = Model(input=inp, output=q)
    print(model.summary())
    return model


def get_dueling_model(hidden_size, mode="mean"):
    """ Defines a dueling Q-network with three different aggregation modes: {avg, max, naive}
        Takes an observation from the environment and outputs a Q value per action
    """
    inp = Input(shape=D)
    h = Dense(hidden_size, activation="relu")(inp)
    # h = Dropout(p=0.25)(h)
    y = Dense(A + 1)(h)
    if mode == 'avg':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                   output_shape=(A,))(y)
    elif mode == 'max':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                   output_shape=(A,))(y)
    elif mode == 'naive':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:], output_shape=(env.action_space.n,))(y)
    else:
        raise ValueError("Invalid mode: {} ".format(mode))
    model = Model(input=inp, output=z)
    print(model.summary())
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--replay_size', type=int, default=100)
    parser.add_argument('--train_repeat', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99, help="reward discount factor")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epsilon', type=float, default=0.1, help="Exploration probability")
    parser.add_argument('--max_episodes', type=int, default=200)
    parser.add_argument('--nn_mode', default="max", help="aggregation mode for dueling network")
    parser.add_argument('--model_save_path', default="cart-pole-mlp")
    parser.add_argument('--non_dueling', action="store_true", help="whether to use a non-dueling network")
    args = parser.parse_args()

    if args.batch_size > args.replay_size:
        logger.warning("Replay size should be bigger than replay size to guarantee batches of the given size")

    # define and maybe load model
    if args.non_dueling:
        model = get_dueling_model(args.hidden_size)
    else:
        model = get_dueling_model(args.hidden_size, mode=args.nn_mode)

    if os.path.exists(args.model_save_path):
        logger.info("Loading model from: {}".format(args.model_save_path))
        model.load_weights(args.model_save_path)
    optim = Adam(lr=args.lr, decay=1e-5)
    model.compile(optimizer=optim, loss='mse')

    pre_states, actions, rewards, post_states, terminal = [], [], [], [], []   # per episode data
    total_reward = 0
    episode_number = 1
    episode_reward = 0
    timesteps = 0
    rendering = False

    observation = env.reset()   # Obtain an initial observation of the environment
        
    while episode_number <= args.max_episodes:
        
        # Rendering the environment slows things down, 
        # so let's only look at it once our model is doing a good job.
        if total_reward/episode_number > 50 or rendering == True: 
            env.render()
            rendering = True
        
        # greedy versus exploration (policy)
        if np.random.uniform() < args.epsilon:   # change this to explore more at the begining
            action = np.random.randint(A)
        else:
            s = np.array([observation])
            q = model.predict(s)[0]
            action = np.argmax(q)

        pre_states.append(observation)     # observation
        actions.append(action)

        # take a step and get new measurements
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        timesteps += 1

        # record reward (has to be done after we call step() to get reward for previous action)
        rewards.append(reward)
        post_states.append(observation)
        terminal.append(done)

        # Experience replay (helps to break temporal correlations and gains in efficiency due to batching)
        if len(pre_states) > args.replay_size:
            for k in range(args.train_repeat):
                # sample from experience buffer
                sample_idx = np.random.choice(len(pre_states), size=args.batch_size)
                # get Q values for all the sample states s --> s'
                q_pre_states = model.predict(np.array(pre_states)[sample_idx])      # q_t(s,a)
                q_post_states = model.predict(np.array(post_states)[sample_idx])    # q_t+1(s,a)
                # compute deltas
                for i, idx in enumerate(sample_idx):
                    if terminal[idx]:   # there's no post_state, i.e: s_(t+1), as the episode finished
                        q_pre_states[i, actions[idx]] = np.array(rewards)[idx]
                    else:
                        q_pre_states[i, actions[idx]] = (np.array(rewards)[idx] + args.gamma * np.max(q_post_states[i]))

                # train model
                model.train_on_batch(np.array(pre_states)[sample_idx], q_pre_states)

        # end of the episode   
        if done:
            total_reward += episode_reward
            avg_reward = total_reward / episode_number
            logger.info("Episode {} finished after {} timesteps,"
                        " episode reward {}, avg reward: {:.3f}".format(episode_number, timesteps,
                                                                        episode_reward, avg_reward))
            # reset vars
            timesteps = 0
            episode_reward = 0
            episode_number += 1
            # reset the replay buffer ==> easy with mem 
            max_buf_size = args.replay_size * 5
            if len(pre_states) > max_buf_size:
                pre_states = pre_states[-max_buf_size:]
                actions = actions[-max_buf_size:]
                rewards = rewards[-max_buf_size:]
                post_states = post_states[-max_buf_size:]
                terminal = terminal[-max_buf_size:]
            observation = env.reset()


logger.info("Average reward per episode {}".format(total_reward / args.max_episodes))
logger.info('{} Episodes completed.'.format(args.max_episodes))

logger.info("saving model as {}".format(args.model_save_path))
model.save(args.model_save_path)
