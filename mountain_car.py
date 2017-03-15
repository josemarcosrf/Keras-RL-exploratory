import os
import gym
import math
import keras
import pprint
import logging
import argparse
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

from gym import envs
from gym import wrappers
from tendo import colorer
from collections import deque
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (Dense, Input, Dropout, Lambda)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: implement prioratized experience replay
# TODO: Allow multi-layer MLPs
# TODO: Try "target" network


def check_args(args):
    avail_envs = [e.id for e in envs.registry.all()]
    if args.env not in avail_envs:
        logger.error("{} is not a valid env".format(args.env))
        raise ValueError("env must be one of: {}".format(pprint.pformat(sorted(avail_envs))))


def get_model(hidden_size, num_actions, space_shape):
    """ Defines an MLP Q-network takes an observation from the environment
        and outputs a Q value per action.
        In this case there are two actions, i.e: force applyied to the cart (-1, +1)
    """
    inp = Input(shape=space_shape)
    x = Dense(hidden_size, activation="relu")(inp)
    x = Dropout(p=0.25)(x)
    q = Dense(num_actions, activation="linear")(x)
    model = Model(input=inp, output=q)
    print(model.summary())
    return model


def get_dueling_model(hidden_size, num_actions, space_shape, mode):
    """ Defines a dueling Q-network with three different aggregation modes: {avg, max, naive}
        Takes an observation from the environment and outputs a Q value per action
    """
    inp = Input(shape=space_shape)
    h = Dense(hidden_size//2, activation="relu")(inp)
    h = Dropout(p=0.25)(h)
    h = Dense(hidden_size, activation="relu")(inp)
    y = Dense(num_actions + 1)(h)
    if mode == 'avg':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                   output_shape=(A,))(y)
    elif mode == 'max':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                   output_shape=(A,))(y)
    elif mode == 'naive':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:], output_shape=(num_actions,))(y)
    else:
        raise ValueError("Invalid mode: {} ".format(mode))
    model = Model(input=inp, output=z)
    print(model.summary())
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('env', default='CartPole-v0', help="GYM Environment")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--replay_size', type=int, default=10000)
    parser.add_argument('--train_repeat', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.95, help="reward discount factor")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epsilon', type=float, default=0.15, help="Exploration probability")
    parser.add_argument('--exploration_decay', type=float, default=0.01)
    parser.add_argument('--max_episodes', type=int, default=10000)
    parser.add_argument('--nn_mode', default="max", help="aggregation mode for dueling network or MLP")
    parser.add_argument('--model_path', default="cart-pole-mlp")
    parser.add_argument('--render', type=float, default=-200, help="minimum avg reward to start rendering")
    args = parser.parse_args()

    # check coherence of arguments
    check_args(args)

    # We aim to solve the balancing pole
    env = gym.make(args.env)
    env_name = args.env.split("-")[0]
    env = wrappers.Monitor(env, '/tmp/{}-experiment-1'.format(env_name), force=True)

    # Environment parameters
    A = env.action_space.n
    D = env.observation_space.shape

    if args.batch_size > args.replay_size:
        logger.warning("Replay size should be bigger than replay size to guarantee batches of the given size")

    # define and maybe load model
    if args.nn_mode == "mlp":
        model = get_model(args.hidden_size,A, D)
    else:
        model = get_dueling_model(args.hidden_size,A, D, mode=args.nn_mode)

    # load model if exists
    if os.path.exists(args.model_path):
        logger.info("Loading model from: {}".format(args.model_path))
        model.load_weights(args.model_path)
    optim = Adam(lr=args.lr, decay=1e-5)
    model.compile(optimizer=optim, loss='mse')

    # per episode data holders
    pre_states = deque([], args.replay_size)
    actions = deque([], args.replay_size)
    rewards = deque([], args.replay_size)
    post_states = deque([], args.replay_size)
    terminal = deque([], args.replay_size)

    timesteps = 0
    total_reward = 0
    avg_reward = -float("inf")
    episode_number = 1
    episode_reward = 0
    rendering = False
    exploration_factor = args.epsilon

    observation = env.reset()   # Obtain an initial observation of the environment
    while episode_number <= args.max_episodes:
        if avg_reward >= args.render: 
            env.render()
        
        # epsilon-greedy policy
        if np.random.uniform() < exploration_factor:
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
        if len(pre_states) >= args.replay_size:
            for k in range(args.train_repeat):
                # sample from experience buffer
                sample_idx = np.random.choice(len(pre_states), size=args.batch_size)
                # get Q values for all the sample states s --> s' (we use a TD(0))
                # (sample backup as we only take one action into account for the update)
                q_pre_states = model.predict(np.array(pre_states)[sample_idx])      # q_t(s,a)
                q_post_states = model.predict(np.array(post_states)[sample_idx])    # q_t+1(s,a)
                # compute deltas
                for i, idx in enumerate(sample_idx):
                    if terminal[idx]:   # there's no post_state, i.e: s_(t+1), as the episode finished
                        q_pre_states[i, actions[idx]] = np.array(rewards)[idx]
                    else:
                        q_pre_states[i, actions[idx]] = np.array(rewards)[idx] + args.gamma * np.max(q_post_states[i])

                # train model
                logger.debug("episode: {} - {}th training on batch".format(episode_number, k))
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
            exploration_factor /= (1.0 + args.exploration_decay)
            observation = env.reset()


logger.info("Average reward per episode {}".format(total_reward / args.max_episodes))
logger.info('{} Episodes completed.'.format(args.max_episodes))

logger.info("saving model as {}".format(args.model_path))
model.save(args.model_path)
