import gym
import gym_farm
import numpy as np

def create_env(env_id, **args):

    env = gym.make(env_id)
    return env

env = create_env('gym_farm/FarmNavigationWorld-v0')
