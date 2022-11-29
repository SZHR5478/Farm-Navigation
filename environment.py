import gym
import gym_farm

def create_env(env_id, **args):

    env = gym.make(env_id)
    return env

create_env('gym_farm/FarmNavigationWorld-v0')