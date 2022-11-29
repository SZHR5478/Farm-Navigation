import gym
import gym_farm

def create_env(env_id, **args):

    env = gym.make(env_id)
    return env

env = create_env('gym_farm/FarmNavigationWorld-v0')
env.reset()
#env._render_frame()