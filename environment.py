import gym

def create_env(env_id, args):

    env = gym.make(env_id)
    return env

