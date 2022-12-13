import gym
import gym_farm

def create_env(args):

    env = gym.make(id = args.env,stack_frames = args.stack_frames)
    return env

