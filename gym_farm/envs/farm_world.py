import gym
from gym import spaces
import numpy as np
import os
import json
from interaction import UE4
from reward import reward_distance

def load_env_setting(filename):
    f = open(get_settingpath(filename))
    setting = json.load(f)
    return setting


def get_settingpath(filename):
    gympath = os.path.dirname(UE4.__file__)
    return os.path.join(gympath, filename)

class FarmNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,setting_file):

        setting = load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.agent_name = setting['agent_name']
        self.continous_actions = setting['continous_actions']
        self.reset_area = setting['reset_area']
        self.master_ip = setting['master_ip']
        self.send_port = setting['send_port']
        self.receive_port = setting['receive_port']

        # connect UE4
        self.target_agent = UE4(master_ip=setting['master_ip'], send_port=setting['send_port'], receive_port=setting['receive_port'],
                                env_name=setting['env_name'], agent_name=setting['target_agent_name'])

        self.move_agent = UE4(master_ip=setting['master_ip'], send_port=setting['send_port'],receive_port=setting['receive_port'],
                                env_name=setting['env_name'], agent_name= setting['target_agent_name'][:-1] + ('0' if setting['target_agent_name'][-1] == '0' else '1'))

        self.action_space = spaces.Box(low=np.array(setting['continous_actions']['low']),high=np.array(setting['continous_actions']['high']))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.move_agent.CurrentImage.shape, dtype=np.uint8)

        self.count_eps = 0
        self.count_steps = 0
        self.count_close = 0
        self.direction = None
        self.ep_lens = []

    def step(self, action):
        actions = np.squeeze(action)
        self.count_steps += 1
        velocity = actions[0]
        angle = actions[1]
        self.move_agent.set_move(angle, velocity)

        # update observation
        self.state = self.move_agent.CurrentImage

        return self.states, reward_distance(self), self.isDone()

    def reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        self.count_eps += 1
        self.ep_lens.append(self.count_steps)
        self.count_steps = 0

        # stop move
        np.random.seed()
        self.move_agent.set_reset()
        self.target_agent.set_reset()

        return self.move_agent.HistoryImages

    def isDone(self):
        distance = 0.5
        move_x,move_y,move_z = self.move_agent.get_pos()
        target_x,target_y,target_z = self.target_agent.get_pos()
        return ((move_x - target_x) ** 2 + (move_y - target_y) ** 2 + (move_z - target_z) ** 2) <= distance