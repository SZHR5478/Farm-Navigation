from gym import Env
from gym import spaces
import numpy as np
import os
import json
import time
from gym_farm.envs.interaction import UE4
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_env_setting(filename):
    f = open(get_settingpath(filename))
    setting = json.load(f)
    return setting


def get_settingpath(filename):
    import gym_farm
    gympath = os.path.dirname(gym_farm.__file__)
    return os.path.join(gympath,'envs',filename)

class FarmNavigationEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,setting_file,render_mode=None):

        setting = load_env_setting(setting_file)
        self.distance_threshold = setting['distance_threshold']

        # connect UE4
        self.target_agent = UE4(master_ip=setting['master_ip'], send_port=setting['send_port'], receive_port=setting['receive_port'],historical_image_length=setting['historical_image_length'],
                                env_name=setting['env_name'], agent_name=setting['target_agent_name'])

        self.move_agent = UE4(master_ip=setting['master_ip'], send_port=setting['send_port'],receive_port=setting['receive_port'],historical_image_length=setting['historical_image_length'],
                                env_name=setting['env_name'], agent_name= setting['target_agent_name'][:-1] + ('1' if setting['target_agent_name'][-1] == '0' else '0'))

        self.action_space = spaces.Box(low=np.array(setting['continous_actions']['low']),high=np.array(setting['continous_actions']['high']))

        #确保收到传感器数据
        time.sleep(3)

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(setting['historical_image_length'],)+self.move_agent.CurrentImage.shape,
                                            dtype=np.uint8)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    def step(self, action):
        actions = np.squeeze(action)
        velocity = actions[0]
        angle = actions[1]
        self.move_agent.set_move(angle, velocity)


        # An episode is done iff the agent has reached the target
        terminated = self._isdone()
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def reset(self, ):
        self.move_agent.set_reset()
        self.target_agent.set_reset()
        time.sleep(3)

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def _isdone(self):
        done = np.linalg.norm(self.move_agent.get_pos() - self.target_agent.get_pos()) <= self.distance_threshold
        return done

    def _get_obs(self):
        return self.move_agent.get_observation()

    def _get_info(self):
        return {"distance": np.linalg.norm(self.move_agent.get_pos() - self.target_agent.get_pos())}

    def _render_frame(self,observation):
        ax = plt.subplot()
        ax.imshow(observation)
        def update(i):
            ax.set_data(self._get_obs())
        ani = FuncAnimation(plt.gcf(),update(),interval=200)