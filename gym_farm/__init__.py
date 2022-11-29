from gym.envs.registration import register

register(
    id="gym_farm/FarmNavigationWorld-v0",
    entry_point="gym_farm.envs:FarmNavigationEnv",
    max_episode_steps=500,
    kwargs={'setting_file': 'farm.json'},
)