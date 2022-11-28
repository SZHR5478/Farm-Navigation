def reward_distance(env):
    move_x, move_y, move_z = env.move_agent.get_pos()
    target_x, target_y, target_z = env.target_agent.get_pos()
    return 1 / ((move_x - target_x) ** 2 + (move_y - target_y) ** 2 + (move_z - target_z) ** 2)
    return reward
