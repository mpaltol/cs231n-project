from gym.envs.registration import register

register(
    id='aircraft-TD-v0',
    entry_point='aircraft_TD.aircraft_TD:ATDEnv',
)
