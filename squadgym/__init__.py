from gym.envs.registration import register

register(
    id='wiki-v0',
    entry_point='squadgym.envs:WikiEnv',
    timestep_limit=1000,
)
