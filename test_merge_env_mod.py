import sys
sys.path.append("./highway-env")
import gym
import highway_env
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

env_config = 'rl-agents/scripts/configs/MergeEnvMod/env_idm.json'
agent_config = 'rl-agents/scripts/configs/MergeEnvMod/agents/MCTSAgent/assume_idm.json'


env = load_environment(env_config)
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=10)
evaluation.test()
env.close()