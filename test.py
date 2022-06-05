import sys
sys.path.append("./highway-env")
import gym
import highway_env
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

env_config = 'rl-agents/scripts/configs/HighwayModEnv/env_multi_agent.json'
agent_config = 'rl-agents/scripts/configs/HighwayModEnv/agents/DQNAgent/dqn.json'


env = load_environment(env_config)
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=3000, recover=True)
evaluation.test
env.close()