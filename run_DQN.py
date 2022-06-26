import sys
import json
sys.path.append("./highway-env")
import gym
import highway_env
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

env_config = 'rl-agents/scripts/configs/HighwayModEnv/env_multi_agent.json'
agent_config = 'rl-agents/scripts/configs/HighwayModEnv/agents/DQNAgent/dqn.json'

env = load_environment(env_config)
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=100, recover=True)
evaluation.load_agent_model("model_saved/55/checkpoint-8000.tar")
evaluation.test()
# evaluation.train()
#if train model please comment 2 line 15 and 6 and uncomment line 17
env.close()