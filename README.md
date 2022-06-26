# projectS
## How to run ProjectS with PyCharm?

1. open **CMD**: **git clone https://github.com/toantranct/projectS.git**
2. install packages in **requirements.txt**
3. Choose 2 folder highway-env, rl-agents. Right Click --> Mark Directory as --> Sources Root
4. Run run_xxxx.py

## FOLDER
1. out : when run run_xxx.py model, video trained will save in folder ./out
2. highway-end: library Highway-env
3. model-saved: The train model is used for testing
4. Test: Folder testing
5. environment file: ./highway-env/highway_env/envs/highway_custom_env.py_
6. environment configuration file: ./rl-agents/scripts/configs/HighwayModEnv/env_multi_agent.json
7. agent configuration folder: ./rl-agents/scripts/configs/HighwayModEnv/agents/DQNAgent 