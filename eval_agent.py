import argparse
from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt

from maddpg import MADDPG 


def run_episode(env, brain_name):
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    agent.reset()
    score = np.zeros(num_agents)
    while True: # The task is episodic
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        states = next_states
        score += rewards
        if any(dones):
            break
        
    return np.max(score) # single score for the episode just played


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', '-e', default='', type=str)
    parser.add_argument('--n_episodes', '-n', default=1, type=int)
    args = parser.parse_args()

    #Instantiate the environment
    env = UnityEnvironment(file_name=args.env_file)

    # get the default brain
    brain_name = env.brain_names[0]

    #Instantiate the agent
    state_size_per_agent = 8
    num_stacked_obs = 3
    state_size = state_size_per_agent * num_stacked_obs
    action_size_per_agent = 2
    agent= MADDPG(state_size=state_size, action_size=action_size_per_agent, random_seed=0)

    #Load weights
    for num_ag, ag in enumerate(agent.agents):
        ag.actor_local.load_state_dict(torch.load('checkpoints/agent%d_checkpoint_actor.pth' %(num_ag+1)))
        ag.critic_local.load_state_dict(torch.load('checkpoints/agent%d_checkpoint_critic.pth' %(num_ag+1)))

    #Play episodes
    scores = []
    for i in range(args.n_episodes):
        score = run_episode(env, brain_name)
        scores.append(score)
    print("average episodes score :  ", np.mean(scores))