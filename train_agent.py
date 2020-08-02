import argparse
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from maddpg import MADDPG 

LEN_DEQUE = 100

def train_maddpg(n_episodes, print_every, threshold, brain_name, max_len_deque=LEN_DEQUE):
    """ learn using maddpg algorithm """
    
    scores_deque = deque(maxlen=max_len_deque)
    scores = []

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        episode_score = 0
        agent.reset()
        while True: # The task is episodic
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if any(dones):
                break

        episode_score = np.max(score) # single score for the episode just played
        scores.append(episode_score)
        scores_deque.append(episode_score) 

        if len(scores_deque) == max_len_deque and np.mean(scores_deque) >= threshold:
            print("environment was solved at episode %d" %(i_episode-max_len_deque))
            agent.save_weights()
            return scores

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_deque)))
        
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', '-e', default='', type=str)
    parser.add_argument('--n_episodes', '-n', default=4000, type=int)
    parser.add_argument('--print_every', default=50, type=int)
    args = parser.parse_args()

    #Instantiate the environment
    env = UnityEnvironment(file_name=args.env_file)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    #Instantiate the agent
    state_size_per_agent = 8
    num_stacked_obs = 3
    state_size = state_size_per_agent * num_stacked_obs
    action_size_per_agent = 2
    agent= MADDPG(state_size=state_size, action_size=action_size_per_agent, random_seed=0)

    #Train with MADDPG algorithm
    threshold = 0.5 
    scores = train_maddpg(args.n_episodes, args.print_every, threshold, brain_name)

    #Plot scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    #close env
    env.close()