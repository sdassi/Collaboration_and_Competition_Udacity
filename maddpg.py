import numpy as np
import torch

from ddpg import DDPGAgent

class MADDPG:
    """Multi-Agent DDPG"""

    def __init__(self, state_size, action_size, random_seed, sharedBuffer, num_agents=2):
        self.num_agents = num_agents
        self.action_size = action_size
        self.sharedBuffer = sharedBuffer
        self.agents = [DDPGAgent(state_size,action_size,random_seed,self.sharedBuffer) for x in range(self.num_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        self.sharedBuffer.add(states, actions, rewards, next_states, dones)

        for agent in self.agents:
            agent.step()

    def act(self, states, add_noise=True):
        actions = np.zeros([self.num_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise)
        return actions

    def save_weights(self):
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(index+1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(index+1))
    
    def reset(self):        
        for agent in self.agents:
            agent.reset()