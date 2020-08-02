# Collaboration_and_Competition_Udacity
This is the final project in deep reinforcement learning nano-degree provided by Udacity

## Project details

### Problem definition
In this environment, two agents control rackets to bounce a ball over a net. The goal of each agent is to keep the ball in play.

![Tennis environment](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)

### State and action spaces
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Reward and score
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

### Solving the environment
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


## Getting started
1. You can start by cloning this project `git clone git@github.com:sdassi/Collaboration_and_Competition_Udacity.git`
2. Install all requirements from the requirements file `pip install -r requirements.txt`
3. Download the environment from the link below (select only the one matching your OS):
    * Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    * Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    * Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    * Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

4. Place the downloaded file in the location you want (you can place it in this repo for example), then unzip (or decompress the file).

## Instructions
This code allow you to train the agent or evaluate it. Note that the project already contains a pre-trained weights, if you want to skip the training part and try to evaluate a trained agent that's totally feasible.

### Train the agent
You can start training the agent with this command: `python train_agent.py --env_file <path of the Tennis env>` <br>
`train_agent.py` file has many arguments. But only `env_file` argument is required. It's totally okay if you use default values for the remaining arguments (The way to use default values for aguments is simply not specifying them in the execution command). This is the list of all arguments that can be passed to `train.py` :
- `env_file` : string argument, path of the Reacher environment file (example `Tennis_Linux/Tennis.x86`) 
- `n_episodes` : integer argument, maximal number of training episodes, default: 4000
- `print_every` : int argument, the frequency of printing the average score during training, default: 50

### Run episode with trained agent
To evaluate the trained agent, you can run: `python eval_agent.py --file_name <path of the Tennis env>` <br>
`eval_agent.py` file has only two arguments:
- `env_file` : The same as defined in the previous section
- `n_episodes`: number of episodes to play, default value 1