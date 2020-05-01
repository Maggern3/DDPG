## DDPG
#### Project environment
33 states for each agent, 20 agents in multi agent environment. Each agent has 4 continous actions.
The environment is considered solved when an average score of 30+ for all agents is maintained for 100 episodes.

#### Installation 
First clone the [udacity deep reinforcement learning repo](https://github.com/udacity/deep-reinforcement-learning) 
and navigate to it's directory then
```
cd python
pip install .
```
this installs the required dependencies. 

Download the Reacher Unity-ML environment from one of the following links:

Windows: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
     
Mac: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
     
Put it in the root of the cloned project folder and unzip to Reacher_20_agents folder. 

You should now be able to build and run the project.

#### How to train the agent
Run the following command to train the agent
```
python runner.py
```