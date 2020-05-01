## Deep Deterministic Policy gradients
Uses fixed q-targets and experience replay
20 agent environment sharing neural networks, one network for Actor, one network for Critic
Three layer fully connected neural networks with batch normalization, relu and tanh activation.
The critic takes in the actions selected by the actor in the second layer of it's neural network.

#### Hyperparameters
actor learning rate 0.0001

critic learning rate 0.001

stores the last 1 million experience tuples in the replay buffer

discounts future rewards at rate gamma 0.95

runs 128 sample SARS tuples through the network for every training run, batch size 128     

copies weights to target networks after every training run at rate tau 0.001

training_interval 20 trains the networks every 20 timesteps

train_steps 1 trains the networks one time for every agent 	

#### Results
[image1]: https://github.com/Maggern3/DDPG/blob/master/training_results.png "Trained Agent"

![Trained Agent][image1]

The environment was solved in 120 episodes.

#### Ideas for future work
Using dropout, adding prioritized experience replay, noise scaling(decrease noise as training progress)
