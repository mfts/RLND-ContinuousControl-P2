[//]: # (Image References)

[image1]: ./img/network.png "Network"
[image2]: ./img/untrained.gif "Untrained Agent"
[image3]: ./img/trained.gif "Trained Agent"
[image4]: ./img/ddpg.png "DDPG Scores"

# Report

## Introduction
The project consists of three programming files and a pre-built Unity-ML Agents environment.

The goal of the project is to train a group of 20 identical agents to maintain the position of their hands near a moving target position. This is achieved through reinforcement learning. The agent receives a reward of +0.1 for every time step it is able to maintain its position near the target location.
The environment in which the agent can act is large (`n=33`) and contains rotation, position, velocity and angular velocities of the arm. The agent may can move its arm, which has two joints by changing any of the 4 actions, which correspond to the torque on each joint. Each action is a continuous variable and can vary from -1 to 1. 
Therefore, we can use neither a traditional Q-Table approach (due to high-dimensional observation spaces) nor a Deep Q-Network [DQN] approach, which can solve high-dimensional observation spaces but only discrete, low-dimensional action spaces. In order to get around this barrier, we resort to a model-free, off-policy actor-critic algorithm that uses function approximation to learn policies in high-dimensional, continuous action spaces. This algorithm is call deep deterministic policy gradient [DDPG] algorithm.

The agents are trained successfully to earn an average cumulative reward of +30 over 100 episodes after 36 episodes. In the next section, I will explain the learning algorithm used.

## Learning Algorithm
The DDPG algorithm used in the project consists of an **actor-critic network** combined with two familiar features from DQN: **experienced replay** and **gradual Q-target updates**.

At the heart of the algorithm is the actor-critic model, which is made up of two networks that aim to *learn the optimal policy* (**actor**) and *evaluate an action by computing the value function* (**critic**).

The **actor** neural network consists of 1 input layer, 1 hidden layer and 1 output layer. All layers are fully connected linear layers and map the observation space (states) to action space (actions). The network takes an input of 33 and expands the network to 400 nodes, then contracts to 300 nodes before returning 4 nodes, one for each action. There is a batch normalization layer after the first layer and between each layer there is also a ReLU activation function. The final output layer was a tanh layer to bound the actions.

The **critic** neural network consists of 1 input layer, 1 hidden layer and 1 output layer. All layers are fully connected linear layers and map state-action pairs to Q values. The network takes an input of 33 and expands the network to 400 nodes, then contracts to 300 nodes before returning 1 node, the Q value for a given state-action pair. There is a batch normalization layer after the first layer and between each layer there is also a ReLU activation function. Actions were not included until the second hidden layer. 

Both of the networks' layers were initialized from a uniform distribution, with the output layers initialized uniformly close to zero.

![actor-critic network][image1]

For the remainder the setup is similar to vanilla DQN. The algorithm is set up by initializing two identical Q-networks for the actor and critic each, for current and targeted Q-network weights, as well as a replay buffer that will save previously taken steps by the agent, in order to sample them for improved learning.

The agent selects an action based on the learned optimal policy of the local actor network.

At each time step, i.e. with each action taken, the agents update the replay buffer with the values for the current state, reward, action, next state, and whether the episode has terminated.
I have chosen to draw past replays of the agents every 20th action – there are 4 actions in total – and update the targeted networks 10 times based on the sampled experiences.

Since we have separated the optimal policy and the evaluation of actions in regard to Q values, we have to update *how the agents learn*. 

The **critic (or value) network learns** as follows, by
1. evaluating the current Q-values from the local critic network given current state-action pairs;
2. getting the next actions based on the target actor network and next states;
3. evaluating the next Q-values from the target critic network given next state-action pairs;
4. calculating the TD-error;
5. minimizing the loss computed as the mean squared difference between current Q-values and TD-error; and
6. gradually updating the weights on the target critic network with a small interpolation parameter.

The **actor (or policy) network learns** as follows, by
1. predicting the actions based on the local actor network for the current states;
2. evaluating the average loss with the local critic network given the state and predicted action pairs;
3. minimizing the loss;
4. gradually updating the weights on the target actor network with a small interpolation parameter.  

### Parameter Selection
```
ACTOR/CRITIC NETWORK PARAMETERS
==================
STATE_SIZE = 33         # agent-environment observation space
ACTION_SIZE = 4         # agent's possible action
FC1_UNITS = 400         # first fully-connected layer of network
FC2_UNITS = 300         # second fully-connected layer of network

AGENT PARAMETERS
================
BUFFER_SIZE = int(1e6)  # size of the memory buffer
BATCH_SIZE = 128        # sample minibatch size
GAMMA = 0.99            # discount rate for future rewards
LR_ACTOR = 1e-4         # learning rate of Actor
LR_CRITIC = 1e-4        # learning rate of Critic
TAU = 1e-3              # interpolation factor for soft update of target network
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 20       # update every 20 time steps
NUM_UPDATES = 10        # number of updates to the network

TRAINING PARAMETERS
===================
n_episodes=500          # max number of episodes to train agent
max_t=1000              # max number of steps agent is taking per episode
```

## Training with DDPG
The agents are trained with the previously described DDPG over 500 episodes with max. 1000 actions per episode. The DDPG learns an optimal policy to select appropriate actions and evaluates those actions in regards to Q values.

Below you can see snippet randomly selected actions (**untrained** agent):

![untrained][image2]

*The agents are not able to "grab" a ball and moving frantically caused by taking actions at random.*

Compared with a **trained** agent: 

![trained][image3]

*The agents maintain their hands' position on the target position (circulating balls).*

![scores][image4]

*The distribution shows the rewards per episode for each agents (in color) and averaged (in black) for all agents.*

The environment has its natural limitation: every agent perfectly maintaining their hand on the moving target infinitely long. Hence, there are diminishing returns from training more episodes. The currently trained algorithm tops out at an average reward of +39 over 100 episodes (with 1000 time steps each).

## Improvements
Further improvements to the algorithm include 
- deep distributed distributional deterministic policy gradient (D4PG)
- N-step returns for inferring velocities using differences between frames 
- prioritized experience replay

## Conclusion
The DDPG algorithm is a huge improvement in the space of off-policy actor-critic algorithm, in particular for high-dimensional, continuous action spaces. Surprisingly, the algorithm is able to solve the environment relatively quickly within a 100,000 steps. That makes me ponder what kind of problems this algorithm could solve that a previous algorithm such as DQN had difficulty solving. 
The addition of being able to solve high-dimensional, continuous action spaces certainly makes this algorithm a staple tool in the reinforcement learning spectrum.

Next I will try the extension of the project, which only relies on pixels as input. This will involve making changes mainly to the underlying neural network, which may use convolutions instead of fully connected layers to make sense of the pixels in an abstract fashion. 

