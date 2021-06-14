#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 3: Partially observable Markov decision problems
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab3-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).

# ### 1. The POMDP model
# 
# Consider once again the "Doom" domain, described in the Homework which you modeled using a partially observable Markov decision process. In this environment,
# 
# * There is an Imp that, if it stands in the same cell as the agent, will inflict a large amount of damage to the agent. The Imp moves between cells 10 and 12. At each step, the Imp moves to each adjacent cell with a 0.2 probability, and remains in the same cell otherwise.
# * The agent can move in any of the four directions: up, down, left, and right. It can also listen for the Imp's grunting.
# * Movement actions across a grey cell division succeed with a 0.8 probability and fail with a 0.2 probability.
# * When the movement fails, the agent remains in the same cell.
# * The action "Listen" always keeps the position of the agent unchanged.
# * The agent is able to see the Imp with probability 1 if it stands in the same cell. 
# * If the agent stands in a cell adjacent to the Imp after executing a movement action, it is able to hear the Imp's grunting with a probability 0.3, and with a probability 0.7 it hears nothing. 
# * If the agent stands in in a cell adjacent to the Imp after executing a listening action, it is able to hear the Imp's grunting with a probability 0.7, but with a probability 0.3 it still hears nothing.
# 
# You should also consider the following additional element:
# 
# * Movement actions across colored cell divisions (blue or red) succeed with a 0.8 probability (and fail with a 0.2 probability) only if the agent has the corresponding colored key. Otherwise, they fail with probability 1. To get a colored key, the agent simply needs to stand in the corresponding cell.
# 
# The action that takes the agent through the exit always succeeds. 
# 
# In this lab you will interact with larger version of the same problem. You will use a POMDP based on the aforementioned domain and investigate how to evaluate, solve and simulate a partially observable Markov decision problem. The domain is represented in the diagram below.
# 
# <img src="maze.png" width="400px">
# 
# We consider that the agent is never in a cell $c\geq 17$ without a red key, and is never in a cell $c\geq28$ without a blue key. **Throughout the lab, unless if stated otherwise, use $\gamma=0.95$.**
# 
# $$\diamond$$
# 
# In this first activity, you will implement an POMDP model in Python. You will start by loading the POMDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, observations, transition probability matrices, observation probability matrices, and cost function.

# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_pomdp` that receives, as input, a string corresponding to the name of the file with the POMDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 6 arrays:
# 
# * An array `X` that contains all the states in the POMDP, represented as strings.
# * An array `A` that contains all the actions in the MDP, also represented as strings. 
# * An array `Z` that contains all the observations in the POMDP, also represented as strings.
# * An array `P` containing as many sub-arrays as the number of actions, each sub-array corresponding to the transition probability matrix for one action.
# * An array `O` containing as many sub-arrays as the number of actions, each sub-array corresponding to the observation probability matrix for one action.
# * An array `c` containing the cost function for the POMDP.
# 
# Your function should create the POMDP as a tuple `(X, A, Z, (Pa, a = 0, ..., nA), (Oa, a = 0, ..., nA), c, g)`, where `X` is a tuple containing the states in the POMDP represented as strings, `A` is a tuple with `nA` elements, each corresponding to an action in the POMDP represented as a string, `Z` is a tuple containing the observations in the POMDP represented as strings, `P` is a tuple with `nA` elements, where `P[u]` is an np.array corresponding to the transition probability matrix for action `u`, `O` is a tuple with `nA` elements, where `O[u]` is an np.array corresponding to the observation probability matrix for action `u`, `c` is an np.array corresponding to the cost function for the POMDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the POMDP tuple.
# 
# **Note**: Don't forget to import `numpy`.
# 
# ---

# In[16]:


import numpy as np
import numpy.random as rand
print("################################################")
print("Activity 1\n\n")


def load_pomdp(filename, gamma):
    try:
        data = np.load(filename)
        M = (data['X'], data['A'], data['Z'], data['P'], data['O'], data['c'], gamma)

    except KeyError:
        print("Error in npz file, expected content not found")
    except:
        print("Error opening '{}'".format(filename))

    return M


M = load_pomdp('maze.npz', 0.95)

rand.seed(42)

# States
print('Number of states:', len(M[0]))

# Random state
s = rand.randint(len(M[0]))
print('Random state:', M[0][s])

# Actions
print('Number of actions:', len(M[1]))

# Random action
a = rand.randint(len(M[1]))
print('Random action:', M[1][a])

# Observations
print('Number of observations:', len(M[2]))

# Random observation
z = rand.randint(len(M[2]))
print('Random observation:', M[2][z])

# Transition probabilities
print('Transition probabilities for the selected state/action:')
print(M[3][a][s, :])

# Observation probabilities
print('Observation probabilities for the selected state/action:')
print(M[4][a][s, :])

# Cost
print('Cost for the selected state/action:')
print(M[5][s, a])

# Discount
print('Discount:', M[6])


# We provide below an example of application of the function with the file `maze.npz` that you can use as a first "sanity check" for your code. The POMDP in this file corresponds to the environment in the diagram above. In this POMDP,
# 
# * There is a total of $217$ states describing the different positions of the agent and Imp in the environment, and whether or not the agent has each of the two keys. Those states are represented as strings taking one of the forms `"NmM"`, indicating that the agent is in cell `N` and the Imp in cell `M`, `"NRmM"`, indicating that the agent is in cell `N` with the red key and the Imp in cell `M`, `"NRBmM"`, indicating that the agent is in cell `N` with both keys and the Imp is in cell `M`, or `"E"`, indicating that the agent has reached the exit.
# * There is a total of five actions, each represented as a string `"up"`, `"down"`, `"left"`, `"right"`, or `"listen"`.
# * There is a total of 99 observations, corresponding to the observable features of the state. Those observations are represented as strings taking one of the forms `"Nm0"`, indicating that the agent is in cell `N` and heard nothing, `"Nmg"`, indicating that the agent is in cell `N` and heard grunting, `"NmN"`, indicating that the agent is in cell `N` with the Imp, `"NRm0"`, indicating that the agent is in cell `N` with the red key and heard nothing, `"NRmg"`, indicating that the agent is in cell `N` with the red key and heard grunting, `"NRmN"`, indicating that the agent is in cell `N` with the red key and the Imp, `"NRBm0"`, indicating that the agent is in cell `N` with both keys and heard nothing, `"NRBmg"`, indicating that the agent is in cell `N` with both keys and heard grunting, `"NRBmN"`, indicating that the agent is in cell `N` with both keys and the Imp, or `"E"`, indicating that the agent has reached the exit.
# 
# Note that, in the code below, even fixing the seed, the results you obtain may slightly differ.
# 
# ```python
# import numpy.random as rand
# 
# M = load_pomdp('maze.npz', 0.95)
# 
# rand.seed(42)
# 
# # States
# print('Number of states:', len(M[0]))
# 
# # Random state
# s = rand.randint(len(M[0]))
# print('Random state:', M[0][s])
# 
# # Actions
# print('Number of actions:', len(M[1]))
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('Random action:', M[1][a])
# 
# # Observations
# print('Number of observations:', len(M[2]))
# 
# # Random observation
# z = rand.randint(len(M[2]))
# print('Random observation:', M[2][z])
# 
# # Transition probabilities
# print('Transition probabilities for the selected state/action:')
# print(M[3][a][s, :])
# 
# # Observation probabilities
# print('Observation probabilities for the selected state/action:')
# print(M[4][a][s, :])
# 
# # Cost
# print('Cost for the selected state/action:')
# print(M[5][s, a])
# 
# # Discount
# print('Discount:', M[6])
# ```
# 
# Output:
# 
# ```
# Number of states: 217
# Random state: 15Rm11
# Number of actions: 5
# Random action: right
# Number of observations: 99
# Random observation: 12m12
# Transition probabilities for the selected state/action:
# [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.2 0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0. ]
# Observation probabilities for the selected state/action:
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0.]
# Cost for the selected state/action:
# 0.25
# Discount: 0.95
# ```

# ### 2. Sampling
# 
# You are now going to sample random trajectories of your POMDP and observe the impact it has on the corresponding belief.

# ---
# 
# #### Activity 2.
# 
# Write a function called `gen_trajectory` that generates a random POMDP trajectory using a uniformly random policy. Your function should receive, as input, a POMDP described as a tuple like that from **Activity 1** and two integers, `x0` and `n` and return a tuple with 3 elements, where:
# 
# 1. The first element is a `numpy` array corresponding to a sequence of `n+1` state indices, $x_0,x_1,\ldots,x_n$, visited by the agent when following a uniform policy (i.e., a policy where actions are selected uniformly at random) from state with index `x0`. In other words, you should select $x_1$ from $x_0$ using a random action; then $x_2$ from $x_1$, etc.
# 2. The second element is a `numpy` array corresponding to the sequence of `n` action indices, $a_0,\ldots,a_{n-1}$, used in the generation of the trajectory in 1.;
# * The third element is a `numpy` array corresponding to the sequence of `n` observation indices, $z_1,\ldots,z_n$, experienced by the agent during the trajectory in 1.
# 
# The `numpy` array in 1. should have a shape `(n+1,)`; the `numpy` arrays from 2. and 3. should have a shape `(n,)`.
# 
# **Note:** Your function should work for **any** POMDP specified as above. Also, you may find useful to import the numpy module `numpy.random`.
# 
# ---

# In[17]:



print("################################################")
print("Activity 2\n\n")


def gen_trajectory(M, x0, n):
    nr_states = len(M[0])
    nr_actions = len(M[1])
    state_trans_prob = M[3]
    obs_prob = M[4]

    state_seq = np.zeros(n + 1, dtype=int)
    state_seq[0] = x0
    action_seq = np.zeros(n, dtype=int)
    obser_seq = np.zeros(n, dtype=int)

    for i in range(n):
        # Choose random action
        action_seq[i] = rand.randint(0, nr_actions)
        # Roll dice for next state
        weightedpick = np.cumsum(state_trans_prob[action_seq[i], state_seq[i]])
        state_seq[i + 1] = weightedpick.searchsorted(np.random.uniform(0, weightedpick[-1]))
        # Roll dice for observation
        weightedpick = np.cumsum(obs_prob[action_seq[i], state_seq[i + 1]])
        obser_seq[i] = weightedpick.searchsorted(np.random.uniform(0, weightedpick[-1]))

    return state_seq, action_seq, obser_seq


rand.seed(42)

# Trajectory of 10 steps from state I - state index 0
t = gen_trajectory(M, 0, 10)

print('Shape of state trajectory:', t[0].shape)
print('Shape of state trajectory:', t[1].shape)
print('Shape of state trajectory:', t[2].shape)

print('\nStates:', t[0])
print('Actions:', t[1])
print('Observations:', t[2])

# Check states, actions and observations in the trajectory
print('Trajectory:\n{', end='')

for idx in range(10):
    ste = t[0][idx]
    act = t[1][idx]
    obs = t[2][idx]

    print('(' + M[0][ste], end=', ')
    print(M[1][act], end=', ')
    print(M[2][obs] + ')', end=', ')

print('\b\b}')


# As an example, you can run the following code on the POMDP from **Activity 1**.
# 
# ```python
# rand.seed(42)
# 
# # Trajectory of 10 steps from state I - state index 0
# t = gen_trajectory(M, 0,  10)
# 
# print('Shape of state trajectory:', t[0].shape)
# print('Shape of state trajectory:', t[1].shape)
# print('Shape of state trajectory:', t[2].shape)
# 
# print('\nStates:', t[0])
# print('Actions:', t[1])
# print('Observations:', t[2])
# 
# # Check states, actions and observations in the trajectory
# print('Trajectory:\n{', end='')
# 
# for idx in range(10):
#     ste = t[0][idx]
#     act = t[1][idx]
#     obs = t[2][idx]
# 
#     print('(' + M[0][ste], end=', ')
#     print(M[1][act], end=', ')
#     print(M[2][obs] + ')', end=', ')
# 
# print('\b\b}')
# ```
# 
# Output:
# 
# ```
# Shape of state trajectory: (11,)
# Shape of state trajectory: (10,)
# Shape of state trajectory: (10,)
# 
# States: [  0 145 145 144   0 145   1   1   0   1   0]
# Actions: [3 4 2 2 3 3 4 2 3 2]
# Observations: [1 1 0 0 1 1 1 0 1 0]
# Trajectory:
# {(1m10, right, 2m0), (2m12, listen, 2m0), (2m12, left, 1m0), (1m12, left, 1m0), (1m10, right, 2m0), (2m12, right, 2m0), (2m10, listen, 2m0), (2m10, left, 1m0), (1m10, right, 2m0), (2m10, left, 1m0)}
# ```

# You will now write a function that samples a given number of possible belief points for a POMDP. To do that, you will use the function from **Activity 2**.
# 
# ---
# 
# #### Activity 3.
# 
# Write a function called `sample_beliefs` that receives, as input, a POMDP described as a tuple like that from **Activity 1** and an integer `n`, and return a tuple with `n` elements **or less**, each corresponding to a possible belief state (represented as a $1\times|\mathcal{X}|$ vector). To do so, your function should
# 
# * Generate a trajectory with `n` steps from a random initial state, using the function `gen_trajectory` from **Activity 2**.
# * For the generated trajectory, compute the corresponding sequence of beliefs, assuming that the agent does not know its initial state (i.e., the initial belief is the uniform belief). 
# 
# Your function should return a tuple with the resulting beliefs, **ignoring duplicate beliefs or beliefs whose distance is smaller than $10^{-3}$.**
# 
# **Note 1:** You may want to define an auxiliary function `belief_update` that receives a belief, an action and an observation and returns the updated belief.
# 
# **Note 2:** Your function should work for **any** POMDP specified as above. To compute the distance between vectors, you may find useful `numpy`'s function `linalg.norm`.
# 
# 
# ---

# In[18]:


print("################################################")
print("Activity 3\n\n")


def sample_beliefs(M, n):
    nr_states = len(M[0])
    state_trans_prob = M[3]
    obs_prob = M[4]
    belief = np.ones((n+1, nr_states)) / nr_states


    state_seq, action_seq, obser_seq = gen_trajectory(M, 0, n)

    #Calculate Belief
    for i in range(n):
        alfahat = belief[i].dot(state_trans_prob[action_seq[i]])
        obdiag = np.diag(obs_prob[action_seq[i], :, obser_seq[i]])
        alfa = alfahat.dot(obdiag)
        belief[i+1] = alfa / np.linalg.norm(alfa, ord=1)

    # remove duplicates
    unique = []
    exclude = []
    for idx, b in enumerate(belief):
        if idx not in exclude:
            for oidx, other in enumerate(belief[idx+1:]):
                if np.allclose(b, other, atol=10e-3, rtol=10e-3):
                    exclude.append(oidx+idx+1)
            unique.append(idx)

    return belief[np.sort(unique)]

rand.seed(42)

# 3 sample beliefs
B = sample_beliefs(M, 3)
print('%i beliefs sampled:' % len(B))
for i in range(len(B)):
    print(B[i])
print('Belief adds to 1?', np.isclose(B[i].sum(), 1.))

# 10 sample beliefs
B = sample_beliefs(M, 100)
print('%i beliefs sampled.' % len(B))


# As an example, you can run the following code on the POMDP from **Activity 1**.
# 
# ```python
# rand.seed(42)
# 
# # 3 sample beliefs
# B = sample_beliefs(M, 3)
# print('%i beliefs sampled:' % len(B))
# for i in range(len(B)):
#     print(B[i])
#     print('Belief adds to 1?', np.isclose(B[i].sum(), 1.))
# 
# # 10 sample beliefs
# B = sample_beliefs(M, 100)
# print('%i beliefs sampled.' % len(B))
# ```
# 
# Output:
# 
# ```
# 2 beliefs sampled:
# [[0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046
#   0.0046 0.0046 0.0046 0.0046 0.0046 0.0046 0.0046]]
# Belief adds to 1? True
# [[0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.3333 0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.3333 0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.3333 0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
#   0.     0.     0.     0.     0.     0.     0.    ]]
# Belief adds to 1? True
# 61 beliefs sampled.
# ```

# ### 3. Solution methods
# 
# In this section you are going to compare different solution methods for POMDPs discussed in class.

# ---
# 
# #### Activity 4
# 
# Write a function `solve_mdp` that takes as input a POMDP represented as a tuple like that of **Activity 1** and returns a `numpy` array corresponding to the **optimal $Q$-function for the underlying MDP**. Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note:** Your function should work for **any** POMDP specified as above. You may reuse code from previous labs.
# 
# ---

# In[4]:


# Add your code here.


# As an example, you can run the following code on the POMDP from **Activity 1**.
# 
# ```python
# Q = solve_mdp(M)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Q-values at state %s:' % M[0][s], Q[s, :])
# print('Best action at state %s:' % M[0][s], np.argmin(Q[s, :]))
# 
# s = rand.randint(len(M[0]))
# print('Q-values at state %s:' % M[0][s], Q[s, :])
# print('Best action at state %s:' % M[0][s], np.argmin(Q[s, :]))
# 
# s = rand.randint(len(M[0]))
# print('Q-values at state %s:' % M[0][s], Q[s, :])
# print('Best action at state %s:' % M[0][s], np.argmin(Q[s, :]))
# ```
# 
# Output:
# 
# ```
# Q-values at state 15Rm11: [4.5168 4.5703 4.5489 4.5489 4.5489]
# Best action at state 15Rm11: 0
# Q-values at state 20Rm12: [3.1507 3.1507 3.242  3.0533 3.1507]
# Best action at state 20Rm12: 3
# Q-values at state 5Rm11: [3.7382 3.7382 3.6718 3.8005 3.7382]
# Best action at state 5Rm11: 2
# ```

# ---
# 
# #### Activity 5
# 
# You will now test the different MDP heuristics discussed in class. To that purpose, write down a function that, given a belief vector and the solution for the underlying MDP, computes the action prescribed by each of the three MDP heuristics. In particular, you should write down a function named `get_heuristic_action` that receives, as inputs:
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The optimal $Q$-function for an MDP (computed, for example, using the function `solve_mdp` from **Activity 4**);
# * A string that can be either `"mls"`, `"av"`, or `"q-mdp"`;
# 
# Your function should return an integer corresponding to the index of the action prescribed by the heuristic indicated by the corresponding string, i.e., the most likely state heuristic for `"mls"`, the action voting heuristic for `"av"`, and the $Q$-MDP heuristic for `"q-mdp"`.
# 
# ---

# In[5]:


# Add your code here.


# For example, if you run your function in the examples from **Activity 3** using the $Q$-function from **Activity 4**, you can observe the following interaction.
# 
# ```python
# for b in B[:10]:
#     
#     if np.all(b > 0):
#         print('Belief (approx.) uniform')
#     else:
#         initial = True
# 
#         for i in range(len(M[0])):
#             if b[0, i] > 0:
#                 if initial:
#                     initial = False
#                     print('Belief: [', M[0][i], ': %.3f' % b[0, i], end='')
#                 else:
#                     print(',', M[0][i], ': %.3f' % b[0, i], end='')
#         print(']')
# 
#     print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
#     print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
#     print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')])
# 
#     print()
# ```
# 
# Output:
# 
# ````
# Belief (approx.) uniform
# MLS action: right; AV action: left; Q-MDP action: left
# 
# Belief: [ 16Rm10 : 0.333, 16Rm11 : 0.333, 16Rm12 : 0.333]
# MLS action: up; AV action: left; Q-MDP action: left
# 
# Belief: [ 17Rm10 : 0.370, 17Rm11 : 0.259, 17Rm12 : 0.370]
# MLS action: left; AV action: left; Q-MDP action: up
# 
# Belief: [ 16Rm10 : 0.348, 16Rm11 : 0.281, 16Rm12 : 0.370]
# MLS action: left; AV action: left; Q-MDP action: left
# 
# Belief: [ 17Rm10 : 0.372, 17Rm11 : 0.226, 17Rm12 : 0.401]
# MLS action: left; AV action: left; Q-MDP action: up
# 
# Belief: [ 17Rm10 : 0.378, 17Rm11 : 0.194, 17Rm12 : 0.428]
# MLS action: left; AV action: left; Q-MDP action: up
# 
# Belief: [ 17Rm10 : 0.419, 17Rm11 : 0.082, 17Rm12 : 0.499]
# MLS action: left; AV action: left; Q-MDP action: up
# 
# Belief: [ 17Rm10 : 0.385, 17Rm11 : 0.110, 17Rm12 : 0.505]
# MLS action: left; AV action: left; Q-MDP action: up
# 
# Belief: [ 17Rm10 : 0.372, 17Rm11 : 0.121, 17Rm12 : 0.506]
# MLS action: left; AV action: left; Q-MDP action: up
# 
# Belief: [ 16Rm10 : 0.349, 16Rm11 : 0.172, 16Rm12 : 0.480]
# MLS action: left; AV action: left; Q-MDP action: left
# ```

# Suppose that the optimal cost-to-go function for the POMDP can be represented using a set of $\alpha$-vectors that have been precomputed for you. 
# 
# ---
# 
# #### Activity 6
# 
# Write a function `get_optimal_action` that, given a belief vector and a set of pre-computed $\alpha$-vectors, computes the corresponding optimal action. Your function should receive, as inputs,
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The set of optimal $\alpha$-vectors, represented as a `numpy` array `av`; the $\alpha$-vectors correspond to the **columns** of `av`;
# * A list `ai` containing the **indices** (not the names) of the actions corresponding to each of the $\alpha$-vectors. In other words, the `ai[k]` is the action index of the $\alpha$-vector `av[:, k]`.
# 
# Your function should return an integer corresponding to the index of the optimal action. 
# 
# Use the functions `get_heuristic_action` and `get_optimal_action` to compute the optimal action and the action prescribed by the three MDP heuristics at the belief 
# 
# ```
# b = np.array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.53,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.47, 0.  , 0.  , 0.  , 0.  , 0.  ,
#                0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])
# ``` 
# 
# and compare the results.
# 
# ---

# In[6]:


# Add your code here.


# <span style="color:blue"> *Add your comments here* </span>

# The binary file `alpha.npz` contains the $\alpha$-vectors and action indices for the Doom environment in the figure abpve. If you compute the optimal actions for the beliefs in the example from **Activity 3** using the $\alpha$-vectors in `alpha.npz`, you can observe the following interaction.
# 
# ```python
# data = np.load('alpha.npz')
# 
# # Alpha vectors
# alph = data['avec']
# 
# # Corresponding actions
# act = list(map(lambda x : M[1].index(x), data['act']))
# 
# # Example alpha vector (n. 3) and action
# print('Alpha-vector n. 3:', alph[:, 3])
# print('Associated action:', M[1][act[3]], '(index %i)' % act[3])
# 
# # Computing the optimal actions
# for b in B[:10]:
#     
#     if np.all(b > 0):
#         print('Belief (approx.) uniform')
#     else:
#         initial = True
# 
#         for i in range(len(M[0])):
#             if b[0, i] > 0:
#                 if initial:
#                     initial = False
#                     print('Belief: [', M[0][i], ': %.3f' % b[0, i], end='')
#                 else:
#                     print(',', M[0][i], ': %.3f' % b[0, i], end='')
#         print(']')
# 
#     print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
#     print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
#     print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')], end='; ')
#     print('Optimal action:', M[1][get_optimal_action(b, alph, act)])
# 
#     print()
# ```
# 
# Output:
# 
# ```
# Alpha-vector n. 3: [2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 3.4092
#  4.5028 4.2367 4.1755 3.7799 2.9735 4.0226 2.5007 2.5007 2.5007 2.5007
#  2.5007 2.5007 2.5007 2.5007 2.5007 3.409  4.4969 4.2326 4.1714 3.7742
#  2.9267 4.0208 4.4343 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007
#  2.5001 2.5    2.5    2.4997 2.4997 2.4998 2.4997 2.4997 2.4998 2.4848
#  1.6861 1.706  2.7364 3.3955 3.2661 3.2101 2.8703 2.4257 2.9628 3.3355
#  2.4998 2.4998 2.4998 2.4998 2.4999 2.5    2.5    2.5001 2.5    2.5
#  2.5    2.4997 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007
#  2.5007 2.6179 6.3866 4.004  3.9705 3.6917 2.9725 4.5128 2.5007 2.5007
#  2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.6176 6.3806 3.9999
#  3.9664 3.6861 2.9256 4.511  5.461  2.5007 2.5007 2.5007 2.5007 2.5007
#  2.5007 2.5007 2.5001 2.5    2.5    2.4997 2.4997 2.4998 2.4997 2.4997
#  2.4998 2.4848 1.6861 1.706  1.9432 5.23   3.009  2.9825 2.7718 2.4497
#  3.406  4.3127 2.4998 2.4998 2.4998 2.4998 2.4999 2.5    2.5    2.5001
#  2.5    2.5    2.5    2.4997 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007
#  2.5007 2.5007 2.5007 2.5858 3.7907 5.5206 4.6264 3.8882 2.9745 3.6297
#  2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5007 2.5855
#  3.7847 5.5165 4.6223 3.8825 2.9278 3.6278 3.7694 2.5007 2.5007 2.5007
#  2.5007 2.5007 2.5007 2.5007 2.5001 2.5    2.5    2.4997 2.4997 2.4998
#  2.4997 2.4997 2.4998 2.4848 1.6861 1.706  1.9148 2.7322 4.5749 3.6836
#  2.9854 2.4007 2.6163 2.7194 2.4998 2.4998 2.4998 2.4998 2.4999 2.5
#  2.5    2.5001 2.5    2.5    2.5    2.4997 0.0009]
# Associated action: left (index 2)
# Belief (approx.) uniform
# MLS action: right; AV action: left; Q-MDP action: left; Optimal action: left
# 
# Belief: [ 16Rm10 : 0.333, 16Rm11 : 0.333, 16Rm12 : 0.333]
# MLS action: up; AV action: left; Q-MDP action: left; Optimal action: down
# 
# Belief: [ 17Rm10 : 0.370, 17Rm11 : 0.259, 17Rm12 : 0.370]
# MLS action: left; AV action: left; Q-MDP action: up; Optimal action: up
# 
# Belief: [ 16Rm10 : 0.348, 16Rm11 : 0.281, 16Rm12 : 0.370]
# MLS action: left; AV action: left; Q-MDP action: left; Optimal action: down
# 
# Belief: [ 17Rm10 : 0.372, 17Rm11 : 0.226, 17Rm12 : 0.401]
# MLS action: left; AV action: left; Q-MDP action: up; Optimal action: up
# 
# Belief: [ 17Rm10 : 0.378, 17Rm11 : 0.194, 17Rm12 : 0.428]
# MLS action: left; AV action: left; Q-MDP action: up; Optimal action: up
# 
# Belief: [ 17Rm10 : 0.419, 17Rm11 : 0.082, 17Rm12 : 0.499]
# MLS action: left; AV action: left; Q-MDP action: up; Optimal action: up
# 
# Belief: [ 17Rm10 : 0.385, 17Rm11 : 0.110, 17Rm12 : 0.505]
# MLS action: left; AV action: left; Q-MDP action: up; Optimal action: up
# 
# Belief: [ 17Rm10 : 0.372, 17Rm11 : 0.121, 17Rm12 : 0.506]
# MLS action: left; AV action: left; Q-MDP action: up; Optimal action: up
# 
# Belief: [ 16Rm10 : 0.349, 16Rm11 : 0.172, 16Rm12 : 0.480]
# MLS action: left; AV action: left; Q-MDP action: left; Optimal action: down
# ```
