#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 1: Markov chains
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab1-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).
# 
# ### 1. The Markov chain model
# 
# Consider once again the shuttle modeling problem described in the Homework and for which you wrote a Markov chain model:
# 
# <img src="shuttle.png" width="800px">
# 
# Recall that your chain should describe the motion of the single shuttle traveling the network, where: 
# 
# * All stops are considered similar, in terms of shuttle behavior;
# * At each moment, there is a single shuttle traversing the city;
# * When at I.S.T. TagusPark, the shuttle will go directly to Sete Rios with a probability of 70%, and to Oeiras with a 30% probability. Similarly, when at Sete Rios, the shuttle will go directly to I.S.T. Alameda with a 50% probability, and through Praça de Londres with a 50% probability.
# 
# In this first activity, you will implement your Markov chain model in Python. You should label the different shuttle stops as `'0'`, `'1'`, `'2'`, `'3'`, `'4'` (note that these are strings), from left to right in the diagram above. For example, "I.S.T. TagusPark" corresponds to `'0'`, while "Praça de Londres" corresponds to `'3'`.

# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_chain` that receives **no input arguments** and returns, as output, a two-element tuple corresponding to the Markov chain, where:
# 
# * ... the first element is a tuple containing an enumeration of the state-space (i.e., each element of the tuple corresponds to a state of the chain, represented as a string).
# * ... the second element is a `numpy` array corresponding to the transition probability matrix for the chain.
# 
# **Note**: Don't forget to import `numpy`.
# 
# ---

# In[2]:


import numpy as np

def load_chain():
    state_space = ('0', '1', '2', '3', '4')
    tp_matrix = np.array([[0, 0.3, 0.7, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0.5, 0.5],
                          [0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0]])

    return state_space, tp_matrix

M = load_chain()

print('Number of states:', len(M[0]))
print('Type of states:', type(M[0][0]))
print('\nTransition probability matrix (type):', type(M[1]))
print('Transition probability matrix (dimension):', M[1].shape)


# We provide below an example of application of the function, that you can use as a first "sanity check" for your code. Note, however, that the fact that you can replicate the examples below is not indicative that your code is correct.
# 
# ```python
# print('Number of states:', len(M[0]))
# print('Type of states:', type(M[0][0]))
# print('\nTransition probability matrix (type):', type(M[1]))
# print('Transition probability matrix (dimension):', M[1].shape)
# ```
# 
# Output:
# ```
# Number of states: 5
# Type of states: <class 'str'>
# 
# Transition probability matrix (type): <class 'numpy.ndarray'>
# Transition probability matrix (dimension): (5, 5)
#  ```

# In the next activity, you will use the Markov chain model to evaluate the likelihood of any given path for the bot.
# 
# ---
# 
# #### Activity 2.
# 
# Write a function `prob_trajectory` that receives, as inputs, 
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a trajectory, corresponding to a sequence of states (i.e., a tuple or list of strings, each string corresponding to a state).
# 
# Your function should return, as output, a floating point number corresponding to the probability of observing the provided trajectory, taking the first state in the trajectory as initial state. 
# 
# ---

# In[3]:


def prob_trajectory(M, trajectory):
    path = np.array([], dtype=int)

    # First we need to create a path, since the states are characters and might
    # not correspond to indexes we must go to the state space (M[0])and get the right index

    for n in trajectory:
        path = np.append(path, M[0].index(n))

    # Since it is a sliding window with width equals 2 we can for simplification index the transition matrix to get
    # all the probabilities we want. The x values are the path minus the last state path[:-1] and y values start
    # at he second entry path[1:] and like this for example the path (1,2,3,4) becomes x = (1,2,3) and y = (2,3,4)
    # feeding this as an index  to an np.ndarray we get the pairs (1,2); (2,3); (3,4) as intended
    individual_prob = M[1][path[:-1], path[1:]]

    return individual_prob.prod()

print('Prob. of trajectory (1, 2, 3):', prob_trajectory(M, ('1', '2', '3')))
print('Prob. of trajectory (4, 0, 1, 2, 3):', prob_trajectory(M, ('4', '0', '1', '2', '3')))
print('Prob. of trajectory (4, 0, 4):', prob_trajectory(M, ('4', '0', '4')))
print('Prob. of trajectory (0, 2, 4):', prob_trajectory(M, ('0', '2', '4')))


# Example of application of the function with the chain $M$ from Activity 1.
# 
# ```python
# print('Prob. of trajectory (1, 2, 3):', prob_trajectory(M, ('1', '2', '3')))
# print('Prob. of trajectory (4, 0, 1, 2, 3):', prob_trajectory(M, ('4', '0', '1', '2', '3')))
# print('Prob. of trajectory (4, 0, 4):', prob_trajectory(M, ('4', '0', '4')))
# print('Prob. of trajectory (0, 2, 4):', prob_trajectory(M, ('0', '2', '4')))
# ```
# 
# Output:
# ```
# Prob. of trajectory (1, 2, 3): 0.5
# Prob. of trajectory (4, 0, 1, 2, 3): 0.15
# Prob. of trajectory (4, 0, 4): 0.0
# Prob. of trajectory (0, 2, 4): 0.35
# ```
# 
# Note that your function should work with **any** Markov chain that is specified as a tuple like the one from Activity 1.

# ### 2. Stability

# The next activities explore the notion of *stationary distribution* for the chain, a central concept in the the PageRank algorithm.
# 
# ---
# 
# #### Activity 3
# 
# Write a function `stationary_dist` that receives, as input, a Markov chain in the form of a tuple like the one returned by the function in Activity 1. Your function should return, as output, a `numpy` array corresponding to a row vector containing the stationary distribution for the chain.
# 
# **Note:** The stationary distribution is a *left* eigenvector of the transition probability matrix associated to the eigenvalue 1. As such, you may find useful the numpy function `numpy.linalg.eig`. Also, recall that the stationary distribution is *a distribution*.
# 
# ---

# In[4]:


def stationary_dist(M):
    # Left eigenvalues of A are equal to right eigenvalues of A^T
    eigval, eigvec = np.linalg.eig(M[1].T)

    # Get the index of the eigenvector associated with eigenvalue 1
    # Note: in our experiments eigenvalue was 1.0000000000000002+0j so we use function is close with an abs tolerence
    index = np.where(np.isclose(eigval, 1 + 0j, atol=1e-10))

    # Get the vector
    dist = eigvec[:, index].T
    dist = dist.real

    # Normalize
    dist = dist / dist.sum()

    return dist

u_star = stationary_dist(M)

print('Stationary distribution:')
print(u_star)

u_prime = u_star.dot(M[1])

print('\nIs u* * P = u*?', np.all(np.isclose(u_prime, u_star)))


# Example of application of the function with the chain $M$ from Activity 1.
# 
# ```python
# u_star = stationary_dist(M)
# 
# print('Stationary distribution:')
# print(u_star)
# 
# u_prime = u_star.dot(M[1])
# 
# print('\nIs u* * P = u*?', np.all(np.isclose(u_prime, u_star)))
# ```
# 
# Output:
# ```
# Stationary distribution:
# [0.263 0.079 0.263 0.132 0.263]
# 
# Is u* * P = u*? True
# ```
# 
# All numbers above have been rounded to 3 decimal cases. You **should not** round your results, but can use the numbers above as a comparison.

# To complement Activity 3, you will now empirically establish that the chain is ergodic, i.e., no matter where the bot starts, its visitation frequency will eventually converge to the stationary distribution.
# 
# ---
# 
# #### Activity 4.
# 
# Write a function `compute_dist` that receives, as inputs, 
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a row vector (a numpy array) corresponding to the initial distribution for the chain;
# * ... an integer $N$, corresponding to the number of steps that the bot is expected to take.
# 
# Your function should return, as output, a row vector (a `numpy` array) containing the distribution after $N$ steps of the chain.
# 
# ---

# In[5]:


def compute_dist(chain, v, t):
    
    p = np.linalg.matrix_power(chain[1], t)
    finaldist = v.dot(p)

    return finaldist

# Number of states
nS = len(M[0])

# Initial, uniform distribution
u = np.ones((1, nS)) / nS

# Distrbution after 100 steps
v = compute_dist(M, u, 10)
print('\nIs u * P^10 = u*?', np.all(np.isclose(v, u_star)))

# Distrbution after 1000 steps
v = compute_dist(M, u, 100)
print('\nIs u * P^100 = u*?', np.all(np.isclose(v, u_star)))


# Example of application of the function with the chain $M$ from Activity 1.
# 
# ```python
# # Number of states
# nS = len(M[0])
# 
# # Initial, uniform distribution
# u = np.ones((1, nS)) / nS
# 
# # Distrbution after 100 steps
# v = compute_dist(M, u, 10)
# print('\nIs u * P^10 = u*?', np.all(np.isclose(v, u_star)))
# 
# # Distrbution after 1000 steps
# v = compute_dist(M, u, 100)
# print('\nIs u * P^100 = u*?', np.all(np.isclose(v, u_star)))
# ```
# 
# Output:
# ```
# Is u * P^10 = u*? False
# 
# Is u * P^100 = u*? True
# ```

# Is the chain ergodic? Justify, based on the results above.

# <span style="color:blue">Yes the distribution is ergodic because even if with ten steps it does not aproximate 
#     the stationary distribuition with 100 steps it does, this means it will eventually reach
#     a stationary distribuition wich is the requirement to be called an ergodic chain.</span>

# ### 3. Simulation
# 
# In this part of the lab, you will *simulate* the actual bot, and empirically compute the visitation frequency of each state.

# ---
# 
# #### Activity 5
# 
# Write down a function `simulate` that receives, as inputs, 
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a row vector (a `numpy` array) corresponding to the initial distribution for the chain;
# * ... an integer $N$, corresponding to the number of steps that the bot is expected to take.
# 
# Your function should return, as output, a tuple containing a trajectory of $N$ steps obtained from the initial distribution provided. Each element in the tuple should be a string corresponding to a state index.
# 
# ---
# 
# **Note:** You may find useful to import the numpy module `numpy.random`.

# In[6]:


def simulate(chain, v, t):
    weightedpick = np.cumsum(v)
    trajectory = ()

    for i in range(t):
        # Dice roll for state
        state_idx = weightedpick.searchsorted(np.random.uniform(0, weightedpick[-1]))
        trajectory += (chain[0][state_idx],)

        # Get dist of current state
        row = chain[1][state_idx]
        weightedpick = np.cumsum(row)

    return trajectory

# Number of states
nS = len(M[0])

# Initial, uniform distribution
u = np.ones((1, nS)) / nS

np.random.seed(42)

# Simulate short trajectory
traj = simulate(M, u, 10)
print(traj)

# Simulate a long trajectory
traj = simulate(M, u, 10000)


# Example of application of the function with the chain $M$ from Activity 1.
# 
# ```python
# # Number of states
# nS = len(M[0])
# 
# # Initial, uniform distribution
# u = np.ones((1, nS)) / nS
# 
# np.random.seed(42)
# 
# # Simulate short trajectory
# traj = simulate(M, u, 10)
# print(traj)
# 
# # Simulate a long trajectory
# traj = simulate(M, u, 10000)
# ```
# 
# Output:
# ```
# ('1', '2', '4', '0', '1', '2', '3', '4', '0', '2')
# ```
# 
# Note that, even if the seed is fixed, it is possible that your trajectories are slightly different.

# ---
# 
# #### Activity 6
# 
# Draw a histogram of the $10,000$-step trajectory generated in the example of Activity #5 using the function `hist` from the module `matplotlib.pyplot`. Make sure that the histogram has one bin for each state. Compare the relative frequencies with the result of Activity #3.
# 
# **Note**: Don't forget to load `matplotlib`. 
# 
# **Note 2**: Recall that the states in the trajectory from Activity #5 consist of strings, which should be converted to state indices to match the entries in the distribution computed in Activity #3.
# 
# ---

# In[9]:


import matplotlib.pyplot as plt

def histosim(M, trajectory):
    states_len = len(M[0])
    path = []

    for n in trajectory:
        path.append(M[0].index(n))

    # One bin for each state -0.5 to center
    bins = np.arange(0, states_len + 1) - 0.5
    n, bins, patches = plt.hist(path, bins, rwidth=0.75, color='blue')
    dist = n / sum(n)
    plt.xticks(np.arange(states_len), labels=list(M[0]))

    # Anotate relative freq in histogram
    for i, rfreq in enumerate(dist):
        plt.annotate("({:.3f})".format(rfreq), xy=(i - 0.2, -0.1), xycoords=('data', 'axes fraction'))

    plt.show()
    return dist


rfreq = histosim(M, traj)
print("Stationary Distribuition: ", u_star)
print("Ssimulation Distribuition: ", rfreq)


# In[ ]:




