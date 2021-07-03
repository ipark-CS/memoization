+++
author = "ipark"
title = "Deep Reinforcement Learning"
date =  2021-07-02
draft =  false
type = "amazon"
layout = "amazon"
description = ""
tags = ["Amazon", "Applied Scientist", "DRL",
]
+++

---

### (D)RL
1. RL is one of AI methods 
2. The main characters of RL are the **agent** and the **environment**. 
3. RL is the study of agents and how they learn by **trial and error** (i.e interaction with environment).
4. RL formalizes the idea that **rewarding** or punishing an agent for its behavior makes it more 
likely to repeat or forego that behavior in the future.  

---

### Environment
the world that the agent lives in and interacts with. 
At every step of interaction, the agent sees a (possibly partial) observation of 
the state of the world, and then decides on an action to take. The environment 
changes when the agent acts on it, but may also change on its own.

---

### Agent
The agent also perceives a reward signal from the environment, a number that tells 
it how good or bad the current world state is. The goal of the agent is to maximize 
its cumulative reward, called return. Reinforcement learning methods are ways that 
the agent can learn behaviors to achieve its goal.

---

### States and Observations
A state s is a complete description of the state of the world. There is no 
information about the world which is hidden from the state. An observation o is 
a partial description of a state, which may omit information.

---


### Action Spaces
Different environments allow different kinds of actions. The set of all valid 
actions in a given environment is often called the action space. Some environments, 
like Atari and Go, have discrete action spaces, where only a finite number of moves 
are available to the agent. Other environments, like where the agent controls a 
robot in a physical world, have continuous action spaces. In continuous spaces, 
actions are real-valued vectors.

---


### Policies
A policy is a rule used by an agent to decide what actions to take. 

---


### Parameterized Policies
In deep RL, we deal with parameterized policies: policies whose outputs are 
computable functions that depend on a set of parameters (eg the weights and biases 
of a neural network) which we can adjust to change the behavior via some optimization algorithm.
>#### Sampling: 
>Given the mean action 
$$\mu_{\theta}(s)$$ and standard deviation $$\sigma_{\theta}(s)$$, 
and a vector z of noise from a spherical Gaussian ($$z \sim \mathcal{N}(0, I)$$), 
an action sample can be computed with $$a = \mu_{\theta}(s) + \sigma_{\theta}(s) \odot z$$,
where $$\odot$$ denotes the elementwise product of two vectors. 
>#### Log-Likelihood: 
>The log-likelihood of a k -dimensional action a, for a diagonal Gaussian with mean 
>$$\mu = \mu_{\theta}(s)$$ and standard deviation $$\sigma = \sigma_{\theta}(s)$$, is given by
>$$\log \pi_{\theta}(a|s) = -\frac{1}{2}\left(\sum_{i=1}^k \left(\frac{(a_i - \mu_i)^2}{\sigma_i^2} + 2 \log \sigma_i \right) + k \log 2\pi \right)$$.

---

### Trajectories
A trajectory $$\tau$$ is a sequence of states and actions in the world, $$\tau = (s_0, a_0, s_1, a_1, ...)$$.

---

### Reward and Return
The reward function $$R$$ depends on the current state of the world, the action 
just taken, and the next state of the world:

$$r_t = R(s_t, a_t, s_{t+1}) \sim R(s_t)$$ or $$R(s_t,a_t)$$.

>#### Infinite-horizon discounted return: 
>sum of all rewards ever obtained by the agent, 
>but discounted by how far off in the future they're obtained. 
>This formulation of reward includes a discount factor $$\gamma \in (0,1)$$:
>
>$$R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t$$.


---

### Bellman Equations 
The value of your starting point is the reward to get from being there, 
plus the value of wherever you land next.

>#### Bellman Equations for On-policy Value Functions:
>$$ V^{\pi}(s) = E_{a \sim \pi \\ s'\sim P}\Bigl[{r(s,a) + \gamma V^{\pi}(s')}\Bigr]$$
>$$ Q^{\pi}(s,a) = E_{s'\sim P}\Bigl[{r(s,a) + \gamma E_{a'\sim \pi}{Q^{\pi}(s',a')}}\Bigr],$$\
>where $$s' \sim P(\cdot |s,a) \sim P$$ (the next state $$s'$$) 
>is sampled from the environment’s transition rules; 
>$$a \sim \pi(\cdot|s) \sim \pi$$ 
>and $$a' \sim \pi(\cdot|s') \sim \pi$$.
>
>#### Bellman Equations for Optimal Value Functions:
>
>$$ V^*(s) = \max_a E_{s'\sim P}\Bigl[{r(s,a) + \gamma V^*(s')}\Bigr]$$
>
>$$ Q^*(s,a) = E_{s'\sim P}\Bigl[{r(s,a) + \gamma \max_{a'} Q^*(s',a')}\Bigr].$$

---

### Advantage Functions
$$A^{\pi}(s,a)$$ corresponding to a policy $$\pi$$ describes how much better 
it is to take a specific action $$a$$ in state $$s$$, relatively, over randomly 
selecting an action according to $$\pi(\cdot|s)$$. 
$$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$$.

---

### RL Problems
The goal in RL is to select a policy which maximizes expected return when the 
agent acts according to it.

>#### Probability distributions over T-step trajectories:
> $$P(\tau|\pi) = \rho_0 (s_0) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi(a_t | s_t)$$.
>
>#### Expected return:
> $$J(\pi) = \int_{\tau} P(\tau|\pi) R(\tau) = E_{\tau\sim \pi}{\Bigl[R(\tau)}\Bigr]$$.
>
>#### Central optimization problem in RL:
> $$\pi^* = \arg \max_{\pi} J(\pi)$$,
> where $$\pi^*$$ is the optimal policy.

---

### Value Functions
The expected return if you start in that state or state-action pair, and then 
act according to a particular policy forever after. Value functions are used, 
one way or another, in almost every RL algorithm.

>#### On-Policy V-Function: 
>$$V^{\pi}(s) = E_{\tau \sim \pi}{\Bigl[R(\tau)\left| s_0 = s\right.\Bigr]}$$
>
>#### On-Policy Q-Function:
>$$Q^{\pi}(s,a) = E_{\tau \sim \pi}{\Bigl[R(\tau)\left| s_0 = s, a_0 = a\right.\Bigr]}$$
>
>#### Optimal V-Function:
>$$V^*(s) = \max_{\pi} E_{\tau \sim \pi}{\Bigl[R(\tau)\left| s_0 = s\right.\Bigr]}$$
>
>#### Optimal Q-Function:
>$$Q^*(s,a) = \max_{\pi} E_{\tau \sim \pi}{\Bigl[R(\tau)\left| s_0 = s, a_0 = a\right.\Bigr]}$$


---

