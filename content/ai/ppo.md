+++
author = "ipark"
title = "Proximal Policy Gradient"
date =  2021-07-02
draft =  false
type = "ai"
layout = "ai"
description = ""
tags = ["AI", "PPO", "DRL"
]
+++

---

### Policy Gradient
* Consider a stochastic, parameterized policy, $$\pi_{\theta}$$. 
* Aim: maximize the expected return $$J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}{\Bigl[R(\tau)\Bigr]}$$. 
* Optimize the policy by gradient ascent, 
$$\theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}$$.
* $$\nabla_{\theta} J(\pi_{\theta})$$ = gradient of policy performance = policy gradient 
* To use this algorithm, need an numerical expression in two steps:
    1) deriving the analytical gradient of policy performance, which turns out to have the form of an expected value, and then 
    2) forming a sample estimate of that expected value, which can be computed with data from a finite number of agent-environment interaction steps.
>#### Policy Gradient Expression for grad-log-prob:
>$$\nabla_{\theta} J(\pi_{\theta}) = \nabla_{\theta} E_{\tau \sim \pi_{\theta}}{\Bigl[R(\tau)\Bigr]}  = ...=  E_{\tau \sim \pi_{\theta}}{\Bigl[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)\Bigr]}$$ 

$$...=\nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau)  $$\
$$...= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau)  $$\
$$...= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) R(\tau) $$\
$$...= E_{\tau \sim \pi_{\theta}}{\Bigl[\nabla_{\theta} \log P(\tau|\theta) R(\tau)\Bigr]} $$

>#### Policy Gradient is an **expectation**, thus estimate it with a sample mean: 
>* If we collect a set of trajectories $$ \mathcal{D} = \\{\tau_i\\}_{i=1,...,N}$$ 
>
> where each trajectory is obtained by letting the agent act in the environment 
> using the policy $$\pi_{\theta}$$, 
> * The policy gradient can be estimated with
> $$\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)$$,
>
> where $$|\mathcal{D}|$$ is the number of trajectories in $$\mathcal{D}$$ (here, N).
>

* Assuming that we have represented our policy in a way which allows us to calculate 
$$\nabla_{\theta} \log \pi_{\theta}(a|s)$$, 
and if we are able to run the policy in the environment to collect the trajectory dataset, we can compute the policy gradient and take an update step.

---

### Implementing the Policy Gradient
1. Making the Policy Network.
```
# make core of policy network
logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

# make function to compute action distribution
def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)

# make action selection function (outputs int actions, sampled from policy)
def get_action(obs):
    return get_policy(obs).sample().item()
```

2. Making the Loss Function.\
When the right data (a set of state, action, weight tuples) collected from  
while acting according to the policy, the gradient of this loss is equal to the policy gradient. 
```
# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()
```

> Note that the **loss function** here is different from supervised learning as following:
>1. In Policy Gradient, the data distribution depends on the parameters we aim to optimize, i.e. the data 
must be sampled on the most recent policy. 
(cf. In supervised learning, a loss function is usually defined on a fixed data distribution)
>2. In Policy Gradient, the loss function doesn't measure performance, is only useful when evaluated 
at the current parameters.  Note that what we care about is expected return,  $$J(\pi_{\theta})$$.
(cf. In supervised learning, a loss function usually evaluates the performance metric that we care about)
