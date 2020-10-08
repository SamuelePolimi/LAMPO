# LAMPO

<p float="middle">
  <a href="https://www.youtube.com/watch?v=LKtnzc4TV98"> <img src="img/cartpole.gif" width="400" hspace="10"> </a>
</p>

## Motivation

Current state-of-the-art robot-learning algorithm, divide the learning in two phases: a first imitation-learning from demonstration and a subsequent reinforcement learning policy improvement. 
Usually, the representation of the policy (movement primitives) is high-dimensional, and makes the policy improvement inefficient.
We advance the state-of-the-art by proposing two significant improvement
1. A dimensionality reduction performed on the movement's parameter space via Mixture of Probabilistic Principal Component Analyzers (MPPCA)
2. An off-policy improvement that
   1. uses self-normalized importance sampling to correct the distribution shift
   2. uses closed-form kl bounds of the policy and of the context distribution
