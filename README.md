# jax-experiments

Implementation in Jax of Impala based algorithms

## QRDQN.py : 
Quantil regression algorithm (see https://arxiv.org/abs/1710.10044 for more details) with prioritized experience replay with sum and min tree (https://arxiv.org/abs/1511.05952).

## V_TRACE : 

V-trace (https://arxiv.org/abs/1802.01561) algorithm with the possibility to using Expexted-Emphatic n-steps TD learning (https://arxiv.org/abs/2107.05405 but now this doesn't work), thrust-region sheme (https://arxiv.org/abs/1909.11583), auto-tuned KL-divergence contraints (https://arxiv.org/abs/1909.12238) and auto-tuned Entropy hyperparameter

