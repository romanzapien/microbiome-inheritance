#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: Stochastic inheritance of the microbiome (parameters - SDE)
@author: Román Zapién-Campos - 2021
(MPI for Evolutionary Biology - zapien@evolbio.mpg.de)
"""

# Carrying capacity for microbes

N = 1E5

# Probability of migration from the pool of colonizers

m = 1E-2

# Frequency of the focal taxon in the pool of colonizers

p = 1E-2

# Probability of not-occupation of empty-space by microbes

b_e = 1E-1

# Probability of host death

t = 1E-4

# Number of hosts in the population

H = int(1E4)

# Initial frequency of the focal and added non-focal (aka 'other') taxa inside each host

x0 = [0., 0.]

# Time to simulate

time_sim = 1E7

# Resolution of the time-steps to compute

dt = 0.1

# Number of time-samples to store

time_samples = int(1E3)

# Is there microbial inheritance?

inh = True

# 1st parameter of the Beta-Binomial distribution

a_n = a_o = 1.
            
# 2nd parameter of the Beta-Binomial distribution

b_n = b_o = 1E1