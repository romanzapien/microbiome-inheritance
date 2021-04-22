#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: Stochastic inheritance of the microbiome (source code - SDE)
@author: Román Zapién-Campos - 2021
(MPI for Evolutionary Biology - zapien@evolbio.mpg.de)
"""

# Import relevant packages

import numpy as np

import random

from numba import jit

from par_sde import H


# Number of microbial taxa

S = 2

# Declare global variables
    
# Expected change
    
expval = np.zeros((1, S, H), dtype = np.float64)
    
# Square-root of the covariance change
    
cov_sqrt = np.zeros((S, S, H), dtype = np.float64)


# Draw a time from an exponential decaying function

@jit(nopython = True)
def deathtime_sample(time_par, draws):
    
    return np.random.exponential(time_par, draws)


# Draw points from a normal distribution with Mean = 0 and SD = sqrt(dt)

@jit(nopython = True)
def deltaW(S, H, dt):

    return np.random.normal(0.0, np.sqrt(dt), (1, S, H))


# Compute the transition rates of all hosts

@jit()
def compute_rates(x_n, x_o, x_e, b_e, N, m, p):

    # Transition probabilities

    # P[{x_n, x_o, x_e} -> {x_n - 1/N, x_o + 1/N, x_e}]

    t_no = x_n * (m * (1. - p) + (1. - m) * x_o / (b_e * x_e + x_n + x_o))

    # P[{x_n, x_o, x_e} -> {x_n + 1/N, x_o - 1/N, x_e}]

    t_on = x_o * (m * p + (1. - m) * x_n / (b_e * x_e + x_n + x_o))

    # P[{x_n, x_o, x_e} -> {x_n + 1/N, x_o, x_e - 1/N}]

    t_en = x_e * (m * p + (1. - m) * x_n / (b_e * x_e + x_n + x_o))

    # P[{x_n, x_o, x_e} -> {x_n - 1/N, x_o, x_e + 1/N}]

    t_ne = x_n * ((1. - m) * (b_e * x_e) / (b_e * x_e + x_n + x_o))

    # P[{x_n, x_o, x_e} -> {x_n, x_o + 1/N, x_e - 1/N}]

    t_eo = x_e * (m * (1. - p) + (1. - m) * x_o / (b_e * x_e + x_n + x_o))

    # P[{x_n, x_o, x_e} -> {x_n, x_o - 1/N, x_e + 1/N}]

    t_oe = x_o * ((1. - m) * (b_e * x_e) / (b_e * x_e + x_n + x_o))
    
    return t_no, t_on, t_en, t_ne, t_eo, t_oe


# Compute the expected change
    
def expvalue(t_no, t_on, t_en, t_ne, t_eo, t_oe):
    
    # Expected change of the focal taxon
    
    expval[:, 0, :] = (- t_no + t_on + t_en - t_ne) * (1. / N)
    
    # Expected change of the added non-focal (aka 'other') taxa
    
    expval[:, 1, :] = (t_no - t_on + t_eo - t_oe) * (1. / N)
    
    return expval


# Compute the square-root of the covariance change

def covar_sqrt(t_no, t_on, t_en, t_ne, t_eo, t_oe):
    
    # Variance of the focal taxon
    
    cov_nn = (t_no + t_on + t_en + t_ne) * (1. / N**2)
    
    # Covariance of the focal and added non-focal (aka 'other') taxa
    
    cov_no = (- t_no - t_on) * (1. / N**2)
    
    # Variance of the added non-focal (aka 'other') taxa
    
    cov_oo = (t_no + t_on + t_eo + t_oe) * (1. / N**2)

    # Square root of a positive definite 2x2 matrix (ref: E. Allen - Modeling with Itô SDEs, 2007. Springer)
    
    w = np.sqrt(cov_nn * cov_oo - cov_no**2)
    
    d = np.sqrt(cov_nn + cov_oo + 2. * w)
    
    cov_sqrt[0, 0, :] = (cov_nn + w) / d
    
    cov_sqrt[0, 1, :] = cov_no / d
    
    cov_sqrt[1, 0, :] = cov_no / d
    
    cov_sqrt[1, 1, :] = (cov_oo + w) / d
    
    return cov_sqrt


# Solve the multi-dimensional SDE using the Euler-Maruyama method

def itoEuler(xt, time_initial, time_total, time_samples):
    
    # Create arrays to store time-series data
    
    t_timeseries = np.logspace(0, np.log10(time_total), time_samples - 1)
    
    xt_timeseries = np.zeros((S, H, time_samples))
    
    # Store the initial condition
    
    xt_timeseries[:, :, 0] = xt
    
    next_sample = 0
    
    # Draw individual host death times
    
    if t > 0: 
        
        time_hosts_death = deathtime_sample(1. / t, H)
    
    # Time iteration increasing dt each step
    
    for time in np.arange(time_initial + dt, time_total + dt, dt):
        
        if t > 0:
        
            # Check if host(s) die during the time interval [time - dt, time]
        
            hosts_to_die = (time_hosts_death < time)
    
            # If host(s) die update xt and time_hosts_death accordingly
    
            if True in hosts_to_die:
        
                xt, time_hosts_death = host_death(xt, hosts_to_die, time_hosts_death)
            
        # Frequencies of the focal taxon
    
        x_n = xt[:, 0, :]
    
        # Added frequencies of the non-focal (aka 'others') taxa
    
        x_o = xt[:, 1, :]
    
        # Frequency of unoccupied space
        
        x_e = 1. - x_n - x_o
            
        x_e[x_e < 0] = 0
        
        # Transition probabilities of the model
        
        t_no, t_on, t_en, t_ne, t_eo, t_oe = compute_rates(x_n, x_o, x_e, b_e, N, m, p)      
                
        # Deterministic part
        
        f = expvalue(t_no, t_on, t_en, t_ne, t_eo, t_oe)
        
        # Stochastic part
        
        G = covar_sqrt(t_no, t_on, t_en, t_ne, t_eo, t_oe)
        
        # Draw of the noise
        
        dW = deltaW(S, H, dt)
        
        # x(t + 1) as a function of x(t) and deterministic and stochastic changes
        
        xt = xt + f * dt + np.einsum('jlk,ijk->ilk', G, dW)
        
        # Keep the solutions bounded to the interval [0, 1]
        
        xt[xt < 0.] = 0.
        
        xt[xt > 1.] = 1.
        
        # Evaluate if the time-point data has to be stored
        
        if time >= t_timeseries[next_sample]:
            
            # Save data
            
            xt_timeseries[:, :, next_sample + 1] = xt
            
            t_timeseries[next_sample] = time
            
            # Next sample point
            
            next_sample += 1
            
    t_timeseries = np.hstack((0, t_timeseries))
        
    return xt, t_timeseries, xt_timeseries


# Host death

def host_death(xt, hosts_to_die, time_hosts_death):
    
    # Iterate through all the hosts that die
    
    for host_to_die in np.where(hosts_to_die)[0]: # Fix to
    
        if inh:
        
            # Pick a host to reproduce
    
            host_to_reproduce = random.randint(0, H - 1)
            
            if a_n == a_o and b_n == b_o:
    
                # Draw the fraction that will be inherited by the new host and removed from the reproducing host
    
                fraction_inherited = np.random.beta(a_n, b_n)
            
                # Gained with inheritance
    
                xt[:, :, host_to_die] = fraction_inherited * xt[:, :, host_to_reproduce]
    
                # Lost with inheritance 
    
                xt[:, :, host_to_reproduce] = (1. - fraction_inherited) * xt[:, :, host_to_reproduce]
            
            else:
            
                # Draw the fraction that will be inherited by the new host and removed from the reproducing host
    
                fraction_inherited_n = np.random.beta(a_n, b_n)
                
                fraction_inherited_o = np.random.beta(a_o, b_o)
            
                # Gained with inheritance
    
                xt[:, 0, host_to_die] = fraction_inherited_n * xt[:, 0, host_to_reproduce]
                
                xt[:, 1, host_to_die] = fraction_inherited_o * xt[:, 1, host_to_reproduce]
    
                # Lost with inheritance 
    
                xt[:, 0, host_to_reproduce] = (1. - fraction_inherited_n) * xt[:, 0, host_to_reproduce]
                
                xt[:, 1, host_to_reproduce] = (1. - fraction_inherited_o) * xt[:, 1, host_to_reproduce]
            
        else:
            
            xt[:, :, host_to_die] = 0
            
        # Sample a death time for the new host
        
        time_hosts_death[host_to_die] = time_hosts_death[host_to_die] + deathtime_sample(1. / t, 1)
    
    return xt, time_hosts_death


# Solve the system

def solve_host_pop(args):
    
    subdir = args[-1]
    
    x0 = [np.float(arg) for arg in args[-3:-1]]
    
    args = [np.float(arg) for arg in args[:-3]]
    
    global inh, N, m, t, p, b_e, a_n, a_o, b_n, b_o, H, time_sim, time_samples, dt
    
    (inh, N, m, t, p, b_e, a_n, a_o, b_n, b_o, H, time_sim, time_samples, dt, rep) = args
    
    H = int(H)
    
    time_samples = int(time_samples)
    
    # Initial frequencies of the focal and non-focal (aka 'other') taxa
    
    xt = np.zeros((1, S, H), dtype = np.float64)
    
    xt[:, 0, :] = x0[0]
    
    xt[:, 1, :] = x0[1]
        
    # Start of time
    
    time = 0
    
    # Solve the system until the specified total time using the Euler - Maruyama method
    
    xt_last, t_timeseries, xt_timeseries = itoEuler(xt, time, time_sim, time_samples)
    
    if time_samples != 0: np.savez_compressed('output_sde/%s/inh_%i_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i'%(subdir, inh, N, m, t, p, b_e, a_n, a_o, b_n, b_o, rep), timeseries_time = t_timeseries, timeseries_data = xt_timeseries, stationary_dist = xt_last[0,:,:], n_hosts = H, N = N, m = m, t = t, p = p, be = b_e, inh = inh, a_n = a_n, a_o = a_o, b_n = b_n, b_o = b_o, init_cond = x0, time_sim = time_sim, n_timepoints = time_samples)
    
    else: np.savez_compressed('output_sde/%s/inh_%i_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i'%(subdir, inh, N, m, t, p, b_e, a_n, a_o, b_n, b_o, rep), stationary_dist = xt_last[0,:,:], n_hosts = H, N = N, m = m, t = t, p = p, be = b_e, inh = inh, a_n = a_n, a_o = a_o, b_n = b_n, b_o = b_o, init_cond = x0, time_sim = time_sim, n_timepoints = time_samples)


# Linage-taxon: get the times and frequencies of extiction or saturation

def ext_time(xt, time_initial, time_total):
        
    # extinction/saturation time
    
    ext_times = np.ones(H) * np.nan
    
    # Hosts were the linage-taxon is present
    
    h_prev = np.ones(H)
    
    h_current = np.ones(H)
    
    # Next timestep
    
    time = time_initial + dt
    
    # Time iteration increasing dt each step
    
    while h_current.sum() > 0 and time <= time_total:
        
        # Frequencies of the focal taxon
    
        x_n = xt[:, 0, :]
    
        # Added frequencies of the non-focal (aka 'others') taxa
    
        x_o = xt[:, 1, :]
    
        # Frequency of unoccupied space
        
        x_e = 1. - x_n - x_o
            
        x_e[x_e < 0] = 0
        
        # Transition probabilities of the model
        
        t_no, t_on, t_en, t_ne, t_eo, t_oe = compute_rates(x_n, x_o, x_e, b_e, N, m, p)      
                
        # Deterministic part
        
        f = expvalue(t_no, t_on, t_en, t_ne, t_eo, t_oe)
        
        # Stochastic part
        
        G = covar_sqrt(t_no, t_on, t_en, t_ne, t_eo, t_oe)
        
        # Draw of the noise
        
        dW = deltaW(S, H, dt)
        
        # x(t + 1) as a function of x(t) and deterministic and stochastic changes
        
        xt = xt + f * dt + np.einsum('jlk,ijk->ilk', G, dW)
        
        # Keep the solutions bounded to the interval [0, 1]
        
        xt[xt < 0.] = 0.
        
        xt[xt > 1.] = 1.
        
        # Check if linage-taxon is extinct
        
        h_current[xt[0,0,:] < 1. / N] = 0
                
        status = (h_prev != h_current)
        
        if status.any():
            
            ext_times[status] = time
            
            h_prev[xt[0, 0, :] < 1. / N] = 0
            
            print(h_current.sum(), time)
                                
        # Next timestep
        
        time += dt
        
    return ext_times


# Figure 5 data: linage-taxon 

def solve_linage_taxa(args):
    
    subdir = args[-1]
    
    x0 = [np.float(arg) for arg in args[-3:-1]]
    
    args = [np.float(arg) for arg in args[:-3]]
    
    global N, m, p, b_e, H, time_sim, dt
    
    (N, m, b_e, H, time_sim, dt, rep, frac) = args
    
    p = 0
    
    H = int(H)
    
    # Initial frequencies of the focal and non-focal (aka 'other') taxa
    
    if x0[0] == -1 and x0[1] == -1:
    
        x_e = m / (1. - b_e)
    
        x0 = [frac * (1. - x_e), (1. - frac) * (1. - x_e)]
    
    xt = np.zeros((1, S, H), dtype = np.float64)
        
    xt[:, 0, :] = x0[0]
    
    xt[:, 1, :] = x0[1]
    
    # Start of time
    
    time = 0
    
    # Solve the system until the specified total time using the Euler - Maruyama method
    
    ext_times = ext_time(xt, time, time_sim)
        
    np.savez_compressed('output_sde/%s/N_%.0e_be_%.0e_m_%.0e_frac_%.0e_load_%.0e_%i'%(subdir, N, b_e, m, frac, x0[0]+x0[1], rep), ext_times = ext_times, n_hosts = H, N = N, m = m, t = 0, p = p, be = b_e, frac_linage_taxa = frac, init_cond = x0, time_sim = time_sim)
    
    # return ext_sat_times