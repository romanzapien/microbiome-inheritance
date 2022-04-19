#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: Stochastic Inheritance of the microbiome (source code - figures)
@author: Román Zapién-Campos - 2021
(MPI for Evolutionary Biology - zapien@evolbio.mpg.de)
"""

# Import packages
import numpy as np

import matplotlib.pyplot as mp

import matplotlib.ticker as mtick

import matplotlib.gridspec as gridspec

import matplotlib.cm as cm

import matplotlib.colors as colors

from matplotlib.lines import Line2D

from numba import jit

from scipy.stats import gaussian_kde

import gc

from datetime import datetime


def diff_prob(data_modified, data_baseline, N, observable):
    
    n_hosts = 10000
    
    if observable == 'prob_col': return (data_modified >= 1./N).sum() / n_hosts - (data_baseline >= 1./N).sum() / n_hosts

    elif observable == 'exp_value': return (data_modified).sum() / n_hosts - (data_baseline).sum() / n_hosts

    elif observable == 'dif_dist':

        counts = np.zeros((2, int(N + 1)))

        for n in np.arange(0, int(N + 1)): counts[:, n] = [(data_modified == n).sum() / n_hosts, (data_baseline == n).sum() / n_hosts]
            
        return abs(counts[0, :] - counts[1, :]).sum() / 2.


@jit
def argmax_jit(array, threshold):
    
    return np.argmax(array >= threshold, axis = 0)


def average_trajectories(N, time, data, time_sim, n_timepoints, observable):
    
    n_hosts = 10000

    X = np.logspace(0, np.log10(time_sim), n_timepoints)
    
    Y = np.zeros(n_timepoints)

    if observable != 'dif_dist':
    
        for i in range(n_timepoints):
        
            inds = argmax_jit(time, X[i])
                    
            if observable == 'prob_col': Y[i] = (data[range(n_hosts), inds] >= 1. / N).sum() / n_hosts
        
            elif observable == 'exp_value': Y[i] = (data[range(n_hosts), inds]).sum() / n_hosts
        
    elif observable == 'dif_dist':

        time_baseline = time[0]
            
        time_modified = time[1]

        for i in range(n_timepoints):
        
            inds_baseline = argmax_jit(time_baseline, X[i]) 
            
            inds_modified = argmax_jit(time_modified, X[i])
        
            data_baseline = data[0][inds_baseline, range(n_hosts)]
            
            data_modified = data[1][inds_modified, range(n_hosts)]
            
            Y[i] = diff_prob(data_modified, data_baseline, N, 'dif_dist')
    
    return X, Y


def fig_timeseries(ax, N, m, t, p, Be, AB_i, rep, taxa, observable, label_text, legend_loc, subdir, n_timepoints, xlim, ylim, title):
    
    ax.tick_params(axis = 'both', which = 'both', direction = 'in')
    
    if observable != 'dif_dist':  colors = ['orange', 'red', 'green', 'blue']
    
    elif observable == 'dif_dist': colors = ['purple', 'darkcyan']
    
    c = 0
    
    n_hosts = 10000
        
    for be in Be:
        
        name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, p, be, AB_i[0], AB_i[1], AB_i[2], AB_i[3], rep)
        
        dataset_modified = np.load(name_modified)

        timesim_modified = dataset_modified['time_sim']
        
        time_modified = dataset_modified['timeseries_time']
        
        if taxa == 'focal': data_modified = dataset_modified['timeseries_data'][0, :, :]
        
        elif taxa == 'all': data_modified = dataset_modified['timeseries_data'].sum(0)
        
        if observable != 'dif_dist': 
            
            X_modified, Y_modified = average_trajectories(N, time_modified, data_modified, timesim_modified, n_timepoints, observable)

            if taxa == 'focal': modified_label = r'inh., $\alpha_0 = %.1f$'%be
            
            elif taxa == 'all': modified_label = r'inh., $\alpha_0 = %.1f$'%be

            ax.plot(X_modified, Y_modified, label = modified_label, color = colors[c])

        elif observable == 'dif_dist': 
            
            X_modified, Y_modified = average_trajectories(N, [time_baseline, time_modified], [data_baseline, data_modified], timesim_modified, n_timepoints, observable)

            ax.plot(X_modified, Y_modified, label = r'$\alpha_0 = %.1f$'%be, color = colors[c])

        c += 1
    
        gc.collect()
        
    for be in Be:
        
        name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, p, be, 0, 0, 0, 0, rep)
    
        dataset_baseline = np.load(name_baseline)
    
        timesim_baseline = dataset_baseline['time_sim']
    
        time_baseline = dataset_baseline['timeseries_time']
            
        if taxa == 'focal': data_baseline = dataset_baseline['timeseries_data'][0, :, :]
        
        elif taxa == 'all': data_baseline = dataset_baseline['timeseries_data'].sum(0)
        
        if observable != 'dif_dist':
    
            X_baseline, Y_baseline = average_trajectories(N, time_baseline, data_baseline, timesim_baseline, n_timepoints, observable)
    
            if taxa == 'focal': baseline_label = r'no inh., $\alpha_0 = %.1f$'%be
            
            elif taxa == 'all': baseline_label = r'no inh., $\alpha_0 = %.1f$'%be
    
            ax.plot(X_baseline, Y_baseline, label = baseline_label, color = colors[c])
    
            c += 1
            
        print('be: %.3f, bn: %.3f'%(be, AB_i[2]))
            
    ax.set_xlabel(r'time', fontsize = 16)
    
    if observable == 'prob_col': 
        
        if taxa == 'focal':
            
            ax.set_ylabel('prob. of occurrence\n'+r'($P[x_1 > 0]$)', fontsize = 16)
        
        elif taxa == 'all': 
            
            ax.set_ylabel('prob. of occurrence\n'+r'($P[x_i + o_i > 0]$)', fontsize = 16)
        
    elif observable == 'exp_value': 
        
        if taxa == 'focal': 
            
            ax.set_ylabel('average frequency\n'+r'($E[x_1]$)', fontsize = 16)
        
        elif taxa == 'all': 
                        
            ax.set_ylabel('average microbial load\n'+r'($E[x_i + o_i]$)', fontsize = 16)

        ax.set_yscale('log')
        
    elif observable == 'dif_dist': 
        
        if taxa == 'focal': ax.set_ylabel(r'$\sum|\Delta\Phi[x_1]|/2$', fontsize = 16)
        
        elif taxa == 'all': ax.set_ylabel(r'$\sum|\Delta\Phi[\sum x_i]|/2$', fontsize = 16)
        
    ax.set_xlim(xlim[0], xlim[1])
    
    ax.set_ylim(ylim[0], ylim[1])
    
    ax.set_xscale('log')
    
    ax.xaxis.set_ticks_position('both')
    
    ax.yaxis.set_ticks_position('both')
    
    ax.set_title(title, fontsize = 16)
    
    ax.text(-0.2, 1.06, label_text, transform = ax.transAxes, fontsize = 25, fontweight = 'bold')
    
    if taxa == 'focal': legend_title = r'faster col. $\alpha_0 \to 0$'
    
    elif taxa == 'all': legend_title = r'faster col. $\alpha_0 \to 0$'
        
    ax.legend(fontsize = 10, loc = legend_loc, framealpha = 0.6, title = legend_title)
    
    return ax


def fig_distribution(ax, N, m, t, p, be, AB_i, rep, taxa, observable, label_text, legend_loc, subdir):
    
    ax.tick_params(axis = 'both', which = 'both', direction = 'in')
    
    colors = ['green', 'orange']
    
    n_hosts = 10000
        
    name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, p, be, 0, 0, 0, 0, rep)
                    
    name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, p, be, AB_i[0], AB_i[1], AB_i[2], AB_i[3], rep)

    dataset_baseline = np.load(name_baseline)
                
    dataset_modified = np.load(name_modified)
    
    if taxa == 'focal': 
    
        data_baseline = dataset_baseline['stationary_dist'][0, :]
    
        data_modified = dataset_modified['stationary_dist'][0, :]
        
    elif taxa == 'all': 
    
        data_baseline = dataset_baseline['stationary_dist'].sum(0)
    
        data_modified = dataset_modified['stationary_dist'].sum(0)

    density_baseline = gaussian_kde(data_baseline)
    
    density_modified = gaussian_kde(data_modified)
    
    x_range = np.logspace(np.log10(7. / (N * 10.)), 0, 10000)
    
    y_baseline = density_baseline(x_range)
    
    y_modified = density_modified(x_range)
    
    ax.plot(x_range, y_baseline, color = colors[0])
    
    ax.plot(x_range, y_modified, color = colors[1])
    
    if taxa == 'focal': ax.axvline(p, color = 'k', linestyle = '--', label = r'freq. in col. pool ($p_1$)')
        
    ax.fill_between(x_range, y_baseline, y_modified, color = 'gray', alpha = 0.3)
    
    if taxa == 'focal':
    
        baseline_label = r'average (no inh., $\alpha_0 = %.1f$)'%be
        
        modified_label = r'average (inh., $\alpha_0 = %.1f$)'%be
        
        ax.set_xlabel(r'freq. in hosts ($x_1$)', fontsize = 16)
        
        ax.set_ylabel('host density', fontsize = 16)
        
    elif taxa == 'all':
    
        baseline_label =  r'average (no inh., $\alpha_0 = %.1f$)'%be 
        
        modified_label =  r'average (inh., $\alpha_0 = %.1f$)'%be 
        
        ax.set_xlabel(r'microbial load ($x_i + o_i$)', fontsize = 16)
        
        ax.set_ylabel('host density', fontsize = 16)

    ax_twin = ax.twinx()
    
    ax_twin.tick_params(axis = 'both', which = 'both', direction = 'in')
    
    ax.axvline(data_modified.sum() / n_hosts, linestyle = (7, (5, 9)) ,color = colors[1], label = modified_label)
    
    ax.axvline(data_baseline.sum() / n_hosts, linestyle = (0, (5, 9)) , color = colors[0], label = baseline_label)
            
    if observable == 'prob_col': 
        
        ax_twin.plot(x_range, (y_modified - y_baseline) / n_hosts, color = 'gray', alpha = 0.6)
        
        if taxa == 'focal': ax_twin.set_ylabel('density difference', fontsize = 16, color = 'gray')
        
        elif taxa == 'all': ax_twin.set_ylabel('density difference', fontsize = 16, color = 'gray')

        ax_twin.tick_params(axis = 'y', which = 'both', direction = 'in', colors='gray')
        
        ax_twin.spines['right'].set_color('gray')

    if observable == 'exp_value': 
        
        ax_twin.plot(x_range, x_range * (y_modified - y_baseline) / n_hosts, color = 'gray', alpha = 0.6)
        
        if taxa == 'focal': ax_twin.set_ylabel(r'dens. difference $\cdot \,x_1$', fontsize = 16)
        
        elif taxa == 'all': ax_twin.set_ylabel(r'dens. diff. $\cdot$ mic. load', fontsize = 16)

    if observable == 'dif_dist': 
        
        ax_twin.plot(x_range, abs(y_modified - y_baseline)/(2. * n_hosts), color = 'gray', alpha = 0.6)
        
        if taxa == 'focal': ax_twin.set_ylabel(r'$\|\Delta \Phi[x_1]\|/2$', fontsize = 16)
        
        elif taxa == 'all': ax_twin.set_ylabel(r'$\|\Delta \Phi[\sum x_i]\|/2$', fontsize = 16)

    legend = ax.legend(fontsize = 10, loc = legend_loc, framealpha = 0.6)
                
    ax.set_xscale('log')

    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useMathText=True)

    ax_twin.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0), useMathText=True)
    
    ax.xaxis.set_ticks_position('both')

    ax.text(-0.2, 1.06, label_text, transform = ax.transAxes, fontsize = 25, fontweight = 'bold')
        
    ax.set_title(r'equilibrium', fontsize = 16)
        
    return ax
    
    
def fig_Diff(ax, var, X, AB_i, Be, N_spe, m, t, p, rep, taxa, observable, label_text, legend_loc, labelling_side, subdir):
    
    ax.tick_params(axis = 'both', which = 'both', direction = 'in')
    
    colors = ['purple', 'darkcyan']
    
    c = 0
    
    n_hosts = 10000
        
    for be in Be:
        
        Y_set = []
        
        for rep_baseline in [0, 1]:
        
            for rep_modified in [0, 1, 2]:
        
                Y = []

                for x in X:
            
                    print('bn: %.3f, be: %.3f, %s: %1.0e'%(AB_i[2], be, var, x))
        
                    if var == 'p': 
                
                        N = N_spe
                
                        name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, x, be, 0, 0, 0, 0, rep_baseline)
                
                        name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, x, be, AB_i[0], AB_i[1], AB_i[2], AB_i[3], rep_modified)

                    elif var == 'm':
                
                        N = N_spe
                
                        name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, x, t, p, be, 0, 0, 0, 0, rep_baseline)
                
                        name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, x, t, p, be, AB_i[0], AB_i[1], AB_i[2], AB_i[3], rep_modified)
    
                    elif var == 't':
                
                        N = N_spe
                
                        name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, x, p, be, 0, 0, 0, 0, rep_baseline)
                
                        name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, x, p, be, AB_i[0], AB_i[1], AB_i[2], AB_i[3], rep_modified)
    
                    elif var == 'N':

                        N = x
                
                        name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, x, m, t, p, be, 0, 0, 0, 0, rep_baseline)
                
                        name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, x, m, t, p, be, AB_i[0], AB_i[1], AB_i[2], AB_i[3], rep_modified)
            
                    dataset_baseline = np.load(name_baseline)
            
                    dataset_modified = np.load(name_modified)
            
                    if  taxa == 'focal':
                
                        data_baseline = dataset_baseline['stationary_dist'][0, :]
    
                        data_modified = dataset_modified['stationary_dist'][0, :]
                
                    elif taxa == 'all':
                
                        data_baseline = dataset_baseline['stationary_dist'].sum(0)
    
                        data_modified = dataset_modified['stationary_dist'].sum(0)
            
                    Y.append(diff_prob(data_modified, data_baseline, N, observable))
            
                Y_set.append(Y)
            
        Y_mean, Y_sd = np.zeros(len(X)), np.zeros(len(X))
        
        for Y in Y_set: Y_mean += Y
        
        Y_mean = Y_mean / len(Y_set)
        
        for Y in Y_set: Y_sd += np.power(Y - Y_mean, 2.)
        
        Y_sd = np.power(Y_sd / len(Y_set), 0.5)
        
        print(Y_mean, Y_sd)
            
        ax.errorbar(X, Y_mean, Y_sd, marker = '.', linestyle='None', label = r'$\alpha_0 = %.1f$'%be, color = colors[c])
                
        if var == 'p': ax.plot(p, Y_mean[np.where(np.array(X) == p)[0][0]], marker = '^', markersize = 7, color = colors[c])
            
        elif var == 'm': ax.plot(m, Y_mean[np.where(np.array(X) == m)[0][0]], marker = '^', markersize = 7, color = colors[c])
            
        elif var == 't': ax.plot(t, Y_mean[np.where(np.array(X) == t)[0][0]], marker = '^', markersize = 7, color = colors[c])
            
        elif var == 'N': ax.plot(N_spe, Y_mean[np.where(np.array(X) == N_spe)[0][0]], marker = '^', markersize = 7, color = colors[c])
        
        c += 1
            
    ax.axhline(0, color = 'k', linestyle = '--')
            
    if var == 'p': ax.set_xlabel(r'freq. in colonizers pool ($p_1$)', fontsize = 16)
        
    elif var == 'm': ax.set_xlabel(r'prob. of migration ($m$)', fontsize = 16)
        
    elif var == 't': ax.set_xlabel(r'prob. of host death ($\tau$)', fontsize = 16)
        
    elif var == 'N': ax.set_xlabel(r'carrying capacity ($N$)', fontsize = 16)

    if observable == 'prob_col': 
        
        if taxa == 'focal': ax.set_ylabel(r'difference ($\Delta P[x_1 > 0]$)', fontsize = 16)
        
        elif taxa == 'all': ax.set_ylabel('difference\n'+r'($\Delta P[x_i + o_i > 0]$)', fontsize = 16)
    
    elif observable == 'exp_value': 
        
        if taxa == 'focal': ax.set_ylabel(r'difference ($\Delta E[x_1]$)', fontsize = 16)
        
        elif taxa == 'all': ax.set_ylabel(r'difference ($\Delta E[x_i + o_i]$)', fontsize = 16)

    elif observable == 'dif_dist': 
        
        if taxa == 'focal': ax.set_ylabel(r'$\sum\| \Delta \Phi[x_1] \| / 2$', fontsize = 16)
        
        elif taxa == 'all': ax.set_ylabel(r'$\sum\| \Delta \Phi[\sum x_i] \| / 2$', fontsize = 16)
        
        ax.axhline(1., color = 'k', linestyle = '--')
        
        ax.set_ylim(-0.05, 1.05)
    
    ax.text(-0.2, 1.06, label_text, transform = ax.transAxes, fontsize = 25, fontweight = 'bold')
    
    ax.set_title(r'time = $10^7$', fontsize = 16)

    ax.set_xscale('log')
    
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,1), useMathText=True)
    
    if labelling_side == 'right':
    
        ax.yaxis.tick_right()
    
        ax.yaxis.set_label_position("right")
    
    ax.xaxis.set_ticks_position('both')
    
    ax.yaxis.set_ticks_position('both')

    ax.legend(fontsize = 10, loc = legend_loc, framealpha = 0.6, title = r'faster col. $\alpha_0 \to 0$')
        
    return ax


def fig_cumsum(ax_cumsum_modified, ax_probcol, var, X, AB_i, be, N_spe, m, t, p, rep, taxa, label_text, ylabel, legend, legend_loc, subdir): 
    
    n_hosts = 10000
    
    Y_probcol_baseline = []
    
    Y_probcol_modified = []
    
    Y_mean_baseline = []
    
    Y_mean_modified = []
    
    truncate_cmap = 1.2
            
    for x in X:
        
        print('bn: %.3f, be: %.3f, %s: %1.0e'%(AB_i[2],be,var,x))
    
        if var == 'p': 
            
            N = N_spe
            
            name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, x, be, 0, 0, 0, 0, rep)
            
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, x, be, AB_i[0], AB_i[1], AB_i[2], AB_i[3], rep)

        elif var == 'm':
            
            N = N_spe
            
            name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, x, t, p, be, 0, 0, 0, 0, rep)
            
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, x, t, p, be, AB_i[0], AB_i[1], AB_i[2], AB_i[3], rep)

        elif var == 't':
            
            N = N_spe
            
            name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, x, p, be, 0, 0, 0, 0, rep)
            
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, x, p, be, AB_i[0], AB_i[1], AB_i[2], AB_i[3], rep)

        elif var == 'N':

            N = x
            
            name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, x, m, t, p, be, 0, 0, 0, 0, rep)
            
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, x, m, t, p, be, AB_i[0], AB_i[1], AB_i[2], AB_i[3], rep)
        
        dataset_baseline = np.load(name_baseline)
        
        dataset_modified = np.load(name_modified)
        
        if taxa == 'focal':
        
            data_baseline = np.sort(dataset_baseline['stationary_dist'][0, :])

            data_modified = np.sort(dataset_modified['stationary_dist'][0, :])
            
        elif taxa == 'all':
            
            data_baseline = np.sort(dataset_baseline['stationary_dist'].sum(0))

            data_modified = np.sort(dataset_modified['stationary_dist'].sum(0))
        
        Y_cumsum_baseline = []
        
        Y_cumsum_modified = []
        
        for percentile in np.arange(0.1, 1.1, 0.1): 
            
            Y_cumsum_baseline.append(data_baseline[int(percentile * n_hosts)-1])
            
            Y_cumsum_modified.append(data_modified[int(percentile * n_hosts)-1])
                                
        ax_cumsum_modified.scatter([x]*10, Y_cumsum_modified, c=np.arange(0.1, 1.1, 0.1), vmin=0, vmax=truncate_cmap, s = 15, cmap = cm.plasma)

        ax_cumsum_modified.scatter([x]*10, Y_cumsum_baseline, c=np.arange(0.1, 1.1, 0.1), vmin=0, vmax=truncate_cmap, s = 20, cmap = cm.plasma, marker = 'x')
    
        Y_probcol_baseline.append((data_baseline >= 1./N).sum() / n_hosts)

        Y_probcol_modified.append((data_modified >= 1./N).sum() / n_hosts)
        
        Y_mean_baseline.append(data_baseline.sum() / n_hosts)

        Y_mean_modified.append(data_modified.sum() / n_hosts)
    
    if taxa == 'focal':
        
        baseline_label = r'no inh.'
        
        modified_label = r'inh.'
        
        if var != 'p': ax_cumsum_modified.axhline(p, linestyle = ':', color = 'k', linewidth = 1)
        
        else: ax_cumsum_modified.plot(X, X, linestyle = ':', color = 'k', linewidth = 1)
        
    elif taxa == 'all':
        
        baseline_label = r'no inh.'
        
        modified_label = r'inh.'
        
        ax_cumsum_modified.axhline(1, linestyle = ':', color = 'k', linewidth = 1)
    
    ax_probcol.axhline(1, linestyle = ':', color = 'k', linewidth = 1)
    
    ax_probcol.plot(X, Y_probcol_baseline, marker = 'x', color = 'k')
    
    ax_probcol.plot(X, Y_probcol_modified, marker = '.', color = 'k')
        
    ax_cumsum_modified.plot(X, Y_mean_modified, marker = '.', color = 'k', linewidth = 1, markersize = 8)
    
    ax_cumsum_modified.plot(X, Y_mean_baseline, marker = 'x', color = 'k', linewidth = 1, markersize = 5)
        
    if var == 'p': ax_cumsum_modified.set_xlabel(r'freq. in colonizers pool ($p_1$)', fontsize = 16)
                
    elif var == 'm': ax_cumsum_modified.set_xlabel(r'prob. of migration ($m$)', fontsize = 16)
                
    elif var == 't': ax_cumsum_modified.set_xlabel(r'prob. of host death ($\tau$)', fontsize = 16)
                
    elif var == 'N': ax_cumsum_modified.set_xlabel(r'carrying capacity ($N$)', fontsize = 16)
        
    if ylabel:
        
        legend_elements = [Line2D([0],[0], color='k', marker='x', markerfacecolor='k', label=baseline_label, lw=0), Line2D([0],[0], marker='.', color='k', markerfacecolor='k', label=modified_label, lw=0)]
                
        if taxa == 'focal':
        
            ax_cumsum_modified.set_ylabel(r'freq. in hosts ($x_1$)', fontsize = 16)
    
            ax_probcol.set_ylabel('occurrence\n'+r'($P[x_1 > 0]$)', fontsize = 16)
        
        elif taxa == 'all':

            ax_cumsum_modified.set_ylabel(r'microbial load ($x_i + o_i$)', fontsize = 15)
    
            ax_probcol.set_ylabel('occurrence\n'+r'($P[x_i + o_i > 0]$)', fontsize = 15)
            
    else:
                
        mp.setp(ax_cumsum_modified.get_yticklabels(), visible=False)
        
        mp.setp(ax_probcol.get_yticklabels(), visible=False)
    
    ax_cumsum_modified.set_xscale('log')
    
    ax_probcol.set_xscale('log')
        
    ax_cumsum_modified.set_yscale('log')
    
    ax_probcol.set_yscale('log')
        
    ax_cumsum_modified.set_ylim(8E-6, 1.4)

    if taxa == 'focal': ax_probcol.set_ylim(1E-2, 1.4)

    elif taxa == 'all': ax_probcol.set_ylim(1E-1, 1.4)

    mp.setp(ax_probcol.get_xticklabels(), visible=False)

    if legend:

        plasma = cm.plasma
        
        legend_elements = [Line2D([0],[0], marker='_', color=plasma(col/truncate_cmap), label='%i'%(col*100)) for col in np.arange(0.1,1.1,.1)]

        legend_elements.append(Line2D([0],[0], marker='_', color='k', label='average'))
        
        if taxa == 'focal': legend_elements.append(Line2D([0],[0], linestyle = ':', color='k', label=r'$p_1$'))
        
        ax_cumsum_modified.legend(handles=legend_elements, loc=legend_loc, title = r'percentile', ncol = 2, fontsize = 8, framealpha = 0.6)
        
    ax_cumsum_modified.tick_params(axis = 'both', which = 'both', direction = 'in', bottom = True, top = True, left = True, right = True)
    
    ax_probcol.tick_params(axis = 'both', which = 'both', direction = 'in', bottom = True, top = True, left = True, right = True)
    
    ax_probcol.text(0., 1.06, label_text, transform = ax_probcol.transAxes, fontsize = 25, fontweight = 'bold')
        
    return ax_cumsum_modified, ax_probcol


def fig_pcolor(axes, var, X, AB_i, N, m, t, p, be, rep, observable, title_text, label_text, subdir):
    
    n_hosts = 1E4
    
    data_axes = np.zeros((4, len(X)))
    
    Y_baseline = []
    
    Y_modified_1 = []
    
    Y_modified_2 = []
            
    for x in X:
    
        if var == 'p':
        
            name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, x, be, 0, 0, 0, 0, rep)
            
            name_modified_1 = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, x, be, AB_i[0][0], AB_i[0][1], AB_i[0][2], AB_i[0][3], rep)
            
            name_modified_2 = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, t, x, be, AB_i[1][0], AB_i[1][1], AB_i[1][2], AB_i[1][3], rep)
            
        elif var == 'm':

            name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, x, t, p, be, 0, 0, 0, 0, rep)
            
            name_modified_1 = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, x, t, p, be, AB_i[0][0], AB_i[0][1], AB_i[0][2], AB_i[0][3], rep)
            
            name_modified_2 = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, x, t, p, be, AB_i[1][0], AB_i[1][1], AB_i[1][2], AB_i[1][3], rep)
                                    
        elif var == 't':

            name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, x, p, be, 0, 0, 0, 0, rep)
            
            name_modified_1 = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, x, p, be, AB_i[0][0], AB_i[0][1], AB_i[0][2], AB_i[0][3], rep)

            name_modified_2 = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, N, m, x, p, be, AB_i[1][0], AB_i[1][1], AB_i[1][2], AB_i[1][3], rep)
            
        elif var == 'N':
            
            N = x

            name_baseline = '%s/inh_0_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, x, m, t, p, be, 0, 0, 0, 0, rep)
            
            name_modified_1 = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, x, m, t, p, be, AB_i[0][0], AB_i[0][1], AB_i[0][2], AB_i[0][3], rep)

            name_modified_2 = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%(subdir, x, m, t, p, be, AB_i[1][0], AB_i[1][1], AB_i[1][2], AB_i[1][3], rep)
            
        dataset_baseline = np.load(name_baseline)
        
        dataset_modified_1 = np.load(name_modified_1)
    
        dataset_modified_2 = np.load(name_modified_2)
            
        if observable == 'prob_col':
    
            Y_baseline.append(np.log10((dataset_baseline['stationary_dist'][0, :] >= 1./N).sum() / n_hosts))
    
            Y_modified_1.append(np.log10((dataset_modified_1['stationary_dist'][0, :] >= 1./N).sum() / n_hosts))
            
            Y_modified_2.append(np.log10((dataset_modified_2['stationary_dist'][0, :] >= 1./N).sum() / n_hosts))
            
        elif observable == 'exp_value':
            
            Y_baseline.append(np.log10(dataset_baseline['stationary_dist'][0, :].sum() / n_hosts))
    
            Y_modified_1.append(np.log10(dataset_modified_1['stationary_dist'][0, :].sum() / n_hosts))

            Y_modified_2.append(np.log10(dataset_modified_2['stationary_dist'][0, :].sum() / n_hosts))    
        
    data_axes[0, :] = Y_baseline
    
    data_axes[1, :] = Y_modified_1
        
    data_axes[2, :] = Y_modified_2
                
    X, Y = np.meshgrid(X, range(4))

    axes.pcolor(X, Y, data_axes, cmap = mp.cm.inferno, edgecolors = 'white', vmin = -7, vmax = 0)
    
    axes.set_yticks(np.arange(0.5,3.,1))
    
    axes.set_yticklabels(['none', 'sym.', 'asym.'])
        
    if var == 'p' and label_text == None: axes.set_xlabel(r'freq. in colonizers pool ($p_1$)', fontsize = 16)
        
    elif var == 'm' and label_text == None: axes.set_xlabel(r'prob. of migration ($m$)', fontsize = 16)
        
    elif var == 't' and label_text == None: axes.set_xlabel(r'prob. of host death ($\tau$)', fontsize = 16)
        
    elif var == 'N' and label_text == None: axes.set_xlabel(r'carrying capacity ($N$)', fontsize = 16)
    
    axes.set_title(title_text, fontsize = 16)
    
    axes.set_xscale('log')
    
    if label_text != None: 
        
        axes.text(0., 1.06, label_text, transform = axes.transAxes, fontsize = 25, fontweight = 'bold')
    
        mp.setp(axes.get_xticklabels(), visible=False)
    
    return axes


# Figure 2
    
def fig2(spe_args):

    rep_spe = 0
    
    [N_spe, m_spe, t_spe, p_spe, AB_i, Be] = spe_args

    p_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    m_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    t_set = [i * j for j in np.logspace(-7, -2, 6) for i in [1., 2., 4., 7.]][:-3]

    N_set = [i * j for j in np.logspace(3, 7, 5) for i in [1., 2., 4., 7.]][:-3]
    
    fig = mp.figure(figsize = (12.5, 7.8))
    
    gs = gridspec.GridSpec(45, 70, wspace=8, hspace=8)
    
    ax00 = mp.subplot(gs[:21, :21])
    
    ax01 = mp.subplot(gs[4:20, 28:44])
    
    ax02 = mp.subplot(gs[1:19, 52:])   
    
    ax10 = mp.subplot(gs[27:, 1:19])
    
    ax11 = mp.subplot(gs[27:, 27:45])
    
    ax12 = mp.subplot(gs[27:, 52:])

    # Fig 2A

    ax00 = fig_timeseries(ax00, N_spe, m_spe, t_spe, p_spe, Be, AB_i, rep_spe, 'all', 'prob_col', 'A', 'lower right', 'output_sde/23E', 100, [1, None], [-0.02, 1.02], 'simulated populations')

    ax01 = fig_timeseries(ax01, N_spe, m_spe, t_spe, p_spe, Be, AB_i, rep_spe, 'all', 'prob_col', '', 'lower right', 'output_sde/23E', 100, [1E3, None], [0.965, 1.], '')

    # Fig 2B

    ax02 = fig_distribution(ax02, N_spe, m_spe, t_spe, p_spe, Be[0], AB_i, rep_spe, 'all', 'prob_col', 'B', 'lower right', 'output_sde/23E')
     
    # Fig 2C

    ax10 = fig_Diff(ax10, 'm', m_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'all', 'prob_col', 'C', 'upper right', 'left', 'output_sde/23D')

    # Fig 2D

    ax11 = fig_Diff(ax11, 't', t_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'all', 'prob_col', 'D', 'upper left', 'left', 'output_sde/23E')

    # Fig 2E

    ax12 = fig_Diff(ax12, 'N', N_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'all', 'prob_col', 'E', 'lower center', 'left', 'output_sde/23F')

    mp.savefig('fig2.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')
    
    
# Figure 3
    
def fig3(spe_args):
    
    rep_spe = 0

    [N_spe, m_spe, t_spe, p_spe, AB_i, Be] = spe_args

    p_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    m_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    t_set = [i * j for j in np.logspace(-7, -2, 6) for i in [1., 2., 4., 7.]][:-3]

    N_set = [i * j for j in np.logspace(3, 7, 5) for i in [1., 2., 4., 7.]][:-3]

    fig, axes = mp.subplots(nrows = 2, ncols = 2, figsize = (6.4, 6.4))

    # Fig 3A

    axes[0,0] = fig_timeseries(axes[0,0], N_spe, m_spe, t_spe, p_spe, Be, AB_i, rep_spe, 'all', 'exp_value', 'A', 'upper left', 'output_sde/23E', 100, [1, None], [5E-4, 1E-2], 'simulated populations')
  
    # Fig 3B

    axes[0,1] = fig_Diff(axes[0,1], 'm', m_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'all', 'exp_value', 'B', 'upper left', 'left', 'output_sde/23D')

    # Fig 3C

    axes[1,0] = fig_Diff(axes[1,0], 't', t_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'all', 'exp_value', 'C', 'upper right', 'left', 'output_sde/23E')

    # Fig 3D

    axes[1,1] = fig_Diff(axes[1,1], 'N', N_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'all', 'exp_value', 'D', 'center right', 'left', 'output_sde/23F')

    fig.subplots_adjust(bottom=0., top=1., left=0., right=1., wspace=0.3, hspace=0.5)

    mp.savefig('fig3.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')
    
    
# Figure 4
    
def fig4(spe_args):
    
    rep_spe = 0
    
    [N_spe, m_spe, t_spe, p_spe, AB_i, Be] = spe_args
    
    p_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    m_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    t_set = [i * j for j in np.logspace(-7, -2, 6) for i in [1., 2., 4., 7.]][:-3]

    N_set = [i * j for j in np.logspace(3, 7, 5) for i in [1., 2., 4., 7.]][:-3]
        
    fig = mp.figure(figsize = (9.9, 5.0))

    gs = gridspec.GridSpec(2, 2, figure = fig, hspace = 0.5, wspace = 0.15)

    gs00 = gridspec.GridSpecFromSubplotSpec(9, 8, subplot_spec=gs[0,0], hspace = 0.9)
    
    gs01 = gridspec.GridSpecFromSubplotSpec(9, 8, subplot_spec=gs[0,1], hspace = 0.9)

    gs10 = gridspec.GridSpecFromSubplotSpec(9, 8, subplot_spec=gs[1,0], hspace = 0.9)
    
    gs11 = gridspec.GridSpecFromSubplotSpec(9, 8, subplot_spec=gs[1,1], hspace = 0.9)

    axes_00_0 = fig.add_subplot(gs00[:4, :])

    axes_01_0 = fig.add_subplot(gs01[:4, :])

    axes_10_0 = fig.add_subplot(gs10[:4, :])

    axes_11_0 = fig.add_subplot(gs11[:4, :])
    
    axes_00_1 = fig.add_subplot(gs00[5:, :])
    
    axes_01_1 = fig.add_subplot(gs01[5:, :])
        
    axes_10_1 = fig.add_subplot(gs10[5:, :])
            
    axes_11_1 = fig.add_subplot(gs11[5:, :])
    
    # Fig 4A
    
    axes_00_0 = fig_pcolor(axes_00_0, 'p', p_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[0], rep_spe, 'exp_value', r'slower col. ($\alpha_0 = %0.1f$)'%Be[0], 'A', 'output_sde/23C')

    axes_00_1 = fig_pcolor(axes_00_1, 'p', p_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[1], rep_spe, 'exp_value', r'faster col. ($\alpha_0 = %0.1f$)'%Be[1], None, 'output_sde/23C')

    axes_00_1.text(-0.2, 0.0, 'inheritance', fontsize = 16, rotation = 'vertical', transform = axes_00_1.transAxes)
    
    # Fig 4B
    
    axes_01_0 = fig_pcolor(axes_01_0, 'm', m_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[0], rep_spe, 'exp_value', r'slower col. ($\alpha_0 = %0.1f$)'%Be[0], 'B', 'output_sde/23D')

    axes_01_1 = fig_pcolor(axes_01_1, 'm', m_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[1], rep_spe, 'exp_value', r'faster col. ($\alpha_0 = %0.1f$)'%Be[1], None, 'output_sde/23D')
    
    # Fig 4C
    
    axes_10_0 = fig_pcolor(axes_10_0, 't', t_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[0], rep_spe, 'exp_value', r'slower col. ($\alpha_0 = %0.1f$)'%Be[0], 'C', 'output_sde/23E')

    axes_10_1 = fig_pcolor(axes_10_1, 't', t_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[1], rep_spe, 'exp_value', r'faster col. ($\alpha_0 = %0.1f$)'%Be[1], None, 'output_sde/23E')
    
    # Fig 4D
    
    axes_11_0 = fig_pcolor(axes_11_0, 'N', N_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[0], rep_spe, 'exp_value', r'slower col. ($\alpha_0 = %0.1f$)'%Be[0], 'D', 'output_sde/23F')
        
    axes_11_1 = fig_pcolor(axes_11_1, 'N', N_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[1], rep_spe, 'exp_value', r'faster col. ($\alpha_0 = %0.1f$)'%Be[1], None, 'output_sde/23F')
    
    axes_00_vl = [i*j for i in [3,5,6,8,9] for j in np.logspace(-5,-1,5)]
    
    axes_01_vl = [i*j for i in [3,5,6,8,9] for j in np.logspace(-5,-1,5)]
        
    axes_10_vl = [i*j for i in [3,5,6,8,9] for j in np.logspace(-7,-3,5)]
            
    axes_11_vl = [i*j for i in [3,5,6,8,9] for j in np.logspace(3,6,4)]
    
    for vl in axes_00_vl: axes_00_0.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
    
    for vl in axes_00_vl: axes_00_1.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
    
    for vl in axes_01_vl: axes_01_0.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
    
    for vl in axes_01_vl: axes_01_1.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
        
    for vl in axes_10_vl: axes_10_0.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
    
    for vl in axes_10_vl: axes_10_1.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
            
    for vl in axes_11_vl: axes_11_0.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
    
    for vl in axes_11_vl: axes_11_1.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
    
    fig.subplots_adjust(bottom=0., top=1., left=0., right=0.93)
    
    cb_ax = fig.add_axes([0.95, 0.2, 0.05, 0.6])

    cb_ax.axis('off')

    img = cb_ax.imshow(np.array([[1E-7,1E0]]), cmap = mp.cm.inferno, norm=colors.LogNorm(vmin=1E-7, vmax=1E0))

    img.set_visible(False)

    cb_ax.set_aspect('auto')

    cbar = fig.colorbar(img, orientation = "vertical", ax = cb_ax, fraction = 1.0)

    cbar.ax.tick_params(labelsize = 12)

    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation = 90)

    cbar.set_label(r'average frequency ($\overline{x_1}$)',fontsize = 16)
    
    mp.savefig('fig4.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')
    

# Figure 5
    
def fig5(spe_args):
    
    rep_spe = 0
    
    [N_spe, Be] = spe_args
        
    fig = mp.figure(figsize = (7.2, 6.6))
    
    gs = gridspec.GridSpec(70, 70, wspace=8, hspace=8)
    
    ax0_0 = mp.subplot(gs[:30, :30])
    
    ax0_1 = mp.subplot(gs[1:30, 41:])
    
    ax1_0 = mp.subplot(gs[41:, 1:30])   
    
    ax1_1 = mp.subplot(gs[40:, 40:])
    
    # Fig 5A
    
    x = np.arange(10) / 9
    
    y = 1. - x
        
    ax0_0.plot(x[:4], x[:4]+y[6], linestyle = '-', color = 'purple', linewidth = 2.5)
        
    ax0_0.plot(x[3:7], y[3:7], linestyle = '-', color = 'darkcyan', linewidth = 2.5)
    
    ax0_0.plot(x[6:], y[6:], linestyle = '--', color = 'darkcyan', linewidth = 2.5)
    
    ax0_0.axvline(x[6], y[6]-1.8/9, y[6], linestyle = '--', color = 'darkcyan', linewidth = 2.5)
    
    ax0_0.axvline(x[6], y[4]-1.8/9, y[5], linestyle = '-', color = 'gray')
    
    ax0_0.plot(x[6:8], y[6:8]-2./9, linestyle = '--', color = 'darkcyan', linewidth = 2.5)

    ax0_0.axvline(x[-3], y[-1]+0.4/9, y[-3]+0.9/9, linestyle = '-', color = 'gray')
    
    ax0_0.axvline(x[-1], y[-1]+0.4/9, y[-1]+2.8/9, linestyle = '-', color = 'gray')
    
    ax0_0.fill_between(x[:4], 0 , 1., color = 'gray', alpha = 0.2)
    
    ax0_0.text(0.31, 0.70, r'$x^*_1$', fontsize=10)
    
    ax0_0.text(0.15, 0.85, 'longer time before extinction due\nto the increase in frequency', fontsize=10)
        
    ax0_0.text(0.65, 0.5, 'inheritance to\noffspring', fontsize=10)
    
    ax0_0.text(0.75, 0.35, 'extinction', fontsize=10)
        
    ax0_0.set_ylabel(r'log freq. in a host (log $x_1$)', fontsize=14)

    ax0_0.set_xlabel(r'time', fontsize=14)
    
    ax0_0.set_title(r'lineage taxon', fontsize=14)
    
    ax0_0.get_xaxis().set_ticks([])
        
    ax0_0.set_yticks(np.log10(range(1,10)))
        
    ax0_0.set_yticklabels([])
        
    ax0_0.spines['right'].set_color('none')
    
    ax0_0.spines['top'].set_color('none')
    
    ax0_0.text(-0.01, -0.05, 0, transform = ax0_0.transAxes, fontsize = 12)
    
    ax0_0.set_xlim(0.0, 1.)

    ax0_0.set_ylim(0.0, 1.)

    ax0_0.set_aspect(1)
        
    # Fig 5B
    
    x_0 = np.linspace(0, 1, 100)
    
    a_0 = np.linspace(0.1, 0.9, 3)
    
    for i in range(len(a_0)):
        
         m = x_0 * (1. - a_0[i])
         
         ax1_0.plot(m, 1. - x_0, label=r'$\alpha_0=$'+str(a_0[i]))
         
         ax1_0.fill_between(m, 0 , 1. - x_0, color = 'gray', alpha = 0.2)

    ci = ax1_0.text(0.1, 0.2, r'$x_1 + o_1 < 1 - \frac{m}{1-\alpha_0}$', fontsize=12)

    ci.set_bbox(dict(facecolor='white', alpha=0.4, edgecolor='none'))

    ax1_0.set_ylabel(r'microbial load ($x_1 + o_1$)', fontsize=14, color = 'purple')

    ax1_0.set_xlabel(r'prob. of migration ($m$)', fontsize=14, color = 'purple')
    
    ax1_0.set_title(r'condition for increase', fontsize=14, color = 'purple')

    ax1_0.set_xlim(0, 1)

    ax1_0.set_ylim(0, 1)

    ax1_0.set_aspect(1)

    ax1_0.legend(title=r'faster col. $\alpha_0 \to 0$')

    # Fig 5C
    
    m_set = np.arange(1,10) * 1E-1
    
    frac_set = [1E-2, 1E-1, 5E-1, 1E0]
    
    for frac in frac_set:
        
        t_ext_mean = np.zeros(len(m_set))
        
        for m in m_set:
            
            t_ext_data = np.load('output_sde/5C/N_%.0e_be_%.0e_m_%.0e_frac_%.0e_%i.npz'%(1E5, 1e-01, m, frac, 0))

            t_ext_mean[m_set == m] = t_ext_data['ext_times'].sum() / 1E4

        ax0_1.plot(m_set, t_ext_mean, '.-', label = '%.2f'%frac)

    ax0_1.set_ylabel('average time to extinction\n'+r'after point $x^*_1$', fontsize=13, color = 'darkcyan')

    ax0_1.set_xlabel(r'prob. of migration ($m$)', fontsize=14, color = 'darkcyan')
    
    ax0_1.set_title(r'faster col. ($\alpha_0 = 0.1$)', fontsize=14, color = 'darkcyan')
    
    ax0_1.set_xlim(-0.01, 1)
    
    ax0_1.set_ylim(-0.01E7, 1E7)
    
    ax0_1.set_aspect(1E-7)
    
    ax0_1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,1), useMathText=True)
    
    ax0_1.legend(title=r'fraction of lineage taxon', ncol=2)

    # Fig 5D
    
    load_set = np.array([1E-3, 2E-3, 4E-3, 7E-3, 1E-2, 2E-2, 4E-2, 7E-2, 1E-1, 2E-1, 4E-1, 7E-1, 1E0])
    
    frac_set = [1E-2, 1E-1, 5E-1, 1E0]
    
    for frac in frac_set:
        
        tau_set = np.zeros(len(load_set))
        
        for load in load_set:        
            
            t_ext_data = np.load('output_sde/5D/N_%.0e_be_%.0e_m_%.0e_frac_%.0e_load_%.0e_%i.npz'%(1E5, 1e-01, 3E-1, frac, load, 0))

            ext_times = t_ext_data['ext_times']
            
            ext_times.sort()

            tau_set[load_set == load] = - 1. / ext_times[4999] * np.log(1 - 0.5)
            
        ax1_1.loglog(load_set, tau_set, '.-', label = '%.2f'%frac)
        
    ax1_1.axhline(1., linestyle = '--', color = 'k')
    
    ax1_1.fill_between([8E-4, 1. - 3E-1 / (1. - 1E-1)], 8E-8, 1.2E-5, color = 'gray', alpha = 0.2, linewidth=0)
    
    ax1_1.text(1E-2, 1.2E-6, 'most hosts maintain\nthe lineage taxon', fontsize=10)
    
    ax1_1.text(1E-2, 1E-7, 'most hosts lose\nthe lineage taxon', fontsize=10)
                
    ax1_1.set_ylabel(r'prob. of host death ($\tau$)', fontsize=14)

    ax1_1.set_xlabel(r'initial microbial load ($x_1 + o_1$)', fontsize=14)
    
    ax1_1.set_xlim(8E-4, 1.2E0)
    
    ax1_1.set_ylim(8E-8, 1.2E-5)
    
    ax1_1.set_aspect(3./2.)
    
    ax1_1.legend(title=r'initial fraction of lineage taxon', ncol=2, loc = 'upper right')
    
    ax1_1.set_title(r'$m = 0.3, \alpha_0 = 0.1$', fontsize = 14)
    
    fig.subplots_adjust(bottom=0., top=1., left=0., right=1., wspace=0.25, hspace=0.3)
    
    ax1_0.tick_params(axis = 'both', which = 'both', direction = 'in', colors='purple')
    
    ax0_1.tick_params(axis = 'both', which = 'both', direction = 'in', colors='darkcyan')
    
    ax1_1.tick_params(axis = 'both', which = 'both', direction = 'in')
    
    ax0_1.xaxis.set_ticks_position('both')
    
    ax0_1.yaxis.set_ticks_position('both')
    
    ax1_0.xaxis.set_ticks_position('both')
    
    ax1_0.yaxis.set_ticks_position('both')
    
    ax1_1.xaxis.set_ticks_position('both')
    
    ax1_1.yaxis.set_ticks_position('both')
    
    ax1_0.spines['bottom'].set_color('purple')
    
    ax1_0.spines['top'].set_color('purple')
        
    ax1_0.spines['right'].set_color('purple')
            
    ax1_0.spines['left'].set_color('purple')
    
    ax0_1.spines['bottom'].set_color('darkcyan')
    
    ax0_1.spines['top'].set_color('darkcyan')
        
    ax0_1.spines['right'].set_color('darkcyan')
            
    ax0_1.spines['left'].set_color('darkcyan')
        
    ax0_0.text(-0.25, 1.03, 'A', transform = ax0_0.transAxes, fontsize = 25, fontweight = 'bold')
    
    ax0_1.text(-0.25, 1.03, 'B', transform = ax0_1.transAxes, fontsize = 25, fontweight = 'bold')
    
    ax1_0.text(-0.25, 1.03, 'C', transform = ax1_0.transAxes, fontsize = 25, fontweight = 'bold')
    
    ax1_1.text(-0.25, 1.03, 'D', transform = ax1_1.transAxes, fontsize = 25, fontweight = 'bold')
            
    mp.savefig('fig5.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')


# Figure Sup 1
    
def figsup1(spe_args):
    
    rep_spe = 0

    [N_spe, m_spe, t_spe, p_spe, AB_i, Be] = spe_args

    p_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    m_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    t_set = [i * j for j in np.logspace(-7, -2, 6) for i in [1., 2., 4., 7.]][:-3]

    N_set = [i * j for j in np.logspace(3, 7, 5) for i in [1., 2., 4., 7.]][:-3]

    fig, axes = mp.subplots(nrows = 2, ncols = 3, figsize = (9.9, 6.4))

    # Fig Sup 1A

    axes[0,0] = fig_timeseries(axes[0,0], N_spe, m_spe, t_spe, p_spe, Be, AB_i, rep_spe, 'focal', 'prob_col', 'A', 'upper left', 'output_sde/23E', 100, [1, None], [3E-1, .72], 'simulated populations')

    # Fig Sup 1B

    axes[0,1] = fig_distribution(axes[0,1], N_spe, m_spe, t_spe, p_spe, Be[0], AB_i, rep_spe, 'focal', 'prob_col', 'B', 'upper right', 'output_sde/23E')

    # Fig Sup 1C

    axes[0,2] = fig_Diff(axes[0,2], 'p', p_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'prob_col', 'C', 'upper left', 'right', 'output_sde/23C')
  
    # Fig Sup 1D

    axes[1,0] = fig_Diff(axes[1,0], 'm', m_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'prob_col', 'D', 'upper left', 'left', 'output_sde/23D')

    # Fig Sup 1E

    axes[1,1] = fig_Diff(axes[1,1], 't', t_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'prob_col', 'E', 'upper left', 'left', 'output_sde/23E')

    # Fig Sup 1F

    axes[1,2] = fig_Diff(axes[1,2], 'N', N_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'prob_col', 'F', 'upper right', 'left', 'output_sde/23F')

    fig.subplots_adjust(bottom=0., top=1., left=0., right=1., wspace=0.3, hspace=0.5)

    mp.savefig('figsup1.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')


# Figure Sup 2

def figsup2(spe_args):
    
    rep_spe = 0
    
    [N_spe, m_spe, t_spe, p_spe, AB_i, be_spe] = spe_args
    
    p_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    m_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    t_set = [i * j for j in np.logspace(-7, -2, 6) for i in [1., 2., 4., 7.]][:-3]

    N_set = [i * j for j in np.logspace(3, 7, 5) for i in [1., 2., 4., 7.]][:-3]
    
    fig = mp.figure(figsize = (9, 4))
        
    gs = gridspec.GridSpec(4, 9, figure = fig, wspace = 0.09)
        
    axes_cumsum_modified = [fig.add_subplot(gs[1:,:3])]
    
    axes_probcol = [fig.add_subplot(gs[:1,:3], sharex = axes_cumsum_modified[0])]
        
    for i in range(3, 8, 3): 
                
        axes_cumsum_modified.append(fig.add_subplot(gs[1:,i:(i+3)], sharey = axes_cumsum_modified[0]))
        
        axes_probcol.append(fig.add_subplot(gs[:1,i:(i+3)], sharex = axes_cumsum_modified[int(i/3)], sharey = axes_probcol[0]))
                    
    # Fig Sup 2A

    axes_cumsum_modified[0], axes_probcol[0] = fig_cumsum(axes_cumsum_modified[0], axes_probcol[0], 'm', m_set, AB_i, be_spe, N_spe, m_spe, t_spe, p_spe, rep_spe, 'all', 'A', True, False, '', 'output_sde/23D')

    # Fig Sup 2B

    axes_cumsum_modified[1], axes_probcol[1] = fig_cumsum(axes_cumsum_modified[1], axes_probcol[1], 't', t_set, AB_i, be_spe, N_spe, m_spe, t_spe, p_spe, rep_spe, 'all', 'B', False, False, '', 'output_sde/23E')
  
    # Fig Sup 2C

    axes_cumsum_modified[2], axes_probcol[2] = fig_cumsum(axes_cumsum_modified[2], axes_probcol[2], 'N', N_set, AB_i, be_spe, N_spe, m_spe, t_spe, p_spe, rep_spe, 'all', 'C', False, False, '', 'output_sde/23F')
    
    legend_elements = [Line2D([0],[0], color='k', marker='x', markerfacecolor='k', label='no inh.', lw=0)]

    legend_elements.append(Line2D([0],[0], marker='.', color='k', markerfacecolor='k', label='inh.', lw=0))
    
    legend_elements = legend_elements + [Line2D([0],[0], marker='_', color=cm.plasma(col/1.2), label='percentile %i'%(col*100)) for col in np.arange(0.1,1.1,.1)]

    legend_elements.append(Line2D([0],[0], marker='_', color='k', label='average'))
    
    axes_cumsum_modified[2].legend(handles=legend_elements, bbox_to_anchor=(1.57, 0.7), loc='center right', ncol = 1)
    
    fig.subplots_adjust(bottom=0., top=1., left=0., right=1., wspace=0.3, hspace=0.05)
    
    mp.savefig('figsup2.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')

    
    
# Figure Sup 3

def figsup3(spe_args):
    
    rep_spe = 0
    
    [N_spe, m_spe, t_spe, p_spe, AB_i, be_spe] = spe_args
    
    p_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    m_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    t_set = [i * j for j in np.logspace(-7, -2, 6) for i in [1., 2., 4., 7.]][:-3]

    N_set = [i * j for j in np.logspace(3, 7, 5) for i in [1., 2., 4., 7.]][:-3]
    
    fig = mp.figure(figsize = (12, 4))
        
    gs = gridspec.GridSpec(4, 12, figure = fig, wspace = 0.09)
        
    axes_cumsum_modified = [fig.add_subplot(gs[1:,:3])]
    
    axes_probcol = [fig.add_subplot(gs[:1,:3], sharex = axes_cumsum_modified[0])]
        
    for i in range(3, 11, 3): 
                
        axes_cumsum_modified.append(fig.add_subplot(gs[1:,i:(i+3)], sharey = axes_cumsum_modified[0]))
        
        axes_probcol.append(fig.add_subplot(gs[:1,i:(i+3)], sharex = axes_cumsum_modified[int(i/3)], sharey = axes_probcol[0]))
                    
    # Fig Sup 3A

    axes_cumsum_modified[0], axes_probcol[0] = fig_cumsum(axes_cumsum_modified[0], axes_probcol[0], 'p', p_set, AB_i, be_spe, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'A', True, False, 'upper left', 'output_sde/23C')

    # Fig Sup 3B

    axes_cumsum_modified[1], axes_probcol[1] = fig_cumsum(axes_cumsum_modified[1], axes_probcol[1], 'm', m_set, AB_i, be_spe, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'B', False, False, '', 'output_sde/23D')

    # Fig Sup 3C

    axes_cumsum_modified[2], axes_probcol[2] = fig_cumsum(axes_cumsum_modified[2], axes_probcol[2], 't', t_set, AB_i, be_spe, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'C', False, False, '', 'output_sde/23E')
  
    # Fig Sup 3D

    axes_cumsum_modified[3], axes_probcol[3] = fig_cumsum(axes_cumsum_modified[3], axes_probcol[3], 'N', N_set, AB_i, be_spe, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'D', False, False, '', 'output_sde/23F')
    
    legend_elements = [Line2D([0],[0], color='k', marker='x', markerfacecolor='k', label='no inh.', lw=0)]

    legend_elements.append(Line2D([0],[0], marker='.', color='k', markerfacecolor='k', label='inh.', lw=0))
    
    legend_elements = legend_elements + [Line2D([0],[0], marker='_', color=cm.plasma(col/1.2), label='percentile %i'%(col*100)) for col in np.arange(0.1,1.1,.1)]

    legend_elements.append(Line2D([0],[0], marker='_', color='k', label='average'))

    legend_elements.append(Line2D([0],[0], linestyle = ':', color='k', label=r'$p_1$'))
    
    axes_cumsum_modified[0].legend(handles=legend_elements, bbox_to_anchor=(0.25, -0.4), loc='lower left', ncol = 7)
    
    fig.subplots_adjust(bottom=0., top=1., left=0., right=1., wspace=0.3, hspace=0.05)
    
    mp.savefig('figsup3.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')
    
    
# Figure Sup 4
    
def figsup4(spe_args):
    
    rep_spe = 0

    [N_spe, m_spe, t_spe, p_spe, AB_i, Be] = spe_args

    p_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    m_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    t_set = [i * j for j in np.logspace(-7, -2, 6) for i in [1., 2., 4., 7.]][:-3]

    N_set = [i * j for j in np.logspace(3, 7, 5) for i in [1., 2., 4., 7.]][:-3]

    fig, axes = mp.subplots(nrows = 2, ncols = 3, figsize = (9.9, 6.4))

    # Fig Sup 4A

    axes[0,0] = fig_timeseries(axes[0,0], N_spe, m_spe, t_spe, p_spe, Be, AB_i, rep_spe, 'focal', 'exp_value', 'A', 'center left', 'output_sde/23E', 100, [1, None], [8E-6, 1E-4], 'simulated populations')

    # Fig Sup 4B

    axes[0,1] = fig_Diff(axes[0,1], 'p', p_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'exp_value', 'B', 'upper left', 'left', 'output_sde/23C')
  
    # Fig Sup 4C

    axes[0,2] = fig_Diff(axes[0,2], 'm', m_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'exp_value', 'C', 'upper left', 'left', 'output_sde/23D')

    # Fig Sup 4D
    
    axes[1,0] = fig_Diff(axes[1,0], 't', t_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'exp_value', 'D', 'lower right', 'left', 'output_sde/23E')

    # Fig Sup 4E
    
    axes[1,1] = fig_Diff(axes[1,1], 'N', N_set, AB_i, Be, N_spe, m_spe, t_spe, p_spe, rep_spe, 'focal', 'exp_value', 'E', 'center right', 'left', 'output_sde/23F')

    axes[1,2].axis('off')

    fig.subplots_adjust(bottom=0., top=1., left=0., right=1., wspace=0.3, hspace=0.5)

    mp.savefig('figsup4.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')
    

# Figure Sup 5
    
def figsup5(spe_args):
    
    rep_spe = 0
    
    [N_spe, m_spe, t_spe, p_spe, AB_i, Be] = spe_args
    
    p_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    m_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    t_set = [i * j for j in np.logspace(-7, -2, 6) for i in [1., 2., 4., 7.]][:-3]

    N_set = [i * j for j in np.logspace(3, 7, 5) for i in [1., 2., 4., 7.]][:-3]
        
    fig = mp.figure(figsize = (9.9, 2.5))

    gs = gridspec.GridSpec(1, 2, figure = fig, hspace = 0.5, wspace = 0.15)

    gs00 = gridspec.GridSpecFromSubplotSpec(9, 8, subplot_spec=gs[0], hspace = 0.9)
    
    gs01 = gridspec.GridSpecFromSubplotSpec(9, 8, subplot_spec=gs[1], hspace = 0.9)

    axes_00_0 = fig.add_subplot(gs00[:4, :])

    axes_01_0 = fig.add_subplot(gs01[:4, :])
    
    axes_00_1 = fig.add_subplot(gs00[5:, :])
    
    axes_01_1 = fig.add_subplot(gs01[5:, :])
   
    # Fig Sup 5A
    
    axes_00_0 = fig_pcolor(axes_00_0, 'p', p_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[0], rep_spe, 'exp_value', r'slower col. ($\alpha_0 = %0.1f$)'%Be[0], 'A', 'output_sde/sup5A')

    axes_00_1 = fig_pcolor(axes_00_1, 'p', p_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[1], rep_spe, 'exp_value', r'faster col. ($\alpha_0 = %0.1f$)'%Be[1], None, 'output_sde/sup5A')
    
    # Fig Sup 5B
    
    axes_01_0 = fig_pcolor(axes_01_0, 'm', m_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[0], rep_spe, 'exp_value', r'slower col. ($\alpha_0 = %0.1f$)'%Be[0], 'B', 'output_sde/sup5B')
    
    axes_01_1 = fig_pcolor(axes_01_1, 'm', m_set, AB_i, N_spe, m_spe, t_spe, p_spe, Be[1], rep_spe, 'exp_value', r'faster col. ($\alpha_0 = %0.1f$)'%Be[1], None, 'output_sde/sup5B')
    
    axes_00_1.text(-0.2, 1.6, 'inheritance', fontsize = 16, rotation = 'vertical', transform = axes_00_1.transAxes)    
    
    axes_00_vl = [i*j for i in [3,5,6,8,9] for j in np.logspace(-5,-1,5)]
    
    axes_01_vl = [i*j for i in [3,5,6,8,9] for j in np.logspace(-5,-1,5)]
    
    for vl in axes_00_vl: axes_00_0.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
    
    for vl in axes_00_vl: axes_00_1.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
    
    for vl in axes_01_vl: axes_01_0.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
    
    for vl in axes_01_vl: axes_01_1.axvline(vl, 0, 3, color = 'w', linewidth = 0.25)
    
    fig.subplots_adjust(bottom=0., top=1., left=0., right=0.86)#, wspace=0.3, hspace=0.05
    
    cb_ax = fig.add_axes([0.88, -0.1, 0.13, 1.2])

    cb_ax.axis('off')

    img = cb_ax.imshow(np.array([[1E-7,1E0]]), cmap = mp.cm.inferno, norm=colors.LogNorm(vmin=1E-7, vmax=1E0))

    img.set_visible(False)

    cb_ax.set_aspect('auto')

    cbar = fig.colorbar(img, orientation = "vertical", ax = cb_ax, fraction = 1.0)

    cbar.ax.tick_params(labelsize = 12)

    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation = 90)

    cbar.set_label(r'average frequency ($\overline{x_1}$)',fontsize = 16)
    
    mp.savefig('figsup5.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')
 
    
# Figure Sup 6

def figsup6(spe_args):
    
    taxa = 'all'
    
    rep_spe = 0

    [N_spe, m_spe, t_spe, p_spe, Be] = spe_args

    m_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    t_set = [i * j for j in np.logspace(-7, -2, 6) for i in [1., 2., 4., 7.]][:-3]

    N_set = [i * j for j in np.logspace(3, 7, 5) for i in [1., 2., 4., 7.]][:-3]

    fig, axes = mp.subplots(nrows = 1, ncols = 3, figsize = (9., 3.6))

    a_i = 1E1
    
    B_i = [1E2, 1E2]
    
    Be = [1E-1, 5E-1]

    markers = ['v', 'X']
    
    marker_label = [r'$\alpha_0 = 0.1$', r'$\alpha_0 = 0.5$']

    for ind in range(2):
        
        # Fig Sup 6A
        
        x_coord, y_coord = [], []
        
        for m in m_set:

            name_baseline = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23D', N_spe, m, t_spe, p_spe, Be[ind], 1E0, 1E0, 1E1, 1E1, rep_spe)
                
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23D', N_spe, m, t_spe, p_spe, Be[ind], a_i, a_i, B_i[ind], B_i[ind], rep_spe)
                
            dataset_baseline = np.load(name_baseline)
        
            dataset_modified = np.load(name_modified)
        
            if taxa == 'focal':
        
                data_baseline = np.sort(dataset_baseline['stationary_dist'][0, :])

                data_modified = np.sort(dataset_modified['stationary_dist'][0, :])
            
            elif taxa == 'all':
            
                data_baseline = np.sort(dataset_baseline['stationary_dist'].sum(0))

                data_modified = np.sort(dataset_modified['stationary_dist'].sum(0))
                    
            x_coord.append(diff_prob(data_modified, data_baseline, N_spe, 'exp_value'))
            
            y_coord.append(diff_prob(data_modified, data_baseline, N_spe, 'prob_col'))
            
        axes[0].scatter(x = x_coord, y = y_coord, c = m_set, vmin = m_set[0], vmax = m_set[-1], cmap = 'viridis', marker = markers[ind], norm = colors.LogNorm(vmin=1E-5, vmax=1E0), linewidths=0.2)
    
        # Fig Sup 6B

        x_coord, y_coord = [], []
        
        for t in t_set:
            
            name_baseline = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23E', N_spe, m_spe, t, p_spe, Be[ind], 1E0, 1E0, 1E1, 1E1, rep_spe)
                
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23E', N_spe, m_spe, t, p_spe, Be[ind], a_i, a_i, B_i[ind], B_i[ind], rep_spe)
                
            dataset_baseline = np.load(name_baseline)
        
            dataset_modified = np.load(name_modified)
        
            if taxa == 'focal':
        
                data_baseline = np.sort(dataset_baseline['stationary_dist'][0, :])

                data_modified = np.sort(dataset_modified['stationary_dist'][0, :])
            
            elif taxa == 'all':
            
                data_baseline = np.sort(dataset_baseline['stationary_dist'].sum(0))

                data_modified = np.sort(dataset_modified['stationary_dist'].sum(0))
                    
            x_coord.append(diff_prob(data_modified, data_baseline, N_spe, 'exp_value'))
            
            y_coord.append(diff_prob(data_modified, data_baseline, N_spe, 'prob_col'))
            
        axes[1].scatter(x = x_coord, y = y_coord, c = t_set, vmin = t_set[0], vmax = t_set[-1], cmap = 'viridis', marker = markers[ind], norm = colors.LogNorm(vmin=1E-7, vmax=1E-2), linewidths=0.2)
  
        # Fig Sup 6C

        x_coord, y_coord = [], []
        
        for N in N_set:

            name_baseline = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23F', N, m_spe, t_spe, p_spe, Be[ind], 1E0, 1E0, 1E1, 1E1, rep_spe)
                
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23F', N, m_spe, t_spe, p_spe, Be[ind], a_i, a_i, B_i[ind], B_i[ind], rep_spe)
                
            dataset_baseline = np.load(name_baseline)
        
            dataset_modified = np.load(name_modified)
        
            if taxa == 'focal':
        
                data_baseline = np.sort(dataset_baseline['stationary_dist'][0, :])

                data_modified = np.sort(dataset_modified['stationary_dist'][0, :])
            
            elif taxa == 'all':
            
                data_baseline = np.sort(dataset_baseline['stationary_dist'].sum(0))

                data_modified = np.sort(dataset_modified['stationary_dist'].sum(0))
                    
            x_coord.append(diff_prob(data_modified, data_baseline, N, 'exp_value'))
            
            y_coord.append(diff_prob(data_modified, data_baseline, N, 'prob_col'))
            
        axes[2].scatter(x = x_coord, y = y_coord, c = N_set, vmin = N_set[0], vmax = N_set[-1], cmap = 'viridis', marker = markers[ind], norm = colors.LogNorm(vmin=1E3, vmax=1E7), linewidths=0.2)

    axes[0].set_xscale('symlog', linthreshx=1E-3, subsx = range(2, 10))
    
    axes[0].set_yscale('symlog', linthreshy=1E-2, subsy = range(2, 10))

    axes[1].set_xscale('symlog', linthreshx=1E-3, subsx = range(2, 10))
    
    axes[1].set_yscale('symlog', linthreshy=1E-2, subsy = range(2, 10))
    
    axes[2].set_xscale('symlog', linthreshx=1E-3, subsx = range(2, 10))
    
    axes[2].set_yscale('symlog', linthreshy=1E-2, subsy = range(2, 10))
    
    axes[0].set_xlim(-1E-1,1E-1)
    
    axes[0].set_ylim(-1E-1,1E-1)
    
    axes[1].set_xlim(-1E-1,1E-1)
        
    axes[1].set_ylim(-1E-1,1E-1)
    
    axes[2].set_xlim(-1E-1,1E-1)
    
    axes[2].set_ylim(-1E-1,1E-1)
    
    axes[0].tick_params(axis='x', labelrotation=15)
    
    axes[1].tick_params(axis='x', labelrotation=15)
        
    axes[2].tick_params(axis='x', labelrotation=15)

    axes[0].set_ylabel('occ. difference\n(seed-like inh. - low inh.)', fontsize=14)

    axes[0].set_xlabel('average load difference\n(seed-like inh. - low inh.)', fontsize=14)
    
    axes[1].set_xlabel('average load difference\n(seed-like inh. - low inh.)', fontsize=14)
    
    axes[2].set_xlabel('average load difference\n(seed-like inh. - low inh.)', fontsize=14)
    
    axes[0].fill_between([-1E0, 1E0], -1E-2, 1E-2, color = 'gray', alpha = 0.1)
    
    axes[0].fill_between([-1E-3, 1E-3], -1E0, 1E0, color = 'gray', alpha = 0.1)
    
    axes[1].fill_between([-1E0, 1E0], -1E-2, 1E-2, color = 'gray', alpha = 0.1)
    
    axes[1].fill_between([-1E-3, 1E-3], -1E0, 1E0, color = 'gray', alpha = 0.1)
        
    axes[2].fill_between([-1E0, 1E0], -1E-2, 1E-2, color = 'gray', alpha = 0.1)

    axes[2].fill_between([-1E-3, 1E-3], -1E0, 1E0, color = 'gray', alpha = 0.1)
            
    fig.subplots_adjust(bottom=0.3, top=1., left=0., right=1., wspace=0.2, hspace=0.)
    
    cb_ax0 = fig.add_axes([1./90, 0.0, 24./90., 0.06])

    cb_ax1 = fig.add_axes([33./90., 0.0, 24./90., 0.06])

    cb_ax2 = fig.add_axes([65./90., 0.0, 24./90., 0.06])

    cb_ax0.axis('off')
    
    cb_ax1.axis('off')
        
    cb_ax2.axis('off')

    img0 = cb_ax0.imshow(np.array([[1E-5,1E0]]), cmap = mp.cm.viridis, norm=colors.LogNorm(vmin=1E-5, vmax=1E0))

    img1 = cb_ax1.imshow(np.array([[1E-7,1E-2]]), cmap = mp.cm.viridis, norm=colors.LogNorm(vmin=1E-7, vmax=1E-2))

    img2 = cb_ax2.imshow(np.array([[1E3,1E7]]), cmap = mp.cm.viridis, norm=colors.LogNorm(vmin=1E3, vmax=1E7))

    img0.set_visible(False)
    
    img1.set_visible(False)
        
    img2.set_visible(False)

    cbar0 = fig.colorbar(img0, orientation = "horizontal", ax = cb_ax0, fraction = 1.)

    cbar1 = fig.colorbar(img1, orientation = "horizontal", ax = cb_ax1, fraction = 1.)

    cbar2 = fig.colorbar(img2, orientation = "horizontal", ax = cb_ax2, fraction = 1.)

    cbar0.ax.tick_params(labelsize = 12)
    
    cbar1.ax.tick_params(labelsize = 12)
        
    cbar2.ax.tick_params(labelsize = 12)

    cbar0.set_label(r'prob. of migration ($m$)', fontsize = 16)

    cbar1.set_label(r'prob. of host death ($\tau$)', fontsize = 16)

    cbar2.set_label(r'carrying capacity ($N$)', fontsize = 16)
    
    axes[0].tick_params(axis = 'both', which = 'both', direction = 'in')
    
    axes[1].tick_params(axis = 'both', which = 'both', direction = 'in')
    
    axes[2].tick_params(axis = 'both', which = 'both', direction = 'in')
    
    axes[0].xaxis.set_ticks_position('both')
    
    axes[1].xaxis.set_ticks_position('both')
    
    axes[2].xaxis.set_ticks_position('both')

    axes[0].yaxis.set_ticks_position('both')
    
    axes[1].yaxis.set_ticks_position('both')
    
    axes[2].yaxis.set_ticks_position('both')

    axes[0].text(-0.18, 1.03, 'A', transform = axes[0].transAxes, fontsize = 25, fontweight = 'bold')
    
    axes[1].text(-0.18, 1.03, 'B', transform = axes[1].transAxes, fontsize = 25, fontweight = 'bold')
    
    axes[2].text(-0.18, 1.03, 'C', transform = axes[2].transAxes, fontsize = 25, fontweight = 'bold')
    
    legend_elements = [Line2D([0],[0],marker=markers[i], color='w', label=marker_label[i], markerfacecolor='k', markersize=8) for i in range(2)]

    axes[1].legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.85), loc='lower center', ncol = 2, title = r"faster col. $\alpha_0 \to 0$")

    mp.savefig('figsup6.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')


# Figure Sup 7  

def figsup7(spe_args):
    
    taxa = 'focal'
    
    rep_spe = 0

    [N_spe, m_spe, t_spe, p_spe, Be] = spe_args
    
    p_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    m_set = [i * j for j in np.logspace(-5, 0, 6) for i in [1., 2., 4., 7.]][:-3]

    t_set = [i * j for j in np.logspace(-7, -2, 6) for i in [1., 2., 4., 7.]][:-3]

    N_set = [i * j for j in np.logspace(3, 7, 5) for i in [1., 2., 4., 7.]][:-3]

    fig, axes = mp.subplots(nrows = 2, ncols = 2, figsize = (7.4, 6.4))

    a_i = 1E1
    
    B_i = [1E2, 1E2]
    
    Be = [1E-1, 5E-1]

    markers = ['v', 'X']
    
    marker_label = [r'$\alpha_0 = 0.1$' , r'$\alpha_0 = 0.5$']

    for ind in range(2):
        
        # Fig Sup 7A
        
        x_coord, y_coord = [], []
        
        for p in p_set:

            name_baseline = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23C', N_spe, m_spe, t_spe, p, Be[ind], 1E0, 1E0, 1E1, 1E1, rep_spe)
                
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23C', N_spe, m_spe, t_spe, p, Be[ind], a_i, a_i, B_i[ind], B_i[ind], rep_spe)
                
            dataset_baseline = np.load(name_baseline)
        
            dataset_modified = np.load(name_modified)
        
            if taxa == 'focal':
        
                data_baseline = np.sort(dataset_baseline['stationary_dist'][0, :])

                data_modified = np.sort(dataset_modified['stationary_dist'][0, :])
            
            elif taxa == 'all':
            
                data_baseline = np.sort(dataset_baseline['stationary_dist'].sum(0))

                data_modified = np.sort(dataset_modified['stationary_dist'].sum(0))
                    
            x_coord.append(diff_prob(data_modified, data_baseline, N_spe, 'exp_value'))
            
            y_coord.append(diff_prob(data_modified, data_baseline, N_spe, 'prob_col'))
            
        axes[0,0].scatter(x = x_coord, y = y_coord, c = p_set, vmin = p_set[0], vmax = p_set[-1], cmap = 'viridis', marker = markers[ind], norm = colors.LogNorm(vmin=1E-5, vmax=1E0), linewidths=0.2)

        # Fig Sup 7B
        
        x_coord, y_coord = [], []
        
        for m in m_set:

            name_baseline = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23D', N_spe, m, t_spe, p_spe, Be[ind], 1E0, 1E0, 1E1, 1E1, rep_spe)
                
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23D', N_spe, m, t_spe, p_spe, Be[ind], a_i, a_i, B_i[ind], B_i[ind], rep_spe)
                
            dataset_baseline = np.load(name_baseline)
        
            dataset_modified = np.load(name_modified)
        
            if taxa == 'focal':
        
                data_baseline = np.sort(dataset_baseline['stationary_dist'][0, :])

                data_modified = np.sort(dataset_modified['stationary_dist'][0, :])
            
            elif taxa == 'all':
            
                data_baseline = np.sort(dataset_baseline['stationary_dist'].sum(0))

                data_modified = np.sort(dataset_modified['stationary_dist'].sum(0))
                    
            x_coord.append(diff_prob(data_modified, data_baseline, N_spe, 'exp_value'))
            
            y_coord.append(diff_prob(data_modified, data_baseline, N_spe, 'prob_col'))
            
        axes[0,1].scatter(x = x_coord, y = y_coord, c = m_set, vmin = m_set[0], vmax = m_set[-1], cmap = 'viridis', marker = markers[ind], norm = colors.LogNorm(vmin=1E-5, vmax=1E0), linewidths=0.2)
    
        # Fig Sup 7C

        x_coord, y_coord = [], []
        
        for t in t_set:
            
            name_baseline = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23E', N_spe, m_spe, t, p_spe, Be[ind], 1E0, 1E0, 1E1, 1E1, rep_spe)
                
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23E', N_spe, m_spe, t, p_spe, Be[ind], a_i, a_i, B_i[ind], B_i[ind], rep_spe)
                
            dataset_baseline = np.load(name_baseline)
        
            dataset_modified = np.load(name_modified)
        
            if taxa == 'focal':
        
                data_baseline = np.sort(dataset_baseline['stationary_dist'][0, :])

                data_modified = np.sort(dataset_modified['stationary_dist'][0, :])
            
            elif taxa == 'all':
            
                data_baseline = np.sort(dataset_baseline['stationary_dist'].sum(0))

                data_modified = np.sort(dataset_modified['stationary_dist'].sum(0))
                    
            x_coord.append(diff_prob(data_modified, data_baseline, N_spe, 'exp_value'))
            
            y_coord.append(diff_prob(data_modified, data_baseline, N_spe, 'prob_col'))
            
        axes[1,0].scatter(x = x_coord, y = y_coord, c = t_set, vmin = t_set[0], vmax = t_set[-1], cmap = 'viridis', marker = markers[ind], norm = colors.LogNorm(vmin=1E-7, vmax=1E-2), linewidths=0.2)
  
        # Fig Sup 7D

        x_coord, y_coord = [], []
        
        for N in N_set:

            name_baseline = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23F', N, m_spe, t_spe, p_spe, Be[ind], 1E0, 1E0, 1E1, 1E1, rep_spe)
                
            name_modified = '%s/inh_1_N_%.0e_m_%.0e_t_%.0e_p_%.0e_be_%.0e_an_%.0e_ao_%.0e_bn_%.0e_bo_%.0e_%i.npz'%('output_sde/23F', N, m_spe, t_spe, p_spe, Be[ind], a_i, a_i, B_i[ind], B_i[ind], rep_spe)
                
            dataset_baseline = np.load(name_baseline)
        
            dataset_modified = np.load(name_modified)
        
            if taxa == 'focal':
        
                data_baseline = np.sort(dataset_baseline['stationary_dist'][0, :])

                data_modified = np.sort(dataset_modified['stationary_dist'][0, :])
            
            elif taxa == 'all':
            
                data_baseline = np.sort(dataset_baseline['stationary_dist'].sum(0))

                data_modified = np.sort(dataset_modified['stationary_dist'].sum(0))
                    
            x_coord.append(diff_prob(data_modified, data_baseline, N, 'exp_value'))
            
            y_coord.append(diff_prob(data_modified, data_baseline, N, 'prob_col'))
            
        axes[1,1].scatter(x = x_coord, y = y_coord, c = N_set, vmin = N_set[0], vmax = N_set[-1], cmap = 'viridis', marker = markers[ind], norm = colors.LogNorm(vmin=1E3, vmax=1E7), linewidths=0.2)

    axes[0,0].set_xscale('symlog', linthreshx=1E-3, subsx = range(2, 10))
    
    axes[0,0].set_yscale('symlog', linthreshy=1E-2, subsy = range(2, 10))
    
    axes[0,1].set_xscale('symlog', linthreshx=1E-3, subsx = range(2, 10))
    
    axes[0,1].set_yscale('symlog', linthreshy=1E-2, subsy = range(2, 10))

    axes[1,0].set_xscale('symlog', linthreshx=1E-3, subsx = range(2, 10))
    
    axes[1,0].set_yscale('symlog', linthreshy=1E-2, subsy = range(2, 10))
    
    axes[1,1].set_xscale('symlog', linthreshx=1E-3, subsx = range(2, 10))
    
    axes[1,1].set_yscale('symlog', linthreshy=1E-2, subsy = range(2, 10))
    
    axes[0,0].set_xlim(-1E-1,1E-1)
    
    axes[0,0].set_ylim(-1E-1,1E-1)
    
    axes[0,1].set_xlim(-1E-1,1E-1)
        
    axes[0,1].set_ylim(-1E-1,1E-1)

    axes[1,0].set_xlim(-1E-1,1E-1)
    
    axes[1,0].set_ylim(-1E-1,1E-1)
    
    axes[1,1].set_xlim(-1E-1,1E-1)
    
    axes[1,1].set_ylim(-1E-1,1E-1)
    
    axes[0,0].tick_params(axis='x', labelrotation=15)
    
    axes[0,1].tick_params(axis='x', labelrotation=15)
    
    axes[1,0].tick_params(axis='x', labelrotation=15)
        
    axes[1,1].tick_params(axis='x', labelrotation=15)

    axes[0,0].set_ylabel('occ. difference\n(seed-like inh. - low inh.)', fontsize=14)
    
    axes[1,0].set_ylabel('occ. difference\n(seed-like inh. - low inh.)', fontsize=14)

    axes[1,0].set_xlabel('average freq. difference\n(seed-like inh. - low inh.)', fontsize=14)
    
    axes[1,1].set_xlabel('average freq. difference\n(seed-like inh. - low inh.)', fontsize=14)
    
    axes[0,0].fill_between([-1E0, 1E0], -1E-2, 1E-2, color = 'gray', alpha = 0.1)
    
    axes[0,0].fill_between([-1E-3, 1E-3], -1E0, 1E0, color = 'gray', alpha = 0.1)
    
    axes[0,1].fill_between([-1E0, 1E0], -1E-2, 1E-2, color = 'gray', alpha = 0.1)
    
    axes[0,1].fill_between([-1E-3, 1E-3], -1E0, 1E0, color = 'gray', alpha = 0.1)
        
    axes[1,0].fill_between([-1E0, 1E0], -1E-2, 1E-2, color = 'gray', alpha = 0.1)
    
    axes[1,0].fill_between([-1E-3, 1E-3], -1E0, 1E0, color = 'gray', alpha = 0.1)
            
    axes[1,1].fill_between([-1E0, 1E0], -1E-2, 1E-2, color = 'gray', alpha = 0.1)
    
    axes[1,1].fill_between([-1E-3, 1E-3], -1E0, 1E0, color = 'gray', alpha = 0.1)
    
    fig.subplots_adjust(bottom=0., top=1., left=0., right=0.9, wspace=0.4, hspace=0.35)
    
    cb_ax00 = fig.add_axes([0.39, 0.6, 0.1, 0.3])

    cb_ax01 = fig.add_axes([0.915, 0.6, 0.1, 0.3])
    
    cb_ax10 = fig.add_axes([0.39, 0.1, 0.1, 0.3])

    cb_ax11 = fig.add_axes([0.915, 0.1, 0.1, 0.3])

    cb_ax00.axis('off')

    cb_ax01.axis('off')
    
    cb_ax10.axis('off')
        
    cb_ax11.axis('off')

    img00 = cb_ax00.imshow(np.array([[1E-5,1E0]]), cmap = mp.cm.viridis, norm=colors.LogNorm(vmin=1E-5, vmax=1E0))

    img01 = cb_ax01.imshow(np.array([[1E-5,1E0]]), cmap = mp.cm.viridis, norm=colors.LogNorm(vmin=1E-5, vmax=1E0))

    img10 = cb_ax10.imshow(np.array([[1E-7,1E-2]]), cmap = mp.cm.viridis, norm=colors.LogNorm(vmin=1E-7, vmax=1E-2))

    img11 = cb_ax11.imshow(np.array([[1E3,1E7]]), cmap = mp.cm.viridis, norm=colors.LogNorm(vmin=1E3, vmax=1E7))

    img00.set_visible(False)
    
    img01.set_visible(False)
    
    img10.set_visible(False)
        
    img11.set_visible(False)

    cbar00 = fig.colorbar(img00, orientation = "vertical", ax = cb_ax00, fraction = 1.)

    cbar01 = fig.colorbar(img01, orientation = "vertical", ax = cb_ax01, fraction = 1.)

    cbar10 = fig.colorbar(img10, orientation = "vertical", ax = cb_ax10, fraction = 1.)

    cbar11 = fig.colorbar(img11, orientation = "vertical", ax = cb_ax11, fraction = 1.)

    cbar00.ax.tick_params(labelsize = 12)

    cbar01.ax.tick_params(labelsize = 12)
    
    cbar10.ax.tick_params(labelsize = 12)
        
    cbar11.ax.tick_params(labelsize = 12)
    
    axes[0,0].set_title(r'freq. in colonizers pool ($p_1$)', fontsize = 16)

    axes[0,1].set_title(r'prob. of migration ($m$)', fontsize = 16)

    axes[1,0].set_title(r'prob. of host death ($\tau$)', fontsize = 16)

    axes[1,1].set_title(r'carrying capacity ($N$)', fontsize = 16)
    
    axes[0,0].tick_params(axis = 'both', which = 'both', direction = 'in')

    axes[0,1].tick_params(axis = 'both', which = 'both', direction = 'in')
    
    axes[1,0].tick_params(axis = 'both', which = 'both', direction = 'in')
    
    axes[1,1].tick_params(axis = 'both', which = 'both', direction = 'in')
    
    axes[0,0].xaxis.set_ticks_position('both')
        
    axes[0,1].xaxis.set_ticks_position('both')
    
    axes[1,0].xaxis.set_ticks_position('both')
    
    axes[1,1].xaxis.set_ticks_position('both')

    axes[0,0].yaxis.set_ticks_position('both')
    
    axes[0,1].yaxis.set_ticks_position('both')
    
    axes[1,0].yaxis.set_ticks_position('both')
    
    axes[1,1].yaxis.set_ticks_position('both')

    axes[0,0].text(-0.23, 1.01, 'A', transform = axes[0,0].transAxes, fontsize = 25, fontweight = 'bold')

    axes[0,1].text(-0.23, 1.01, 'B', transform = axes[0,1].transAxes, fontsize = 25, fontweight = 'bold')
    
    axes[1,0].text(-0.23, 1.01, 'C', transform = axes[1,0].transAxes, fontsize = 25, fontweight = 'bold')
    
    axes[1,1].text(-0.23, 1.01, 'D', transform = axes[1,1].transAxes, fontsize = 25, fontweight = 'bold')
    
    legend_elements = [Line2D([0],[0],marker=markers[i], color='w', label=marker_label[i], markerfacecolor='k', markersize=8) for i in range(2)]

    axes[1,1].legend(handles=legend_elements, bbox_to_anchor=(1.25, 1.25), loc='upper left', ncol = 1, title = r"faster col. $\alpha_0 \to 0$")

    mp.savefig('figsup7.pdf', dpi = 300, bbox_inches = 'tight', format = 'pdf')
