#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: Stochastic Inheritance of the microbiome (execute - figures)
@author:Román Zapién-Campos - 2021
(MPI for Evolutionary Biology - zapien@evolbio.mpg.de)
"""

# Import functions to make figures from the script sc_figs.py
from sc_figs import fig2, fig3, fig4, fig5, figsup1, figsup2, figsup3, figsup4, figsup5, figsup6


# Figure 2 (Microbial occurrence in hosts under microbial inheritance.)
# Arrows and lines have been added later using the software Inkscape 1.0.2

N_spe, m_spe, t_spe, p_spe, AB_i, Be = 1E5, 1E-2, 1E-4, 1E-2, [1, 1, 1E1, 1E1], [1E-1, 5E-1]

fig2([N_spe, m_spe, t_spe, p_spe, AB_i, Be])


# Figure 3 (Average microbial load in hosts under microbial inheritance.)

N_spe, m_spe, t_spe, p_spe, AB_i, Be = 1E5, 1E-2, 1E-4, 1E-2, [1, 1, 1E1, 1E1], [1E-1, 5E-1]

fig3([N_spe, m_spe, t_spe, p_spe, AB_i, Be])


# Figure 4 (Effect of asymmetric inheritance on the average frequency of a taxon in hosts.)

N_spe, m_spe, t_spe, p_spe, AB_i, Be = 1E5, 1E-2, 1E-4, 1E-2, [[1, 1, 1E1, 1E1], [1, 1, 1E1, 1E2], [1, 1, 1E1, 1E3]], [5E-1, 1E-1]

fig4([N_spe, m_spe, t_spe, p_spe, AB_i, Be])


# Figure 5 (Persistence of lineage taxa in hosts.)
# Arrows have been added later using the software Inkscape 1.0.2

N_spe, Be = 1E5, 1E-1

fig5([N_spe, Be])


# Figure Sup 1 (Occurrence of a microbial taxon in hosts under microbial inheritance.)

N_spe, m_spe, t_spe, p_spe, AB_i, Be = 1E5, 1E-2, 1E-4, 1E-2, [1, 1, 1E1, 1E1], [1E-1, 5E-1]

figsup1([N_spe, m_spe, t_spe, p_spe, AB_i, Be])


# Figure Sup 2 (Microbial load distribution across a host population, with or without microbial inheritance.)

N_spe, m_spe, t_spe, p_spe, AB_i, be_spe = 1E5, 1E-2, 1E-4, 1E-2, [1, 1, 1E1, 1E1], 1E-1

figsup2([N_spe, m_spe, t_spe, p_spe, AB_i, be_spe])


# Figure Sup 3 (Frequency of a microbial taxon distribution across the host population, with or without inheritance.)

N_spe, m_spe, t_spe, p_spe, AB_i, be_spe = 1E5, 1E-2, 1E-4, 1E-2, [1, 1, 1E1, 1E1], 1E-1

figsup3([N_spe, m_spe, t_spe, p_spe, AB_i, be_spe])


# Figure Sup 4 (Average frequency of a microbial taxon in hosts under microbial inheritance.)

N_spe, m_spe, t_spe, p_spe, AB_i, Be = 1E5, 1E-2, 1E-4, 1E-2, [1, 1, 1E1, 1E1], [1E-1, 5E-1]

figsup4([N_spe, m_spe, t_spe, p_spe, AB_i, Be])


# Figure Sup 5 (Effect of selective inheritance - focal microbial taxon)

N_spe, m_spe, t_spe, p_spe, AB_i, Be = 1E5, 1E-2, 1E-6, 1E-2, [[1, 1, 1E1, 1E1], [1, 1, 1E1, 1E2], [1, 1, 1E1, 1E3]], [5E-1, 1E-1]

figsup5([N_spe, m_spe, t_spe, p_spe, AB_i, Be])


# Figure Sup 6 (Comparison of the modes of inheritance - microbial load)

N_spe, m_spe, t_spe, p_spe, Be = 1E5, 1E-2, 1E-4, 1E-2, 1E-1

figsup6([N_spe, m_spe, t_spe, p_spe, Be])


# Figure Sup 7 (Comparison of the modes of inheritance - focal microbial taxon)

N_spe, m_spe, t_spe, p_spe, Be = 1E5, 1E-2, 1E-4, 1E-2, 1E-1

figsup7([N_spe, m_spe, t_spe, p_spe, Be])
