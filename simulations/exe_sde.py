#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: Stochastic inheritance of the microbiome (execute - SDE)
@author: Román Zapién-Campos - 2021
(MPI for Evolutionary Biology - zapien@evolbio.mpg.de)
"""

# Import relevant packages

from sys import argv

from datetime import datetime

from sc_sde import solve_host_pop


time_start = datetime.now()

# Input parameters

args = argv[1:]

# Solve the system

solve_host_pop(args)

print(datetime.now() - time_start)