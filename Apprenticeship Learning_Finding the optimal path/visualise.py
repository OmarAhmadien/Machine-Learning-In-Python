# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:56:25 2016

@author: jarethmoyo
"""

import cPickle as pickle
import grid_implementation as gi
import RL_model as rl
import Tkinter as tk

# you may ignore this function
def grid_setup(configure):
    term_states=[]
    term_state_vals={}
    for m in range(len(configure)):
        for n in range(len(configure[0])):
            if configure[m][n]==2:
                term_states.append((len(configure)-m-1,n))
                term_state_vals[(len(configure)-m-1,n)]=-10
            elif configure[m][n]==1:
                term_states.append((len(configure)-m-1,n))
                term_state_vals[(len(configure)-m-1,n)]=-5
            elif configure[m][n]==3:
                term_states.append((len(configure)-m-1,n))
                term_state_vals[(len(configure)-m-1,n)]=30
    return configure, term_states, term_state_vals

# you may ignore this function
def move_on_grid(grid, policy):
    mat = grid_setup(grid)
    config = mat[0]
    terminal_states = mat[1]
    terminal_values = mat[2]
    grid=rl.Grid(8,8,terminal_values,terminal_states)
    grid.compute_v_values(-1.)
    path= grid.convert_policy_to_path(policy,terminal_states)
    root = tk.Tk()
    app=gi.App(root, config)
    app.move_on_path(path)
    root.mainloop()


dataset = pickle.load(open('training_data.p', 'rb'))
sample_grid = dataset[0][0]
sample_policy = dataset[0][1]
move_on_grid(sample_grid, sample_policy)
