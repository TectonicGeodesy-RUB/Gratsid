#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:32:16 2023

This module contains:
    
- 'gratsid_fit' function for performing gratsid algorithm
- 'fit_decompose' function for fitting time series given 
   the tables output by gratsid and the options dictionary

When importing functions from gratsid.py, you also import functions that 
are imported in other modules.
Flow of functions through modules:
gratsid.py  <--- gratsid_iteration_stages.py <----- gd_functions.py <--- common_functions.py

Therefore, by running:
"from gratsid import *"
you are also importing all functions found in modules:
gratsid_iteration_stages.py
gd_functions.py
common_functions.py


Practically, these two functions (gratsid_fit, fit_decompose) are what the 
gratsid user will run.

In running function gratsid_fit, you have initial "stages" that QC data and
perform a first fit.

- Stage 1: QC-ing input data
- Stage 2: performing initial fit to data to create initial residual
- Stage 3: Initializing a range of parameters and lists that will be used and 
           filled after successive iterations inside the while loop.

Then you perform "gratsid iterations" which consist of several stages inside a
while loop:
 - Stage 4: finding new potential transient onsets from residual vectors
 - Stage 5: combinatorial analysis to find which combination of transient onsets 
             improve fit beyond threshold, while minimizing number of transients
 - Stage 6: Sequentially adjusting slightly the existing transient onsets 
             (in time and type of function) to further improve fit.
 - Stage 7: Evaluating current iteration and preparing the next iteration.
 
Finally, in Stage 8, we output the information that allows a user to recreate
the fits from tables listing transient onset times and the same options used
as input into the 
 
According to the number of solutions (options['nsols']), we converge this many
times inside the while loop, and then the while loop is broken and the 
function returns tables of onset times and types of transient.  These tables
can then be fed into function "fit_decompose", along with the same options used
in the gratsid_fit function, to output the trajectory model and residuals.
             


@author: jon
"""

from gratsid_iteration_stages import *


def gratsid_fit(x,y,err,known_steps,options):
    options_out = options.copy()

    ######################################
    ### Stage 1: QC-ing input data #######
    ######################################
    x, y, err, ndims, proceed_to_signal_decomposition, known_steps = \
        initial_checks(x, y, err, options, known_steps)

    # return x,y,err,ndims,proceed_to_signal_decomposition, known_steps

    ######################################
    ### Stage 2: Initial fit #############
    ######################################
    perm_table, residual, current_misfit, perm_steps_list = \
        do_initial_fit(x, y, err, options, known_steps)

    # return perm_table, residual, current_misfit, perm_steps_list

    #############################################################
    ### Stage 3: Initializing outside of while loop #############
    #############################################################
    current_misfit, previous_misfit, fractional_tolerance_triggered, \
    trans_table, previous_trans_table, no_improvement_count, \
    quarantine_list, sols, progress = \
        initialize_before_while(current_misfit)
    if options['verbose'] == True:
        print('Misfit after initial fit:', current_misfit)
    # return current_misfit,previous_misfit,fractional_tolerance_triggered,\
    #    trans_table,previous_trans_table,no_improvement_count,\
    #    quarantine_list, sols, progress,target_reached

    while len(sols) < options['nsols']:
        #######################################################################
        ##Stage 4: finding new potential transient onsets from residual vectors
        #######################################################################
        candidates_table = \
            find_new_transient_onsets(residual, quarantine_list, options, \
                                      x, err, perm_steps_list)
        # break
        # return candidates_table, residual

        #######################################################################
        ##Stage 5: Combinatorial analysis
        #######################################################################
        if options['verbose'] == True:
            print('Misfit before comb.-analysis:', current_misfit)
        trans_table, current_misfit, swap_out_flag, no_improvement_count = \
            combinatorial_analysis(previous_trans_table,
                                   candidates_table, x, y, err, no_improvement_count,
                                   current_misfit, options)
        if options['verbose'] == True:
            print('Misfit after comb.-analysis:', current_misfit)
        # break
        # return 10

        #######################################################################
        ##Stage 6: Sequential adjustment loop
        #######################################################################
        if swap_out_flag == 1:
            trans_table, current_misfit, residual = swap_out(perm_steps_list, perm_table, trans_table, \
                                                             current_misfit, x, y, err, options)
            if options['verbose'] == True:
                print('Misfit after swap-out:', current_misfit)
        # break
        # return 10

        #######################################################################
        ##Stage 7: Evaluating current iteration and preparing the next iteration
        #######################################################################
        if options['verbose'] == True:
            print('End of gratsid iteration,')
            print('# transients found: ', trans_table.shape[0])
        progress, quarantine_list, previous_trans_table, previous_misfit, sols, \
        trans_table, current_misfit, previous_misfit, \
        no_improvement_count, converged, residual = \
            end_of_gratsid_iteration_evaluation(progress, trans_table, \
                                                candidates_table, current_misfit, options, \
                                                no_improvement_count, x, y, err, sols, perm_table, residual)
        if options['verbose'] == True:
            if converged == True:
                print('******************************************************')
                print('**************************************')
                print('****************************')
                print('****************')
                print('CONVERGED solution', str(len(sols)), ' of ', str(options['nsols']))
                print('****************')
                print('****************************')
                print('**************************************')
                print('******************************************************')
                print('')
            else:
                print('')
                print('Still searching solution ', str(len(sols) + 1), ' of ', str(options['nsols']))

    #######################################################################
    ##Stage 8: Outputting the things that should be saved
    #######################################################################

    return perm_table, sols, options_out


def fit_decompose(x, y, err, sols, perm_table, options):
    if len(y.shape) < 2:
        y = y.reshape(y.size, 1)
        if err != None:
            if len(err.shape) < 2:
                err = err.reshape(err.size, 1)
    TO_types = np.arange(5)  ## needs to be all possible types of TO plus extra index for residuals
    signal = []
    for i in range(TO_types.size):
        signal.append([])
    signal.append([])  ## the final portion of the decomposed signal will be the residual

    for i in range(len(sols)):
        table = np.vstack([perm_table, sols[i][-1]])
        G, m_keys = assemble_G_return_keys(x, table, options)

        if options['gradient_descent'] == True:

            if options['opt_name'] == 'Adam':
                options['opt'] = tf.keras.optimizers.legacy.Adam()
            
            options['osc_cols'] = np.arange(m_keys.size)[m_keys == 2]  ### to be used as a global variable
            options['S'] = my_temporal_smoothing(G.shape[0], options['S_each_side'])
            options['W'] = np.random.randn(G.shape[0], len(options['osc_cols']))
            options['W'] = tf.convert_to_tensor(options['W'].astype('float32'))
            residual, misfit, m, W_best = \
                single_fit_predict_GD(G, y, err, options)
            ### Now making a new G so that we can get predictions with distored oscillations
            G = distort_osc_G(G, W_best, options)

        else:
            residual, residual_val, m = single_fit_predict_SNE(G, y, err, options)

        for j in range(TO_types.size):
            signal[j].append(np.matmul(G[:, m_keys == j], m[m_keys == j]))
        signal[-1].append(residual)

    return signal
