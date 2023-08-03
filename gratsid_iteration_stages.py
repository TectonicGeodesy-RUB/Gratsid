#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:22:06 2023

@author: jon
"""

from gd_functions import *


def do_initial_fit(x, y, err, options, known_steps):
    perm_steps_list = generate_permanent_step_list(x, known_steps,
                                                   options)  # <---- should max gap be for the sampling or the time?
    if perm_steps_list.size == 0:
        perm_steps_list = np.array([]).reshape(0, 3)
    elif (len(perm_steps_list.shape) == 1) * (perm_steps_list.size != 0) == 1:
        perm_steps_list = perm_steps_list.reshape(1, 3)
    # "#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,
    # "#, Gathering the permanent table of keys #,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"
    perm_table = gather_perm_table(options, perm_steps_list)
    G_perm = assemble_G(x, perm_table, options)
    options['G_perm_undistorted'] = G_perm.copy()
    # "#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,
    # "#, If using GD, then we need to know which columns correspond to
    # "#,  oscillation/periodic Green's functions
    G_throw, m_keys = assemble_G_return_keys(x, perm_table, options)
    options['osc_cols'] = np.arange(m_keys.size)[m_keys == 2]  ### to be used as a global variable
    options['S'] = my_temporal_smoothing(G_perm.shape[0], options['S_each_side'])
    options['W'] = np.random.randn(G_throw.shape[0], len(options['osc_cols']))
    options['W'] = tf.convert_to_tensor(options['W'].astype('float32'))
    # "#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,
    # "#, Calculating the intitial residual #,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"

    if options['gradient_descent'] == False:
        residual, current_misfit, m = single_fit_predict_SNE( \
            G_in=G_perm, y_in=y, err_in=err, options=options)
    else:
        ## setting optimizer
        ### future version will allow choice of optimizer...
        if options['opt_name'] == 'Adam':
            options['opt'] = tf.keras.optimizers.legacy.Adam()
            
            ### doing fit
        residual, current_misfit, m, W_out = \
            single_fit_predict_GD(G_perm, y, err, options)
        G_perm = distort_osc_G(G_perm, W_out, options)

    return perm_table, residual, current_misfit, perm_steps_list


def initialize_before_while(residual_val):
    current_misfit = residual_val  # needed for later in the combinatorial analysis
    previous_misfit = residual_val.copy()
    fractional_tolerance_triggered = False
    trans_table = np.array([]).reshape(0,
                                       3)  ## instantiating the table that will carry the changing list of transient functions
    previous_trans_table = np.array([]).reshape(0,
                                                3)  ## instantiating the table that will carry the changing list of transient functions
    no_improvement_count = 0
    quarantine_list = []
    sols = []
    progress = []
    target_reached = 0
    return current_misfit, previous_misfit, fractional_tolerance_triggered, \
           trans_table, previous_trans_table, no_improvement_count, \
           quarantine_list, sols, progress


def find_new_transient_onsets(residual, quarantine_list, options, \
                              x, err, perm_steps_list):
    TO_list = find_TO(residual, quarantine_list, options)  ### returns a python list
    G_TOs = make_TO_G_lists(options=options, x=x,
                            TO_list=TO_list)  ## with this python list making lists of greens functions lists depending on how many types of transients allowed
    candidates_table = evaluate_candidate_TOs(TO_list,
                                              G_TOs, x, residual, err, perm_steps_list, options)

    return candidates_table


def combinatorial_analysis(previous_trans_table, candidates_table,
                           x, y, err, no_improvement_count, current_misfit, options):
    """
    Main function for doing the combinatorial analysis of the 
    existing and candidate transient onsets in the time series
    """
    G_trans_list = []
    trans_width = []
    trans_table = np.vstack([previous_trans_table, candidates_table])
    trans_table = my_sortrows(trans_table, [0], [1])  # sort by ascending first column
    T = assemble_G(x, trans_table, options)  # All the transient functions in one matrix
    for i in range(trans_table.shape[0]):
        G_trans_list.append(assemble_G(x, trans_table[i, :][None, :],
                                       options))  # do we really need to repeat on each iteration? Can we rather save this?
        trans_width.append(G_trans_list[-1].shape[1])
    ## Make the transient function indices which reference columns in T
    cs = np.cumsum(trans_width)
    trans_indices = []
    for i in range(cs.size):
        trans_indices.append(np.arange(cs[i] - trans_width[i], cs[i], 1))
    ## Generating the combinations of functions that will be removed
    remove_n = 1
    remove_list = (np.arange(options['max_remove']) * np.nan)[None,
                  :]  # first row corresponds to none of the TOs being removed (all being kept)
    while (remove_n <= options['max_remove']) * (remove_n <= trans_table.shape[0]):
        new_list = np.array(list(itertools.combinations \
                                     (np.arange(trans_table.shape[0]), remove_n)))
        to_fill = np.zeros([len(new_list), options['max_remove']])
        to_fill.fill(np.nan)
        to_fill[:, 0:remove_n] = new_list.astype(int)
        remove_list = np.vstack([remove_list, to_fill])
        remove_n += 1
    ## So that we can invert matrixes of the same sizes as a batch with tf, 
    ## we need to track the width of each matrix with trnasient functions removed
    G_widths = np.ones([remove_list.shape[0]]) * (options['G_perm_undistorted'].shape[1] + np.sum(trans_width))
    for i in range(G_widths.size):
        r = (remove_list[i, :][~np.isnan(remove_list[i, :])]).astype(int)
        G_widths[i] += -1 * (np.sum(np.array(trans_width)[r]))
    ## Making table that references index of remove list (i), the number of transient functions removed (n), and misfit (mf)
    i_n_mf = np.hstack([np.arange(remove_list.shape[0])[:, None], \
                        (~np.isnan(remove_list)).sum(axis=1)[:, None], \
                        np.nan * (np.arange(remove_list.shape[0]))[:, None]])

    ## If doing gradient descent, we need to temporarily store the solved
    ## time weightings ('W') for each solution.  This is later used to update
    ##  G_perm.
    if options['gradient_descent'] == True:
        W_all = np.zeros([i_n_mf.shape[0], \
                          options['G_perm_undistorted'].shape[0], \
                          2 * len(options['osc_periods'])])
    ## Now, looping through unique matrix widths, batch inverting, and filling the i_n_mf table
    widths_unique = np.unique(G_widths)
    for i in range(widths_unique.size):
        ## First filling the transient portion of the G matrix
        ii = G_widths == widths_unique[i]
        T_new = np.zeros([ii.sum(), x.size, \
                          widths_unique[i].astype(int) - options['G_perm_undistorted'].shape[1]])
        for j in range(T_new.shape[0]):
            to_remove = np.array(trans_indices, dtype=object) \
                [(remove_list[ii, :][j, :][~np.isnan(remove_list[ii, :][j, :])]).astype(int)]
            remove_indices = get_indices(to_remove)  # function to unravel this array of arrays.
            T_new[j, :, :] = T[:, ~np.isin(np.arange(T.shape[1]), remove_indices)]
        ## Making the full matrix (as a tensor for the batch) then inserting G_perm and T
        G = np.zeros([ii.sum(), x.size, widths_unique[i].astype(int)])
        G[:, :, 0:options['G_perm_undistorted'].shape[1]] = \
            options['G_perm_undistorted']
        G[:, :, options['G_perm_undistorted'].shape[1]:] = T_new
        ## Inverting and saving residuals in table
        if options['gradient_descent'] == False:
            misfits = multi_fit_predict_SNE(G_in=G, y_in=y, err_in=err, options=options)
        else:
            misfits, W_out = multi_fit_predict_GD(G, y, err, options)
            W_all[ii] = W_out.copy()

        i_n_mf[ii, -1] = misfits

    ## Now determining if any of the misfits decrease by the minimum amount
    ## If so, edit the trans_table, if not, increase the no_improvement_count
    i_n_mf = i_n_mf[i_n_mf[:, -1] < (1 - options['fractional_tolerance']) * \
                    (current_misfit)]  ##

    i_n_mf = my_sortrows(i_n_mf, [1, 2], [-1, 1])  # sorting by minimum number of transients
    if i_n_mf.shape[0] > 0:
        trans_table = trans_table[~np.isin(np.arange(trans_table.shape[0]), \
                                           (remove_list[i_n_mf[0, 0].astype(int)] \
                                               [~np.isnan(remove_list[i_n_mf[0, 0].astype(int)])]).astype(int)), :]
        current_misfit = i_n_mf[0, -1]
        swap_out_flag = 1  ## we can proceed to swap-out phase
        no_improvement_count = 0  ## resetting to zero

        if options['gradient_descent'] == True:
            ii = int(i_n_mf[0, 0])
            options['G_perm_distorted'] = \
                distort_osc_G(options['G_perm_undistorted'], W_all[ii], options)
    else:
        trans_table = previous_trans_table
        no_improvement_count += 1
        swap_out_flag = 0  ## we can skip the swap-out phase

    return trans_table, current_misfit, swap_out_flag, no_improvement_count


def swap_out(perm_steps_list, perm_table, trans_table, current_misfit, x, y, err, \
             options):
    if options['gradient_descent'] == True:
        G_perm = options['G_perm_distorted']
    else:
        G_perm = options['G_perm_undistorted']

    i = 0
    carry_on = True
    no_change_count = 0
    while carry_on == True:
        if no_change_count == trans_table.shape[0]:
            carry_on = False
        else:
            misfit_start = current_misfit.copy()
            if i == trans_table.shape[0]:
                i = 0
            # Splitting the transient table into two groups, the single transient will be moved around and type will be changed
            single_trans = trans_table[i, :].copy()
            remaining_trans = trans_table[np.arange(trans_table.shape[0]) != i, :]
            G_remaining_trans = assemble_G(x, remaining_trans, options)
            indices = np.arange(single_trans[0] - options['each_side'], \
                                single_trans[0] + options['each_side'] + 1, 1)
            G_additional = []
            for j in range(len(options['TO_types'])):
                if options['TO_types'][j] == 0:  ## checking to see no collision with onset of permanent steps
                    indices = indices[~np.isin(indices, perm_steps_list[:, 0])]
                indices = indices[(indices >= 0) * (indices < x.size) == True]
                table = np.zeros([indices.size, 3])
                table[:, 1] = options['TO_types'][j]
                table[:, 0] = indices
                if options['TO_types'][j] == 0:
                    width = 1
                if options['TO_types'][j] == 3:
                    width = len(options['bigTs'])
                if options['TO_types'][j] == 4:
                    width = options['order_rooted_polynomial']
                    table[:, 2] = options['order_rooted_polynomial']
                G_all = np.zeros([indices.size, x.size, width])
                for k in range(G_all.shape[0]):
                    G_all[k, :, :] = assemble_G(x, table[k, :][None, :], options)
                if indices.size > 0:
                    width = G_perm.shape[1] + G_remaining_trans.shape[1] + G_all.shape[2]
                    G = np.zeros([G_all.shape[0], x.size, width])
                    G[:, :, 0:G_perm.shape[1]] = G_perm
                    G[:, :, G_perm.shape[1]:G_perm.shape[1] + G_remaining_trans.shape[1]] = G_remaining_trans
                    G[:, :, G_perm.shape[1] + G_remaining_trans.shape[1]:] = G_all
                    mf = multi_fit_predict_SNE(G, y, err, options)
                    if mf.min() < current_misfit:
                        current_misfit = mf.min()
                        trans_table[i, :] = table[mf == mf.min(), :][0, :]
            if current_misfit == misfit_start:
                no_change_count += 1
            i += 1

    ### Here getting the residual to be used in the next full gratsid iteration
    if trans_table.shape[0] > 0:
        full_table = np.vstack([perm_table, trans_table])
    else:
        full_table = perm_table.copy()
    G_full = assemble_G(x, full_table, options)
    G_full[:, 0:G_perm.shape[1]] = G_perm

    residual, residual_val, m = \
        single_fit_predict_SNE(G_full, y, err, options)

    return trans_table, residual_val, residual


def end_of_gratsid_iteration_evaluation(progress, trans_table, candidates_table, \
                                        current_misfit, options, no_improvement_count, x, y, err, sols, \
                                        perm_table, residual):
    ### Settign some value sin case we iterate again
    progress.append(trans_table)
    quarantine_list = list(np.unique(np.append(candidates_table[:, 0], trans_table[:, 0])).astype(int))
    previous_trans_table = trans_table.copy()
    previous_misfit = current_misfit.copy()

    ### Now determining convergence
    converged = False
    if no_improvement_count == options['max_no_improvement_count']:
        converged = True
    if trans_table.shape[0] >= options['max_TOs']:
        converged = True

    #### Performing tasks if converged
    if converged == True:
        sols.append(progress)
        trans_table = remove_frac_rows(trans_table, options['frac_remove'])
        table = np.vstack([perm_table, trans_table])
        G = assemble_G(x, table, options)
        residual, residual_val, m = single_fit_predict_SNE(G, y, err, options)
        previous_trans_table = trans_table.copy()
        progress = []
        no_improvement_count = 0
        quarantine_list = []
        current_misfit = residual_val.copy()
        previous_misfit = residual_val.copy()

    return progress, quarantine_list, previous_trans_table, previous_misfit, sols, \
           trans_table, current_misfit, previous_misfit, \
           no_improvement_count, converged, residual
