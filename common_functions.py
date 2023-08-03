#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:23:23 2023

@author: jon
"""

import numpy as np
import tensorflow as tf
import scipy
import itertools

def generate_options():
    """
    Creates dictionary of options for gratsid that contains 
    default parameters.
    """
    options = {}
    options['max_gap'] = 30  # this is expressed in the unit of time 
    options['bigTs'] = 10.0**np.array([-3,-2,-1])
    options['Fs'] = 1 # Sampling interval,important for defining the heavisides and fourier oscillations
    options['osc_periods'] = [365.25,365.25/2]  ## Oscillation periods
    options['polynomial_order'] = 1 ## Order of polynomial in the permanent basis functions
    options['tik_mul'] = 1e-5   ## Tikhonov weight for the Tikhonov regularization
    options['TO_types'] = [0,3] # keys correspond to which transient types are allowed: 0=steps, 3=multi-transients, 4=rooted polynomials (can be any combination of 0,3, or 4)
    options['n_search'] = 10 # number of TO onsets to try in each residual vector
    options['max_remove'] = 2 # maximum number of functions to remove during redundancy check
    options['max_no_improvement_count'] = 1 # number of times for the algo to be stuck before declaring convergence  (currently irrelevant while I develop the convergence options)
    options['frac_remove'] = 0.5 # fraction of basis functions to remove at convergence
    options['each_side'] = 3  #  During the swap-out phase, number of samples each side of original time to try shifting the basis function
    options['nsols'] = 5  ## Number of solutions (number of convergences) to run
    options['noise_flag'] = 1  # If 1, adds noise to the residual upon each new iteration (useful for getting an idea of model uncertainty if doing multiple convergences)
    options['verbose'] = 0  ## If 1, spits out some information as gratsid runs
    options['order_rooted_polynomial'] = 1 # must be greater than 0
    options['fractional_tolerance'] = 0.02  ## fraction that the solution must increase by or else convergence criterion triggered
    options['max_TOs'] = 15  ## Max number of Transient Onsets to allow in each solution
    options['res_each_side'] = 15  ## smoothing the residual with median filter with num elements each side controlled by this value.  This helps find candidate transient onsets in particularly noisy time series
    
    
    options['gradient_descent'] = False
    options['frac'] = 0.5  ## if you allow seasonal to vary by 20% from year to year, frac=0.2
    options['S_each_side'] = 90  ## later need to make 90 corresponding to a hyperparam
    options['max_its'] = 450 ## maximum # iterations for a single gradient descent
    options['chances'] = 50 ### number of iterations to wait for non-improved fit during gradient descent
    options['min_frac_improvement'] = 1e-2  ## the fractional improvement in fit needed to consider that the solution has improved (during gradient descent)
    options['lrate'] = 1e-2  ## learning rate of the optimizer used for gradient descent
    options['damp'] = 1e-9  ## damping to apply in the evaluation of cost (for both Solution of Normal Equations and for gradient descent)
    options['opt_name'] = 'Adam'  ## Future versions will allow optimizers other than Adam
    options['verbose_GD'] = False  ### Spitting out text during gradient descent
    return options

def initial_checks(x_in,y_in,err_in,options,known_steps):
    proceed_to_signal_decomposition = True
    ### Checking dimensions of y then sorting by time
    if len(y_in.shape) == 1:
        if err_in is None:
            err_in = np.nan*np.ones(y_in.shape[0])
        y_out = y_in[:,None]
        err_out = err_in[:,None]
    else:
        y_out = y_in
        if err_in is None:
            err_out = np.nan*np.ones([y_in.shape[0],y_in.shape[1]])
        else:
            err_out = err_in
    xy = np.hstack([x_in[:,None],y_out,err_out])
    xy = my_sortrows(xy,[0],[1])
    x_out = xy[:,0]
    y_out = xy[:,1:1+y_out.shape[1]]
    err_out = xy[:,1+err_out.shape[1]:]
    if len(y_out.shape) == 1:
        y_out = y_out[:,None]
        err_out = err_out[:,None]
    ndims = y_out.shape[1]
    ### Checking that there is no redundant value (in time)
    u, indices = np.unique(x_out, return_index=True)
    if indices.size < x_out.size:
        proceed_to_signal_decomposition = False
        print('Time series contains non-unique samples \n\
              as determined from the input time vector.')
    ### Checking that the max_gap before a step is invoked is appropriate
    if options['max_gap'] < np.min(np.diff(x_out)):   ## fix this later to make this a break in the program.  For now continue with boolean
        proceed_to_signal_decomposition = False
        print('Specified time gap for which to assume \n\
              a step is smaller than the smallest time-steps.\n\
              To fix this error make the "max_gap" argument larger.')
    if np.isnan(err_out).sum()>0:
        err_out = None
    
    known_steps = np.array(known_steps)
    
    return x_out,y_out,err_out,ndims,proceed_to_signal_decomposition,\
             known_steps

def my_sortrows(a,columns,asc_desc_flag):
    """
    # function aims to copy matlab function "sortrows.m"
    # 'a' must be the table (np.array) that you want to be sorted by columns
    # 'columns' must be a list
    # 'asc_desc_flag' (also a list) is the ascending or descending order for 
    #  each column: (1) ascending, (-1) descending
    """

    if len(columns) == 1:
        columns.append(columns[0])
        asc_desc_flag.append(asc_desc_flag[0])

    columns_arr = np.asarray(columns)
    asc_desc_flag_arr = np.asarray(asc_desc_flag)
    b = columns_arr[::-1]
    b_sgn = asc_desc_flag_arr[::-1]
    b = np.absolute(b)
    scols = np.array([]).reshape(a.shape[0],0)
    scols = ()
    for i in range(b.size):
        scols = scols +(b_sgn[i]*a[:,b[i]],)
    inds = np.lexsort(scols)
    out = a[inds,:]
    return out

def get_indices(arr_in):
    all_indices = np.array([])
    for i in range(len(arr_in)):
        all_indices = np.append(all_indices,arr_in[i].ravel())
    return all_indices.astype(int)

def remove_frac_rows(table,frac):
    ii = np.random.permutation(table.shape[0])
    bad_list = np.arange(np.round(frac*ii.size).astype(int))
    table = table[~np.isin(np.arange(table.shape[0]),ii[bad_list]),:]
    return table

def generate_permanent_step_list(x,known_steps,options):
    """
    Making a list of permanent steps from the list of 
    known_steps and also given the maximum allowable 
    gap size in the data beyond which a step must be 
    implemented (options['max_gap'])
    """
    perm_steps_list = []
    indices = np.arange(x.size)
    ## First dealing with known steps
    if len(np.array(known_steps).shape) == 1:
        known_steps = known_steps[None,:]
    if known_steps.size > 0:
        for i in range(known_steps.shape[0]):
            ii = indices[x==known_steps[i,0]]
            if ii.size > 0:
                perm_steps_list.append([ii[0],\
                                        0,known_steps[i,1]])
            else:
                time_before = x-known_steps[i,0]
                if (time_before[time_before<0]).size>0:
                    ii = indices[time_before<0]
                    if ii.size>0:
                        ii = ii[-1]
                    perm_steps_list.append([ii,\
                                        0,0])
    ## Next dealing with gaps that exceed the max_gap variable
    time_diff = np.diff(x)
    ii_diff = indices[0:-1]
    exceeds_gap = ii_diff[time_diff > options['max_gap']]
    
    
    for i in range(exceeds_gap.shape[0]):
        perm_steps_list.append([exceeds_gap[i],\
                                    0,0])
    return np.asarray(perm_steps_list)


def gather_perm_table(options,perm_steps_list):
    """
    Given the list of steps and assumed oscillation periods,
    listing these in a perm_table (permanent table).
    Permanent in the sense that the corresponding basis functions
    will always be in the solution.
    """
    osc_list = []
    for i in range(len(options['osc_periods'])):
        osc_list.append([0,2,options['osc_periods'][i]])
    
        
    perm_elements = [[0,1,options['polynomial_order']],osc_list,perm_steps_list]
                     
    perm_table = np.array([]).reshape(0,3)
    for i in range(len(perm_elements)):
        single_table = np.array(perm_elements[i])
        if len(single_table.shape) == 1:
            single_table = single_table[None,:]
        if np.array(single_table).size == 0:
            single_table = np.array([]).reshape(0,3)
            perm_table = np.vstack([perm_table,single_table])
        else:
            perm_table = np.vstack([perm_table,single_table])
    return perm_table

def assemble_G(x,table,options):
    """
    Given the table and time, making the basis functions
    """
    # First establishing the size of the matrix from the keys (column 2)
    width = 0
    for i in range(table.shape[0]):
        if table[i,1] == 0:
            width+=1
        if table[i,1] == 1:
            width+= table[i,2]+1
        if table[i,1] == 2:
            width+= 2
        if table[i,1] == 3:
            width+= len(options['bigTs'])
        if table[i,1] == 4:
            width+= table[i,2].astype(int)
    G = np.zeros([x.size,int(width)])
    along = 0
    for i in range(table.shape[0]):
        if table[i,1] == 0:
            width = 1
            G[:,along:along+width] = create_heav(x,table[i,:],options['Fs'])
            along+=width
        if table[i,1] == 1:
            width = table[i,2].astype(int)+1
            G[:,along:along+width] = create_polynomial(x,table[i,2])
            along+=width
        if table[i,1] == 2:
            width = 2
            G[:,along:along+width] = create_osc(x,table[i,2])
            along+=width
        if table[i,1] == 3:
            width = len(options['bigTs'])
            G[:,along:along+width] = create_mt(x,table[i,:],options['bigTs'])
            along+=width
        if table[i,1] == 4:
            width = table[i,2].astype(int)
            G[:,along:along+width] = create_rooted_polynomial(x,table[i,:],table[i,2])
            along+=width
    return G

def assemble_G_return_keys(x,table,options):
    """
    Given the table and time, making the basis functions.
    Also, returning the function type for each element of the solution
    vector.
    """
    # First establishing the size of the matrix from the keys (column 2)
    width = 0
    for i in range(table.shape[0]):
        if table[i,1] == 0:
            width+=1
        if table[i,1] == 1:
            width+= table[i,2]+1
        if table[i,1] == 2:
            width+= 2
        if table[i,1] == 3:
            width+= len(options['bigTs'])
        if table[i,1] == 4:
            width+= table[i,2].astype(int)
    G = np.zeros([x.size,int(width)])
    m_keys = np.zeros([int(width)])
    along = 0
    for i in range(table.shape[0]):
        if table[i,1] == 0:
            width = 1
            G[:,along:along+width] = create_heav(x,table[i,:],options['Fs'])
            m_keys[along:along+width] = 0
            along+=width
        if table[i,1] == 1:
            width = table[i,2].astype(int)+1
            G[:,along:along+width] = create_polynomial(x,table[i,2])
            m_keys[along:along+width] = 1
            along+=width
        if table[i,1] == 2:
            width = 2
            G[:,along:along+width] = create_osc(x,table[i,2])
            m_keys[along:along+width] = 2
            along+=width
        if table[i,1] == 3:
            width = len(options['bigTs'])
            G[:,along:along+width] = create_mt(x,table[i,:],options['bigTs'])
            m_keys[along:along+width] = 3
            along+=width
        if table[i,1] == 4:
            width = table[i,2].astype(int)
            G[:,along:along+width] = create_rooted_polynomial(x,table[i,:],table[i,2])
            m_keys[along:along+width] = 4
            along+=width
    return G, m_keys

#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,#"#,
#"#, creating basis functions (vectors) from input key (index, type, fraction)

def create_mt(x,keys,bigTs):
    out = np.zeros([x.shape[0],len(bigTs)])
    t = x-x[keys[0].astype(int)]
    for i in range(len(bigTs)):
         fill = 1-np.exp(-t[keys[0].astype(int):]*bigTs[i])
         if fill.max() > 0:
             fill = fill/fill.max()
         out[keys[0].astype(int):,i] = fill
    return out

def create_rooted_polynomial(x,keys,order):
    out = np.zeros([x.shape[0],order.astype(int)])
    t = x-x[keys[0].astype(int)]
    orders = np.arange(order.astype(int))
    for i in range(len(orders)):
        fill = (t[keys[0].astype(int):])**(orders[i]+1)
        if fill.max() > 0:
            fill = fill/fill.max()
        out[keys[0].astype(int):,i] = fill
    return out

def create_heav(x,keys,Fs):
    #print(keys)
    out = np.zeros([x.shape[0],1])
    step_vector = np.zeros([x.shape[0],1])
    #print('debug 101')
    #print(keys)
    t = x-x[keys[0].astype(int)]
    
    # Making the step vector
    if keys[0] < x.size-1:
        if t[keys[0].astype(int)+1]-t[keys[0].astype(int)] == Fs:
            step_vector[1] = 1-keys[2]
            step_vector[2:] = 1
        else:
            step_vector[1:] = 1
    else:
        None
    # Shifting the step vector to begin at index and be truncated at the end of series
    out[keys[0].astype(int):] = step_vector[0:out[keys[0].astype(int):].size]
    
    return out

def create_osc(x,period):
    t = x
    y_add_sin = np.sin(t/(period)*2*np.pi)
    y_add_cos = np.cos(t/(period)*2*np.pi)
    return np.hstack([y_add_sin[:,None],y_add_cos[:,None]])

def create_polynomial(x,order):
    t = x-x[0]
    out = np.zeros([x.size,order.astype(int)+1])
    for i in range(order.astype(int)+1):
        out[:,i] = t**i
        out[:,i] = out[:,i]/out[:,i].max()
    return out


###############################################################################
###  Performing inversions  ###################################################
###############################################################################
def expand_G_ndims(G_in,ndims):
    if ndims == 1:
        G_out = G_in
    else:
        G_out = G_in
        for i in range(ndims)[1:]:
            G_out = scipy.linalg.block_diag(G_out,G_in)
    return G_out

def expand_Gtensor_ndims(G_in,ndims):
    G_out = np.zeros([G_in.shape[0],ndims*G_in.shape[1],ndims*G_in.shape[2]])
    for i in range(ndims):
        G_out[:,i*G_in.shape[1]:(i+1)*G_in.shape[1],\
              i*G_in.shape[2]:(i+1)*G_in.shape[2]] = G_in
    return G_out

def expand_y_ndims(y_in,ndims):
    if ndims == 1:
        y_out = y_in
    else:
        y_out = y_in.T.ravel()[:,None]
    return y_out

def reduce_y_ndims(x,y,ndims):
    return y.ravel().reshape(ndims,x.size).T

def reduce_ytensor_ndims(x,y,ndims):
    return tf.transpose(y.reshape(y.shape[0],ndims,x.size),[0,2,1])

def single_fit_predict_SNE(G_in,y_in,err_in,options):
    '''
    Using tensorflow to solve a System of Normal Equations (S.N.E.)
    '''
    ndims = y_in.shape[1]
    G = expand_G_ndims(G_in,ndims)
    y = expand_y_ndims(y_in,ndims)
    if err_in is None:
        m = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G),G)+options['tik_mul']*np.eye(G.shape[1])),\
         np.transpose(G)),y)
    else:
        C_dinv = np.diag((1/err_in).T.ravel())
        m = tf.matmul(\
            tf.matmul(tf.matmul(\
            tf.linalg.inv(tf.matmul(tf.matmul(tf.transpose(G,[1,0]),C_dinv\
                                ),G)+options['tik_mul']*np.eye(G.shape[1])),tf.transpose(G,[1,0])),C_dinv),y)    
    
    residual = y-tf.matmul(G,m)
    y_pred = np.array(tf.matmul(G,m))
    residual = reduce_y_ndims(np.arange(G_in.shape[0]),np.array(residual),ndims)
    m = reduce_y_ndims(np.arange(G_in.shape[1]),np.array(m),ndims)
    if err_in is not None:
        wt_residual = np.multiply((1/err_in.ravel()),residual.ravel())
        residual_val = np.sqrt(np.sum(\
            np.square(wt_residual))) + \
            options['damp']*(np.sqrt(np.sum\
                (np.square(m.ravel()))))
    else:
        residual_val = np.sqrt(np.sum(\
            np.square(y.ravel()-y_pred.ravel()))) + \
            options['damp']*(np.sqrt(np.sum\
                (np.square(m.ravel()))))
    
    return residual,residual_val, m

def multi_fit_predict_SNE(G_in,y_in,err_in,options):
    '''
    Using tensorflow to solve a System of Normal Equations (S.N.E.)
    multiple times (with a 3D G tensor instead of G matrix).
    Returns misfit (redidual_val), for each inversion.
    '''
    ndims = y_in.shape[1]
    G = expand_Gtensor_ndims(G_in,ndims)
    y = expand_y_ndims(y_in,ndims)
    if err_in is None:
        m = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(G,[0,2,1]),G)+options['tik_mul']*np.eye(G.shape[2])),\
         tf.transpose(G,[0,2,1])),y)
    else:
        C_dinv = np.diag((1/err_in).T.ravel())
        m = tf.matmul(\
            tf.matmul(tf.matmul(\
            tf.linalg.inv(tf.matmul(tf.matmul(tf.transpose(G,[0,2,1]),C_dinv\
                                ),G)+options['tik_mul']*np.eye(G.shape[2])),tf.transpose(G,[0,2,1])),C_dinv),y)    
    
    residual = y-tf.matmul(G,m)
    #print('multi_fit_predict_SNE residual.shape:',residual.shape)
    if err_in is not None:
        err = expand_y_ndims(err_in,ndims)
        wt_residual = np.multiply(1/err.ravel()[:,None],residual)
        residual_val = np.sqrt(np.sum(\
            np.square(wt_residual),axis=1)) + \
            options['damp']*(np.sqrt(np.sum\
                (np.square(m),axis=1)))
    else:
        residual_val = np.sqrt(np.sum(\
            np.square(residual),axis=1)) + \
            options['damp']*(np.sqrt(np.sum\
                (np.square(m),axis=1)))
    
    #print('multi_fit_predict_SNE residual_val.shape:',residual_val.shape)    
    residual_val = residual_val.ravel()
        
    return residual_val

############################################################################
### Finding transient onsets in residual
############################################################################
    
def find_TO(y,quarantined_list,options):
    n_search = options['n_search']
    res_each_side = options['res_each_side']
    
    ## for each component of the input time series, finding TOs
    TO_all_components = []
    for i in range(y.shape[1]):
        TO_all_components.append(list(find_TO_locations(y[:,i],quarantined_list,n_search,res_each_side)))
    ## Returning a list of the unique TOs from list of each component
    do_loop = any(isinstance(i, list) for i in TO_all_components)
    if do_loop == True:
        new_list = []
        for i in range(len(TO_all_components)):
            new_list += TO_all_components[i]
    else:
        new_list = TO_all_components
    return list(np.unique(new_list))

def find_TO_locations(y,quarantined_list,n_search,res_each_side):    
    y_smooth = np.zeros(y.size)
    for i in range(y.size):
        pp = np.arange((i-res_each_side),(i+res_each_side+1),1)
        pp = pp[pp>0]
        pp = pp[pp<y.size]
        
        y_smooth[i] = np.nanmedian(\
                        y[pp],\
                        axis=0)
    
    
    available = np.arange(y.size)
    i_mf = np.hstack([available[:,None],np.abs(y_smooth[:,None]),y_smooth[:,None]])
    i_mf = my_sortrows(i_mf,[1],[-1])
    i_mf = i_mf[~np.isin(i_mf[:,0],quarantined_list),:]
    available = available[~np.isin(available,quarantined_list)]
    TO_list = []
    continue_loop = True
    while continue_loop == True:
        ## Reordering the matrix by ascending absolute misfit and getting the indec of highest misfit
        i_mf = my_sortrows(i_mf,[1],[-1])
        ii = i_mf[0,0].astype(int)
        ## According to the sign of this peak misfit, finding the indices  
        ## that the zero crossings occur either side of this peak (bb,aa).
        sgn_peak = np.sign(i_mf[0,2])  # getting sign of the peak residual
        bb = available[available<ii][np.sign(y_smooth[available][available<ii])==-1*np.sign(sgn_peak)]  # finding zero crossing before peak
        if bb.size>0:
            bb = bb[-1]
        aa = available[available>ii][np.sign(y_smooth[available][available>ii])==-1*np.sign(sgn_peak)] # finding zero crossing after peak 
        if aa.size>0:
            aa = aa[0]
        ## Now having established zero crossings, outputting TOs and eliminating available points in time series    
        if (bb.size==1)*(aa.size==1):
            TO_list.append(i_mf[0,0].astype(int))
            TO_list.append(bb)
            TO_list.append(aa)
            i_mf = i_mf[((i_mf[:,0]<bb)+(i_mf[:,0]>aa))>0,:]
            available = available[((available<bb)+(available>aa))>0]
        if (bb.size==1)*(aa.size==0):
            TO_list.append(i_mf[0,0].astype(int))
            TO_list.append(bb)
            #TO_list.append(aa) # for debugging only
            i_mf = i_mf[(i_mf[:,0]<bb)>0,:]
            available = available[(available<bb)>0]
        if (bb.size==0)*(aa.size==1):
            TO_list.append(i_mf[0,0].astype(int))
            #TO_list.append(bb) # for debugging only
            TO_list.append(aa)
            i_mf = i_mf[(i_mf[:,0]>aa)>0,:]
            available = available[(available>aa)>0]
        if (bb.size==0)*(aa.size==0):
            continue_loop = False
        if (available.size>2)*(len(TO_list)<n_search)==0:
            continue_loop = False
    return TO_list

def make_TO_G_lists(options,x,TO_list):
    TO_by_type = []
    for i in range(len(options['TO_types'])):
        a = []
        for j in range(len(TO_list)):
            if options['TO_types'][i] == 4:
                a.append(assemble_G(x,np.array([TO_list[j],\
                    options['TO_types'][i],options['order_rooted_polynomial']])[None,:],\
                                options))
            else:
                a.append(assemble_G(x,np.array([TO_list[j],options['TO_types'][i],0])[None,:],\
                                options))
        TO_by_type.append(a)
    return TO_by_type

def evaluate_candidate_TOs(TO_list,G_TOs,x,residual,err,perm_steps_list,options):
    if options['noise_flag'] == 1:
        noise_matrix = np.random.randn(residual.shape[0],residual.shape[1])
        for i in range(noise_matrix.shape[1]):
            noise_matrix[:,i] = noise_matrix[:,i]*np.std(residual[:,i]) # multiplying by the std of the current residual
        residual = residual+noise_matrix
    
    best_misfit=np.inf # initial bets misifit is infinity, important for logic below
    candidates_table = np.zeros([2,3])
    TO_type_combos = list(itertools.permutations(np.arange(len(options['TO_types'])),2))
    TO_combos = np.array(list(itertools.combinations(np.arange(len(TO_list)),2)))
    for i in range(len(options['TO_types'])):
        TO_type_combos.append((i,i))
    for i in range(len(TO_type_combos))[:]:
        type_A = options['TO_types'][TO_type_combos[i][0]]
        type_B = options['TO_types'][TO_type_combos[i][1]]
        if type_A == 0:
            width_A = 1
            TO_combos_sub = TO_combos[~np.isin(TO_combos[:,0],perm_steps_list[:,0]),:] ## making sure that a step can't start where one has already been pre-specified
        if type_A == 3:
            width_A = len(options['bigTs'])
            TO_combos_sub = TO_combos.copy()
        if type_A == 4:
            width_A = options['order_rooted_polynomial']
            TO_combos_sub = TO_combos.copy()
        if type_B == 0:
            width_B = 1
            TO_combos_sub = TO_combos[~np.isin(TO_combos[:,1],perm_steps_list[:,0]),:] ## making sure that a step can't start where one has already been pre-specified
        if type_B == 3:
            width_B = len(options['bigTs'])
            TO_combos_sub = TO_combos.copy()
        if type_B == 4:
            width_B = options['order_rooted_polynomial']
            TO_combos_sub = TO_combos.copy()
        
        G = np.zeros([len(TO_combos_sub),x.size,width_A+width_B])
        G[:,:,0:width_A] = np.array(G_TOs[TO_type_combos[i][0]])[TO_combos_sub[:,0],:,:]
        G[:,:,width_A:width_A+width_B] = np.array(G_TOs[TO_type_combos[i][1]])[TO_combos_sub[:,1],:,:]
        misfits = multi_fit_predict_SNE(G_in=G,y_in=residual,err_in=err,options=options)
        
        if misfits.min() < best_misfit:
            candidates_table[:,0] = np.array(TO_list)[TO_combos_sub[misfits==misfits.min(),:]][0,:]
            candidates_table[:,1] = np.array(options['TO_types'])[np.array(TO_type_combos[i])]
            candidates_table[candidates_table[:,1]==4,2] = options['order_rooted_polynomial']
            best_misfit = misfits.min()
            #print('Updating best misfit:',best_misfit)
    return candidates_table







