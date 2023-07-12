#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:23:08 2023

@author: jon
"""

from common_functions import *

def tf_flatten_eager_tensor(ET):
    return tf.reshape(ET,np.prod(tf.shape(ET)))

def my_temporal_smoothing(time_series_length,each_side):
    """
    If allowing oscillations to vary in apmplitude between cycles,
    there is a smoothness imposed on this variation.  This function
    creates a matrix to impose this smoothness
    """
    L = np.diag(np.ones(time_series_length))
    for i in range(each_side):
        j = i+1
        v = np.ones(L.shape[0]-j)
        L += np.diag(v,j)
        L += np.diag(v,-j)
    denom = np.repeat(np.sum(L,axis=1)[:,None],L.shape[1],axis=1)
    L = np.divide(L,denom)
    
    return L



def distort_osc_G(G,W,options):
    """
    Given G, the locations of the oscillatory basis functions in G,
    the distortion weighting matrix, W, and the corresponding options,
    we output G_new which is G with the non-distorted oscillatory functions
    replaced by distorted oscillatory functions
    """
    G_fo = G[:,options['osc_cols']]
    lb = 1
    ub = 1+options['frac'] ## if you allow seasonal to vary by 20% from year to year, frac=0.2
    G_fo_distorted = np.multiply(\
        np.matmul(options['S'],\
        0.5*(lb+ub)+0.5*(ub-lb)*np.sin(np.array(W))),\
                G_fo)
    G_new = G.copy()
    G_new[:,options['osc_cols']] = np.array(G_fo_distorted)
    return G_new


def single_fit_predict_GD(G,y,err,options):    
    """
    Using autograd, performing gradient descent
    to arrive at a converged fit.
    """
    ###########################################################################
    ### Dealing with input arguments so they can be used in the cost function
    ###########################################################################
    lb = 1
    ub = 1+options['frac'] ## if you allow seasonal to vary by 20% from year to year, frac=0.2
    non_osc_cols = ~np.isin(np.arange(G.shape[1]),options['osc_cols'])
    #print('non_osc_cols',non_osc_cols)
    G_non = G[:,non_osc_cols]
    G_fo = G[:,options['osc_cols']]
    
    
    ###########\################################################################
    ## Getting some a-priori values for m_non and m_fo
    r1,mf1,m_apriori = single_fit_predict_SNE(\
           G_in=G,y_in=np.array(y),err_in=err,options=options)
    
    m_fo_apriori = m_apriori[options['osc_cols'],:]/(0.5*(lb+ub))
    m_non_apriori = m_apriori[non_osc_cols,:]
    
    #print('m_fo_apriori.shape:',m_fo_apriori.shape)
    #print('m_non_apriori.shape:',m_non_apriori.shape)
    
    
    
    #### Initializing the variables and tensors in the way required by 
    #### tensorflow.
    f1632 = 32########## Determining if we use floating point 16 or 32
    if f1632 == 32:
        G_non = tf.convert_to_tensor(G_non.astype('float32'))
        G_fo = tf.convert_to_tensor(G_fo.astype('float32'))
        S = tf.convert_to_tensor(options['S'].astype('float32'))
        y = tf.convert_to_tensor(y.astype('float32'))
        if err is not None:
            err = tf.convert_to_tensor(err.astype('float32'))
        
        #m_non = np.random.randn(G_non.shape[1],y.shape[1])
        m_non = m_non_apriori
        m_non = tf.convert_to_tensor(m_non.astype('float32'))
        m_non = tf.Variable(m_non)
        
        #m_fo = np.random.randn(G_fo.shape[1],y.shape[1])
        m_fo = m_fo_apriori.copy()
        m_fo = tf.convert_to_tensor(m_fo.astype('float32'))
        m_fo = tf.Variable(m_fo)
        
        #W = np.random.randn(G_fo.shape[0],G_fo.shape[1])
        W = np.pi*np.ones([G_fo.shape[0],G_fo.shape[1]])
        W = tf.convert_to_tensor(W.astype('float32'))
        W = tf.Variable(W)
        W = tf.Variable(options['W'])
    
    def cost_for_descent():
        y_pred = tf.matmul(G_non,m_non) + \
            tf.matmul(tf.multiply(\
                tf.matmul(S,0.5*(lb+ub)+0.5*(ub-lb)*tf.sin(W)),\
                G_fo),m_fo)
        if err is not None: 
            out = tf.sqrt(tf.reduce_sum(\
                tf.square(tf.multiply((1/tf_flatten_eager_tensor(err)),\
                tf_flatten_eager_tensor(y)-tf_flatten_eager_tensor(y_pred))))) + \
                options['damp']*(tf.sqrt(tf.reduce_sum\
                    (tf.square(tf.cast(tf_flatten_eager_tensor(m_non),'float32'))))) + \
                options['damp']*(tf.sqrt(tf.reduce_sum\
                    (tf.square(tf.cast(tf_flatten_eager_tensor(m_fo),'float32')))))
        else:
            out = tf.sqrt(tf.reduce_sum(\
                tf.square(tf.cast(tf_flatten_eager_tensor(y)-\
                    tf_flatten_eager_tensor(y_pred),'float32')))) + \
                options['damp']*(tf.sqrt(tf.reduce_sum\
                (tf.square(tf.cast(tf_flatten_eager_tensor(m_non),'float32'))))) + \
                options['damp']*(tf.sqrt(tf.reduce_sum\
                (tf.square(tf.cast(tf_flatten_eager_tensor(m_fo),'float32')))))
        return out

    def cost(m_non,m_fo,W):
        y_pred = tf.matmul(G_non,m_non) + \
            tf.matmul(tf.multiply(\
                tf.matmul(S,0.5*(lb+ub)+0.5*(ub-lb)*tf.sin(W)),\
                G_fo),m_fo)
        if err is not None: 
            out = tf.sqrt(tf.reduce_sum(\
                tf.square(tf.multiply((1/tf_flatten_eager_tensor(err)),\
                tf_flatten_eager_tensor(y)-tf_flatten_eager_tensor(y_pred))))) + \
                options['damp']*(tf.sqrt(tf.reduce_sum\
                    (tf.square(tf.cast(tf_flatten_eager_tensor(m_non),'float32'))))) + \
                options['damp']*(tf.sqrt(tf.reduce_sum\
                    (tf.square(tf.cast(tf_flatten_eager_tensor(m_fo),'float32')))))
        else:
            out = tf.sqrt(tf.reduce_sum(\
                tf.square(tf.cast(tf_flatten_eager_tensor(y)-\
                    tf_flatten_eager_tensor(y_pred),'float32')))) + \
                options['damp']*(tf.sqrt(tf.reduce_sum(tf.square(tf.cast(tf_flatten_eager_tensor(m_non),'float32'))))) + \
                options['damp']*(tf.sqrt(tf.reduce_sum\
                (tf.square(tf.cast(tf_flatten_eager_tensor(m_fo),'float32')))))
        
        return out
    
    ### Setting the learning rate
    options['opt'].learning_rate = options['lrate'] # setting the learning rate
    
    ### Resetting the optimizer variables!!! VERY IMPORTANT TO DO BETWEEN SEPARATE OPTIMIZATIONS
    for var in options['opt'].variables():
        var.assign(tf.zeros_like(var))
    
    #### This loops through the gradient descent until convergence.
    i = 0
    current_lowest_cost = cost(m_non,m_fo,W).numpy()
    m_non_best, m_fo_best, W_best = \
        m_non.numpy(), m_fo.numpy(), W.numpy()
    
    carry_on = 1
    no_improvement_count = 0
    while (i < options['max_its'])*(carry_on==1)==True:
        i+=1
        options['opt'].minimize(cost_for_descent, var_list=[m_non,m_fo,W])
        new_cost = cost(m_non,m_fo,W).numpy()
        frac_improvement = 1-(new_cost/current_lowest_cost)
        
        #if options['verbose_GD'] == True:
        #    print(i, frac_improvement,\
        #        (frac_improvement > options['min_frac_improvement']), new_cost, current_lowest_cost)
        
        if (frac_improvement < options['min_frac_improvement'])+\
            (new_cost>current_lowest_cost) > 0:
            no_improvement_count+=1
        if (new_cost < current_lowest_cost)*(frac_improvement > options['min_frac_improvement'])==True:
            current_lowest_cost = new_cost
            m_non_best, m_fo_best, W_best = \
                m_non.numpy(), m_fo.numpy(), W.numpy()
            no_improvement_count = 0
        if no_improvement_count >= options['chances']:
            carry_on = 0
    
    #### Now distort the oscillatory functions of G using W_best
    G_new = distort_osc_G(G,W_best,options)
    if err is not None:
        err_in = np.array(err)
    else:
        err_in = None
    residual,misfit,m = single_fit_predict_SNE(\
     G_in=G_new,y_in=np.array(y),err_in=err_in,options=options)
    
    if options['verbose_GD'] == True:
        print('#_its, misfit, GD:SNE: ',i,misfit,current_lowest_cost/misfit)
        
    return residual, misfit, m, W_best  ## finally, using the System of Normal Equations (SNE) to do the inversion with the distorted seasonal


def multi_fit_predict_GD(G_all,y,err,options):
    """
    Doing the gradient descent in a loop if there are many sets of 
    Green's Functions to test.  Outputting also the solved time weightings
    with W_fill
    """
    residual_vals = []
    if len(np.shape(G_all))<3:
        G_new = np.zeros([1,G_all.shape[0],G_all.shape[1]])
        G_new[0] = G_all.copy()
    else:
        G_new = G_all.copy()
    
    
    W_fill = np.zeros([G_new.shape[0],G_new.shape[1],\
                      2*len(options['osc_periods'])])
    
    for i in range(G_new.shape[0]):
        G_in = G_new[i]
        residual, current_lowest_cost, m, W_best = \
        single_fit_predict_GD(G_in,y,err,options)
        residual_vals.append(current_lowest_cost.copy())
        W_fill[i] = np.array(W_best)
        
    residual_vals = np.array(residual_vals)
    
    return residual_vals, W_fill


