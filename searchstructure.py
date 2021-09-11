#The necessary imports
import os
import sys
import copy
import math
import numpy as np
from classes import *
from seq_tests import *
from con_tests import *
from ele_tests import *

"""
The Helpers needed for the algorithm
They consist of functions that extract subseries, positions of series which are numbers etc.
"""
def extract_subseries(series, start_index = 0, stepsize = 1):
    """
    extract_subseries(series, start_index = 0, stepsize = 1)
    Subsamples a given series.
    Outputs is a subseries and the remaining series.
    E.g. every second element of a given series and the remaining series.
    """
    samples       = np.arange(start_index, series.size, stepsize).astype('int')
    all_positions = np.arange(series.size).astype('int')
    not_samples   = np.setdiff1d(all_positions, samples)
    return series[samples],series[not_samples]

def extract_subseries_cons_elem(series, start_index = 0, cons_elem_len = 1, stepsize = 1):
    """
    extract_subseries_cons_elem(series, start_index = 0, cons_elem_len = 1, stepsize = 1)    +1   *2   +1   *2
    Subsamples a given series. Such that one can check for repeating blocks of rules, i.e. a -> b -> c -> d -> e
    Outputs is a set of indices and the remaining indices.
    The remaining indices repeat the indices at the border, in the example up top this would be:
    set of indices = [[0,1],[2,3]]; remaining indices = [[1,2][3,4]]
    Observe, that the elements in set of indices all have the same length.
    Same thing for the elements in remining indices. This means, that the end of the series is disregarded
    if it can't be subsampled adequatly.
    """
    samples_start = np.arange(start_index, series.size, stepsize).astype('int')
    samples       = np.asarray([np.arange(pos,pos+cons_elem_len).astype('int')
                                for pos in samples_start if pos+cons_elem_len<=series.size]
                               ,dtype=object).astype('int')
    remaining     = np.asarray([np.arange(pos+cons_elem_len-1,pos+stepsize+1).astype('int')
                                for pos in samples_start 
                                if (pos + cons_elem_len <= series.size-1) & (pos + stepsize <= series.size-1)]
                               ,dtype=object).astype('int')
    return samples, remaining

"""
In the following the structure of the searchspace is defined.
If one wants another structure please feel free to alter the following functions.
"""

"""
Defines the structure between 'layers' of the searchspace
"""
def gen_next(candidates):
    """
    gen_next(candidates):
    Generates the next layer (localy) belonging to each candidate.
    returns all of these seies.
    """
    next_candidates = np.asarray([], dtype = 'object')
    for candidate in candidates:
        series_dif, series_quo      = gen_next_layer(candidate.series)
        candidate_series_dif        = data(candidate.pos, candidate.series, None, candidate, 'dif')
        candidate_series_dif.series = series_dif
        candidate_series_dif.pos    = np.arange(candidate_series_dif.series.size)[np.isnan(series_dif)]
        next_candidates             = np.append(next_candidates, candidate_series_dif)
        if series_quo.size != 0:
            candidate_series_quo        = data(candidate.pos, candidate.series, None, candidate, 'quo')
            candidate_series_quo.series = series_quo
            candidate_series_quo.pos    = np.arange(candidate_series_quo.series.size)[np.isnan(series_quo)]
            next_candidates             = np.append(next_candidates, candidate_series_quo)
    return next_candidates
    
"""
Generate the next series which define the next layer of the BFS search-structure
"""
def gen_next_layer(series):
    """
    gen_next_layer(series):
    generates new series, which store:
        (i)  the absoloute difference between consecutive elements
        (ii) the quotient between consecutive elements
    """
    series_dif = series[1:]-series[:-1]
    if np.all(series != 0.):
        series_quo = series[1:]/series[:-1]
    else: series_quo = np.asarray([])
    return series_dif, series_quo

"""
Defines the structure of each 'layer' in the searchspace 
"""
def perf_layer(candidates, tests, cons_tests, elem_tests, mode, database):
    """
    perf_layer(operating_series, tests, elem_tests, mode, database)
    Input: A list of object(s) from class data,
           a list of tests to perform on subseries (compare `perform_tests(series, mode)´),
           a list of tests, regarding consecutive Elements, to perform (compare `perform_cons_tests(...)´),
           a list of tests, regarding the properties of the Elements, to perform (compare `perform_element_tests(...)´),
           mode denotes wether or not the given database is online or not.
    Performs a series of checks to fill the given series correctly.
    """
    #preliminaries
    states         = -1
    candidates_new = candidates
    for object_ in candidates:
        states = np.append(states, object_.pos.size)
    states = states[1:]
    #performs the elementwise tests
    for dat_object in candidates_new:
        new_rule = False
        obj = copy.deepcopy(dat_object)
        series = obj.series
        upper_bound = int((2*series.size-1)/4.)+1
        for test in elem_tests:
            i=0
            for i in range(1,upper_bound):
                j=0
                for j in np.arange(i):
                    thesis, rest = extract_subseries(series, start_index = j, stepsize = i)
                    test_result = perform_element_tests(-thesis, mode = test)
                    validation_ = test_result[(np.squeeze(np.nonzero(np.logical_not(np.isnan(thesis)))))]
                    if validation_.size >= 2:
                        if all(validation_):
                            fill_ele_test_space(obj, -thesis, j, i, mode = test, sign = '-')
                            new_rule = True
                            series = obj.series
                    else:
                        if validation_:
                            fill_ele_test_space(obj, -thesis, j, i, mode = test, sign = '-')
                            new_rule = True
                            series = obj.series
                    test_result  = perform_element_tests(thesis, mode = test)
                    validation   = test_result[(np.squeeze(np.nonzero(np.logical_not(np.isnan(thesis)))))]
                    if validation.size >= 2:
                        if all(validation):
                            fill_ele_test_space(obj, thesis, j, i, mode = test)
                            new_rule = True
                            series = obj.series
                    else:
                        if validation:
                            fill_ele_test_space(obj, thesis, j, i, mode = test)
                            new_rule = True
                            series = obj.series
                    if dat_object.pos.size == 0:
                        break
                if dat_object.pos.size == 0:
                    break
            if dat_object.pos.size == 0:
                break
        if new_rule:
            #save the new result(s)
            candidates_new = np.append(candidates_new, obj)
            states         = np.append(states, obj.pos.size)
    #performs the tests on subseries
    for dat_object in candidates_new:
        new_rule    = False
        obj         = copy.deepcopy(dat_object)
        series      = obj.series
        upper_bound = int((2*series.size-1)/4.)+1
        if upper_bound >=2: #There are no usefull Rules for masked series of length 1 or 2
            for test in tests:
                i=0
                for i in range(1,upper_bound):
                    j=0
                    for j in np.arange(i):
                        thesis, rest = extract_subseries(series, start_index = j, stepsize = i)
                        test_result = perform_tests(thesis, mode = test)
                        if test_result[0]:
                            new_rule = True
                            fill_tests(obj, thesis, j, i, test_result[1], mode = test)
                            series = obj.series
                        if dat_object.pos.size == 0:
                            break
                    if dat_object.pos.size == 0:
                        break
                if dat_object.pos.size == 0:
                    break
        if new_rule:
            #save the new result(s)
            candidates_new = np.append(candidates_new, obj)
            states         = np.append(states, obj.pos.size)
    #performs the tests on consecutive Elements
    for dat_object in candidates_new:
        new_rule    = False
        obj         = copy.deepcopy(dat_object)
        series      = obj.series
        upper_bound = int((2*series.size-1)/4.)+1
        if upper_bound >=2: #There are no usefull Rules for masked series of length 1 or 2
            for test in cons_tests:
                i=0
                for i in range(1,upper_bound):
                    k=0
                    for k in range(1,upper_bound):
                        j=0
                        for j in np.arange(i):
                            thesis, rest = extract_subseries_cons_elem(series, start_index=j, cons_elem_len=k, stepsize=i)
                            test_result = perform_cons_tests(series, thesis, mode = test)
                            if test_result[0]:
                                new_rule = True
                                fill_cons_tests(obj, thesis, j, k, i, test_result[1], mode = test)
                                series = obj.series
                            if dat_object.pos.size == 0:
                                break
                        if dat_object.pos.size == 0:
                            break
                    if dat_object.pos.size == 0:
                        break
                if dat_object.pos.size == 0:
                    break
        if new_rule:
            #save the new result(s)
            candidates_new = np.append(candidates_new, obj)
            states         = np.append(states, obj.pos.size)
    return candidates_new, states

"""
Performs BFS on the Searchspace defined through gen_next() and perf_layer().
"""
def bfs(data_object, tests = ['const','sum','fac','fib'], cons_tests = ['const','sum','fac'], elem_tests = ['prime','cube','square'], mode = 'offline', database = None, verbose = True):
    """
    performs bfs in the following searchtree:
    
                    start
          a     -     b     -     c
      a - b - c   a - b - c   a - b - c
     abc abc abc abc abc abc abc abc abc
     
     in a te elementwise tests are performed
     in b the tests on subseries are performed
     in c the tests on consecutive elements are performed
     b considers the positive tests from a
     c considers the positive tests from a and b
     
     returns all possible soloutions given the series, the tests above and the way we traverse the searchspace.
     """
    if verbose:
        operating_object        = data_object
        print("Entering Layer 0")
        candidates_L0, state_L0 = perf_layer(np.asarray([operating_object]), tests, cons_tests, elem_tests, mode, database)
        if(state_L0[state_L0 == 0].size == 0):
            candidates_L0           = gen_next(candidates_L0)
            print("Entering Layer 1")
            candidates_L1, state_L1 = perf_layer(candidates_L0, tests, cons_tests, elem_tests, mode, database)
            if(state_L1[state_L1 == 0].size == 0):
                candidates_L1           = gen_next(candidates_L1)
                print("Entering Layer 2")
                candidates_L2, state_L2 = perf_layer(candidates_L1, tests, cons_tests, elem_tests, mode, database)
                if(state_L2[state_L2 == 0].size == 0):
                    print("No soloution found, try other search structure or extend the tests")
                else: return candidates_L2[state_L2 == 0]
            else: return candidates_L1[state_L1 == 0]
        else: return candidates_L0[state_L0 == 0]
    else:
        operating_object        = data_object
        candidates_L0, state_L0 = perf_layer(np.asarray([operating_object]), tests, cons_tests, elem_tests, mode, database)
        if(state_L0[state_L0 == 0].size == 0):
            candidates_L0           = gen_next(candidates_L0)
            candidates_L1, state_L1 = perf_layer(candidates_L0, tests, cons_tests, elem_tests, mode, database)
            if(state_L1[state_L1 == 0].size == 0):
                candidates_L1           = gen_next(candidates_L1)
                candidates_L2, state_L2 = perf_layer(candidates_L1, tests, cons_tests, elem_tests, mode, database)
                if(state_L2[state_L2 == 0].size == 0):
                    return 0
                else: return candidates_L2[state_L2 == 0]
            else: return candidates_L1[state_L1 == 0]
        else: return candidates_L0[state_L0 == 0]
    pass