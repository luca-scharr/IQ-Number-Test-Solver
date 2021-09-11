import copy
import functools
import random
import math
import numpy as np
import numba as nb

"""
get series at indices where series is not np.nan
"""
def series_non_nan_cons_elem_test(series, mat_indices):
    """
    series_non_nan_cons_elem_test(series, mat_indices):
    returns series[mat_indices[i]] where it's not np.nan 
    """
    series_pos = np.asarray([series[ind] for ind in mat_indices[0]])
    series_pos_non_nan = np.asarray([series_pos[row][np.logical_not(np.isnan(series_pos[row]))!=0]
                            for row in np.arange(series_pos.shape[0])],dtype=object)
    return series_pos_non_nan

"""
constant test for series of consecutive elements 
"""
def const_cons_elem_test(series, mat_indices):
    """
    const_cons_elem_test(series, mat_indices):
    returns the test result for const_test for each block if all tests were positive. (all(const_tests))
    """
    if(series[np.isnan(series)].size != 0):
        ser_non_nan      = series_non_nan_cons_elem_test(series, mat_indices)
        len_ser_non_nan  = ser_non_nan.shape[0]
        bool_vec_val     = np.asarray([const_test(ser_non_nan[i]) 
                                       for i in np.arange(len_ser_non_nan) if ser_non_nan[i].size >=2])
        if bool_vec_val.size >= 1:
            if all(bool_vec_val.T[0,:]):
                return True, bool_vec_val.T[1:,:]
            return False, np.nan
    return False, np.nan

"""
sum test for series of consecutive elements 
"""
def sum_cons_elem_test(series, mat_indices):
    """
    sum_cons_elem_test(series, mat_indices):
    returns the boolean value of the sum_test for the consecutive elements (all(sum_tests)),
    as well as the difference and which block was tested.
    """
    if(series[np.isnan(series)].size != 0):
        ser_non_nan      = series_non_nan_cons_elem_test(series, mat_indices)
        len_ser_non_nan  = ser_non_nan.shape[0]
        bool_vec_val     = np.asarray([sum_test(ser_non_nan[i])+(i,)
                                       for i in np.arange(len_ser_non_nan) if ser_non_nan[i].size >=3])
        if bool_vec_val.size >= 1:
            if all(bool_vec_val.T[0,:])&(bool_vec_val.size>=2):
                return True, bool_vec_val.T[1:,:]
            return False, bool_vec_val.T[1:,:]
    return False, np.nan

"""
fac test for series of consecutive elements 
"""
def fac_cons_elem_test(series, mat_indices):
    """
    fac_cons_elem_test(series, mat_indices):
    returns the boolean value of the fac_test for the consecutive elements (all(fac_tests)),
    as well as the factor and which block block was tested.
    """
    if(series[np.isnan(series)].size != 0):
        ser_non_nan      = series_non_nan_cons_elem_test(series, mat_indices)
        len_ser_non_nan  = ser_non_nan.shape[0]
        ser_non_nan      = np.asarray([ser_non_nan[i][ser_non_nan[i] != 0] for i in np.arange(len_ser_non_nan) 
                                       if (ser_non_nan[i][ser_non_nan[i] != 0]).size > 0])
        len_ser_non_nan  = ser_non_nan.shape[0]
        bool_vec_val     = np.asarray([fac_test(ser_non_nan[i])+(i,)
                                       for i in np.arange(len_ser_non_nan) if ser_non_nan[i].size >=3])
        if bool_vec_val.size >= 1:
            if all(bool_vec_val.T[0,:])&(bool_vec_val.size>=2):
                return True, bool_vec_val.T[1:,:]
            return False, bool_vec_val.T[1:,:]
    return False, np.nan


"""
Performs a set of predefined Tests regarding the structure of consecutive elements
"""
def perform_cons_tests(series, thesis, mode = 'const'):
    """
    perform_cons_ptests(series, thesis, mode = 'const'):
    Tests the named test on a given series, if the test is known.
    """
    if mode == 'const':
        return const_cons_elem_test(series, thesis)
    elif mode == 'sum':
        return sum_cons_elem_test(series, thesis)
    elif mode == 'fac':
        return fac_cons_elem_test(series, thesis)
    else:
        print('ERROR: This mode doesn\'t exist yet.')
        return False, None
    pass

"""
Fills the result of the tests given to `perform_cons_tests´ into the data object
"""
def fill_cons_tests(data_object, thesis, start_index, cons_elem_len, step_size, value, mode = 'const'):
    """
    fill_cons_tests(data_object, thesis, start_index, cons_elem_len, step_size, value, mode = 'const'):
    Fills the series with the corresponding values, according to the result of the corresponding test.
    """
    j = start_index
    i = step_size
    series = data_object.series
    if mode == 'const':
        mode    = 'cons_const'
        counter = 0
        for block in thesis:
            positions = block
            values    = np.full(positions.shape, value[counter])
            if positions.ndim != 0:
                data_object.fill_in(positions,values)
            else:
                data_object.fill_in(np.asarray([positions]),np.asarray([values]))
            counter += 1
        data_object.positive_tests.append((mode, start_index, cons_elem_len, step_size, value))
        return
    elif mode == 'sum':
        mode    = 'cons_sum'
        counter = 0
        for block in thesis:
            positions = block
            if np.nonzero(np.logical_not(np.isnan(series[block])))[0].ndim != 0:
                index = j+i*np.nonzero(np.logical_not(np.isnan(series[block])))[0][0]
            else:
                index = j+i*np.nonzero(np.logical_not(np.isnan(series[block])))[0]
            values = series[index]+((positions-index)/i)*value
            if positions.ndim != 0:
                data_object.fill_in(positions,values)
            else:
                data_object.fill_in(np.asarray([positions]),np.asarray([values]))
            counter += 1
        data_object.positive_tests.append((mode, start_index, cons_elem_len, step_size, value))
        return
    elif mode == 'fac':
        mode    = 'cons_fac'
        counter = 0
        for block in thesis:
            positions = block
            indices = (np.nonzero(np.logical_not(np.isnan(series[block])))[0])
            indices = indices[(np.nonzero(series[block][indices]!=0.)[0])]
            if indices.ndim != 0:
                index = j+i*indices[0]
            else:
                index = j+i*indices
            values = (series[index])*(value**((positions-index)/i))
            if positions.ndim != 0:
                data_object.fill_in(positions,values)
            else:
                data_object.fill_in(np.asarray([positions]),np.asarray([values]))
            counter += 1
        data_object.positive_tests.append((mode, start_index, step_size, value))    
        return
    else:
        print('ERROR: This mode doesn\'t exist yet.')
        return False, None
    pass