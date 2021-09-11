import copy
import functools
import random
import math
import numpy as np
import numba as nb

"""
test for constant value of a (sub)series
"""
def const_test(series):
    """
    const_test(series):
    checks if all non np.nan elements in the series have the same constant value.
    returns the boolean value of this check and value checked against.
    """
    if(series[np.isnan(series)].size != 0):
        indices = np.nonzero(np.logical_not(np.isnan(series)))[0]
        if indices.size <2:
            return False, np.nan
        value = series[indices[0]]
        if indices.size ==2:
            if math.isclose(series[indices[0]],series[indices[1]]):
                return True, value
        if np.allclose(series[indices],np.full(series[indices].shape, value)):
            return True, value
        return False, value
    else: return False, np.nan
    return False, np.nan

"""
test for constant summand between the elements of a (sub)series
"""
def sum_test(series):
    """
    sum_test(series):
    checks if all non np.nan elements in the series can be obtained by adding a constant value.
    returns the boolean value of this check and the distance checked against.
    """
    if(series[np.isnan(series)].size != 0):
        series = copy.deepcopy(series).astype('float64')
        indices = np.nonzero(np.logical_not(np.isnan(series)))[0]
        if indices.size <3:
            return False, np.nan
        differences_indices = indices[1:]-indices[0:-1]
        #Check the distance of first two non nan items and the distance of their indices. 
        #Compare if difference between non nan items is:
        #(distance of non nan)*(distance of first two)/(distance of first two)
        #If yes, then found a subseries of constant spacing
        difference_consecutive_elements = (series[indices[1]]-series[indices[0]])/differences_indices[0]
        if np.allclose((difference_consecutive_elements*differences_indices), series[indices[1:]]-series[indices[0:-1]]):
            return True, difference_consecutive_elements
        return False, difference_consecutive_elements
    else: return False, np.nan
    return False, np.nan

"""
test for constant factor between the elements of a (sub)series
"""
def fac_test(series):
    """
    fac_test(series):
    checks if all non np.nan elements in the series can be obtained by multiplying a constant factor.
    returns the boolean value of this check and the distance checked against.
    """
    if(series[np.isnan(series)].size != 0):
        series = copy.deepcopy(series).astype('float64')
        indices = (np.nonzero(np.logical_not(np.isnan(series)))[0])
        indices = indices[(np.nonzero(series[indices]!=0.)[0])]
        if indices.size <3:
            return False, np.nan
        differences_indices = indices[1:]-indices[0:-1]
        #print("differences_indices",differences_indices)
        #Check the distance of first two non nan items and the relative distance of their indices. 
        #Check if quotient between non nan items is equal to 
        #((quotient of first two)^{1/(distance of first two)})^(distance of non nan)
        #If yes, then found a subseries of constant spacing
        quotient_consecutive_elements = (series[indices[1]]/series[indices[0]]) ** (1./differences_indices[0])
        if np.allclose((quotient_consecutive_elements ** differences_indices), series[indices[1:]]/series[indices[0:-1]]):
            return True, quotient_consecutive_elements
        return False, quotient_consecutive_elements
    else: return False, np.nan
    return False, np.nan

"""
test for sum between consecutive elements (Fibonacci numbers etc.)
"""
def fib_test(series):
    """
    fib_test(series):
    checks if all non np.nan elements in the series can be obtained by adding the previous two elements,
    if they are non np.nan.
    returns the boolean value of this check and the positions where it was true.
    """
    if(series[np.isnan(series)].size != 0):
        indices             = np.nonzero(np.logical_not(np.isnan(series)))[0]
        differences_indices = (indices[2:]-indices[1:-1])+(indices[1:-1]-indices[0:-2])
        if np.argwhere(differences_indices == 2).size <2:
            return False, np.asarray([np.nan])
        if np.allclose((series[indices[np.argwhere(differences_indices == 2)]]
                        + series[indices[np.argwhere(differences_indices == 2)]+1]),
                        series[indices[np.argwhere(differences_indices == 2)]+2]):
            return True, indices[np.argwhere(differences_indices == 2)]
        return False, indices[np.argwhere(differences_indices == 2)]
    else: return False, np.nan
    return False, np.nan

"""
Performs a set of predefined Tests regarding the structure of the (sub)series
"""
def perform_tests(series, mode = 'const'):
    """
    perform_tests(series, mode = 'const'):
    Tests the named test on a given series, if the test is known.
    """
    if mode == 'const':
        return const_test(series)
    elif mode == 'sum':
        return sum_test(series)
    elif mode == 'fac':
        return fac_test(series)
    elif mode == 'fib':
        return fib_test(series)
    else:
        print('ERROR: This mode doesn\'t exist yet.')
        return False, None
    pass

"""
Fills the result of the tests given to `perform_tests´ into the data object
"""
def fill_tests(data_object, thesis, start_index, step_size, value, mode = 'const'):
    """
    fill_tests(data_object, thesis, start_index, step_size, value, mode = 'const'):
    Fills the series with the corresponding values, according to the result of the corresponding test.
    """
    j = start_index
    i = step_size
    series = data_object.series
    if mode == 'const':
        positions = j+i*(np.squeeze(np.nonzero(np.isnan(thesis))))
        values = np.full(positions.shape, value)
        if positions.ndim != 0:
            data_object.fill_in(positions,values)
        else:
            data_object.fill_in(np.asarray([positions]),np.asarray([values]))
        data_object.positive_tests.append((mode, start_index, step_size, value))
        return
    elif mode == 'sum':
        positions = j+i*(np.squeeze(np.nonzero(np.isnan(thesis))))
        if np.nonzero(np.logical_not(np.isnan(thesis)))[0].ndim != 0:
            index = j+i*np.nonzero(np.logical_not(np.isnan(thesis)))[0][0]
        else:
            index = j+i*np.nonzero(np.logical_not(np.isnan(thesis)))[0]
        values = series[index]+((positions-index)/i)*value
        if positions.ndim != 0:
            data_object.fill_in(positions,values)
        else:
            data_object.fill_in(np.asarray([positions]),np.asarray([values]))
        data_object.positive_tests.append((mode, start_index, step_size, value))
        return
    elif mode == 'fac':
        positions = j+i*(np.squeeze(np.nonzero(np.isnan(thesis))))
        indices = (np.nonzero(np.logical_not(np.isnan(thesis)))[0])
        indices = indices[(np.nonzero(thesis[indices]!=0.)[0])]
        if indices.ndim != 0:
            index = j+i*indices[0]
        else:
            index = j+i*indices
        values = (series[index])*(value**((positions-index)/i))
        if positions.ndim != 0:
            data_object.fill_in(positions,values)
        else:
            data_object.fill_in(np.asarray([positions]),np.asarray([values]))
        data_object.positive_tests.append((mode, start_index, step_size, value))
        return 
    elif mode == 'fib':
        all_indices_thesis = np.arange(0,thesis.size)
        all_indices        = j+i* all_indices_thesis
        positions_thesis   = np.squeeze(np.nonzero(np.isnan(thesis)))
        positions          = j+i*positions_thesis
        indices            = np.setdiff1d(j+i* all_indices_thesis, j+i*positions_thesis)
        if positions.size >=2:
            #Filling downwards, that is if pos 2 and 3 are filled, then pos 1 can be filled
            for index in np.flip(all_indices):
                if (index in positions) and (index+i in indices) and (index+(2*i) in indices):
                    data_object.fill_in(np.asarray([index]),np.asarray([series[index+(2*i)]-series[index+i]]))
                    indices = np.concatenate((index, indices), axis=None)
                    positions = np.delete(positions, np.where(positions==index), axis=0)
            #Filling upwards, that is if pos 2 and 3 are filled, then pos 4 can be filled
            for index in all_indices:
                if (index in indices) and (index+i in indices) and (index+(2*i) in positions):
                    data_object.fill_in(np.asarray([index+(2*i)]),np.asarray([series[index+i]+series[index]]))
                    indices = np.concatenate((indices, index+(2*i)), axis=None)
                    positions = np.delete(positions, np.where(positions==index+(2*i)), axis=0)
        else:
            for index in np.flip(all_indices):
                if (index in np.asarray([positions,-1,-1])) and (index+i in indices) and (index+(2*i) in indices):
                    data_object.fill_in(np.asarray([index]),np.asarray([series[index+(2*i)]-series[index+i]]))
                    indices = np.concatenate((index, indices), axis=None)
                    if positions.size >= 2:
                        positions = np.delete(positions, np.where(positions==index+(2*i)), axis=0)
                    else:
                        if positions==index+(2*i):
                            positions = np.asarray([])    
            #Filling upwards, that is if pos 2 and 3 are filled, then pos 4 can be filled
            for index in all_indices:
                if (index in indices) and (index+i in indices) and (index+(2*i) in np.asarray([positions,-1,-1])):
                    data_object.fill_in(np.asarray([index+(2*i)]),np.asarray([series[index+i]+series[index]]))
                    indices = np.concatenate((indices, index+(2*i)), axis=None)
                    if positions.size >= 2:
                        positions = np.delete(positions, np.where(positions==index+(2*i)), axis=0)
                    else:
                        if positions==index+(2*i):
                            positions = np.asarray([])    
        data_object.positive_tests.append((mode, start_index, step_size, value))
        return 
    else:
        print('ERROR: This mode doesn\'t exist yet.')
        return False, None
    pass