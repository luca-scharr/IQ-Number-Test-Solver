import copy
import functools
import random
import math
import numpy as np
import numba as nb


"""
Find the position of one arrays' elements in another array
"""
def where(array1,array2):
    """
    returns where element i of the first array is found in the second array.
    """
    return np.asarray([[i,j] for i in np.arange(array1.size) for j in np.arange(array2.size) if array1[i]==array2[j]]).T

"""
The following method finds the position at which the sequence contains primes
"""

def is_prime(nparray):
    """
    The vectorized version of prime
    """
    vprime = np.vectorize(prime)
    return vprime(nparray)

def prime(n):
    """
    Returns True if n is prime.
    Uses the fact that each prime>3 is of the form 6k+-1
    """
    if np.isnan(n):
        return False
    if n == 2.:
        return True
    if n == 3.:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return False
    i = 5
    w = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += w
        w = 6 - w
    return True

def vec_prime_pos(primeVec):
    """
    The vectorized version of prime_pos(primenumber, primenumbers = np.array([]))
    """
    vprime_pos = np.vectorize(prime_pos)
    return vprime_pos(primeVec)

def prime_pos(primenumber, primenumbers = np.array([])):
    """
    Given a primenumber, find out which primenumber (first, second, third, etc.) it is.
    """
    if primenumber in primenumbers:
        return np.where(primenumbers == primenumber)
    elif primenumber >= 11:
        counter = 0.
        upbound = math.floor(math.log10(primenumber))
        for i in np.arange(upbound):
            operating_range = np.arange(10**(i),10**(i+1))
            primes          = is_prime(operating_range)
            counter        += primes.sum()
            operating_range = operating_range[primes]
            primenumbers    = np.concatenate((primenumbers,
                                              operating_range[np.logical_not(np.isin(operating_range,primenumbers))]))
        operating_range = np.arange(10**(upbound),primenumber)
        primes          = is_prime(operating_range)
        counter        += primes.sum()
        operating_range = operating_range[primes]
        primenumbers    = np.concatenate((primenumbers,
                                          operating_range[np.logical_not(np.isin(operating_range,primenumbers))]))
        return counter
    else:
        counter = 0.
        operating_range = np.arange(primenumber)
        if operating_range.size >= 2:
            primes          = is_prime(operating_range)
            counter        += primes.sum()
            operating_range = operating_range[primes]
            primenumbers    = np.concatenate((primenumbers,
                                              operating_range[np.logical_not(np.isin(operating_range,primenumbers))]))
        else:
            primes = prime(operating_range)
            counter        += primes
            operating_range = operating_range[primes]
            if operating_range.size >= 1:
                primenumbers    = np.concatenate((primenumbers,
                                                  operating_range[np.logical_not(np.isin(operating_range,primenumbers))]))
        return counter
    pass

"""
The following method finds the position at which the sequence contains squares
"""

def is_square(nparray):
    """
    The vectorized version of square
    """
    vsquare = np.vectorize(square)
    return vsquare(nparray)
    
def square(n):
    """
    Returns True if n is square; Includes 0.
    """
    if np.isnan(n):
        return False
    i = 0.
    while (not math.isclose(i ** 2, n))&(i ** 2 <=n+2):
        i+=1
    if math.isclose(i ** 2, n):
        return True
    return False

def vec_square_pos(squareVec):
    """
    Vectorized Version of square_pos(squarenumber)
    """
    vsquare_pos = np.vectorize(square_pos)
    return vsquare_pos(squareVec)


def square_pos(squarenumber):
    """
    Given a squarenumber, find out which squarenumber (first, second, third, etc.) it is.
    """
    return math.sqrt(squarenumber)

"""
The following method finds the position at which the sequence contains cubes
"""

def is_cube(nparray):
    """
    The vectorized version of cube
    """
    vcube = np.vectorize(cube)
    return vcube(nparray)
    
def cube(n):
    """
    Returns True if n is cube; Includes 0.
    """
    if np.isnan(n):
        return False
    i = 0.
    while ((not math.isclose(i ** 3, n))&(i ** 3 <=n+2))|((not math.isclose((-i) ** 3, n))&((-i) ** 3 >=n-2)):
        i+=1
    if math.isclose(i ** 3, n):
        return True
    elif math.isclose((-i) ** 3, n):
        return True
    return False

def vec_cube_pos(cubeVec):
    """
    Vectorized Version of cube_pos(cube)
    """
    vcube_pos = np.vectorize(cube_pos)
    return vcube_pos(cubeVec)

def cube_pos(cube):
    """
    Given a cube, find out which cube (first, second, third, etc.) it is.
    """
    return np.cbrt(cube)


"""
Performs a set of predefined Tests regarding the structure of each element
"""
def perform_element_tests(series, mode = 'prime'):
    """
    perform_element_tests(series, mode = 'prime'):
    Tests the named test on a given series, if the test is known.
    These tests focus on properties of an Element.
    """
    if mode == 'prime':
        return is_prime(series)
    elif mode == 'square':
        return is_square(series)
    elif mode == 'cube':
        return is_cube(series)
    else:
        print('ERROR: This mode doesn\'t exist yet.')
        return False
    pass

"""
Fills the result of the elementwise tests on subseries given to `perform_element_tests´ into the data object
"""
def fill_ele_test_space(data_object, thesis, start_index, step_size, mode = 'prime', sign = '+'):
    """
    fill_ele_test_space(data_object, thesis, start_index, step_size, mode = 'prime'):
    Input:
     data_object -> data type object
     thesis -> a subseries which fullfills mode and is hypothesised to contain usefull information
     start_index -> the index where thesis starts
     step_size -> the size of the steps between two elements of the thesis
     mode -> what test was succsessfull
     sign -> the added prefix to the thesis
    Output:
     None
    Fills the series with the corresponding values, according to the result of the corresponding test.
    """
    series         = data_object.series
    thesis_non_nan = thesis[np.nonzero(np.logical_not(np.isnan(thesis)))]
    if mode == 'prime':
        thesis_non_nan = vec_prime_pos(thesis_non_nan)
        if sign == '+':
            series[start_index + step_size * np.asarray(np.nonzero(np.logical_not(np.isnan(thesis))))] = thesis_non_nan
        elif sign == '-':
            series[start_index + step_size * np.asarray(np.nonzero(np.logical_not(np.isnan(thesis))))] = -thesis_non_nan
        else:
            print("Error: I\'ve never seen that funky sign in my life.'")
    elif mode == 'square':
        thesis_non_nan = vec_square_pos(thesis_non_nan)
        if sign == '+':
            series[start_index + step_size * np.asarray(np.nonzero(np.logical_not(np.isnan(thesis))))] = thesis_non_nan
        elif sign == '-':
            series[start_index + step_size * np.asarray(np.nonzero(np.logical_not(np.isnan(thesis))))] = -thesis_non_nan
        else:
            print("Error: I\'ve never seen that funky sign in my life.'")
    elif mode == 'cube':
        thesis_non_nan = vec_cube_pos(thesis_non_nan)
        if sign == '+':
            series[start_index + step_size * np.asarray(np.nonzero(np.logical_not(np.isnan(thesis))))] = thesis_non_nan
        elif sign == '-':
            series[start_index + step_size * np.asarray(np.nonzero(np.logical_not(np.isnan(thesis))))] = -thesis_non_nan
        else:
            print("Error: I\'ve never seen that funky sign in my life.'")
    else:
        print('ERROR: This mode doesn\'t exist yet.')
    data_object.enter_values(series)
    data_object.positive_tests.append((mode, start_index, step_size, sign))
    pass

"""
In the following there are tests in whichs output one would try to find new subseries.
This would be another option for defining the searchstructure. 
"""

"""
Test for Primenumbers
"""
def test_prime(series, primenumbers=np.asarray([])):
    """
    test_prime(series, primenumbers=None):
    Input: A series and potentialy an array of primes to check against.
    Looks for positions in the series that are prime.
    """
    if primenumbers.size != 0:
        indice = np.arange(series.size)
        indice = indice[where(series,primenumbers)]
    else:
        indice = test_prime_offline(series)
    if indice.size !=0:
        return True, indice, np.asarray([])
    return False, np.asarray([]), np.asarray([])
    
def test_prime_offline(series):
    """
    test_prime_offline(series):
    Input: A series
    Looks for positions in the series that are prime. 
    """
    indice = np.arange(series.size)
    indice = indice[is_prime(series)]
    return indice

"""
Test for squares
"""
def test_square(series, squares=np.asarray([])):
    """
    test_square(series, squares=None):
    Input: A series and potentialy an array of squares to check against.
    Looks for positions in the series that are square.
    """
    if squares.size != 0:
        indice = np.arange(series.size)
        indice, positions = where(series,squares)
        #print([np.isin(squares,series)])
        #arrays = [ for _ in range(2)]
    else:
        indice, positions = test_square_offline(series)
    if indice.size !=0:
        return True, indice, positions
    return False, np.asarray([]), np.asarray([])
    
def test_square_offline(series):
    """
    test_square_offline(series):
    Input: A series
    Looks for positions in the series that are square. 
    """
    indice = np.arange(series.size)
    indice = indice[is_square(series)[0]]
    return indice,is_square(series)[1][np.logical_not(np.isnan(is_square(series)[1]))]


"""
Test for cubes
"""
def test_cube(series, cubes=np.asarray([])):
    """
    test_cube(series, cubes=None):
    Input: A series and potentialy an array of cubes to check against.
    Looks for positions in the series that are cubes.
    """
    if cubes.size != 0:
        indice = np.arange(series.size)
        indice, positions = where(series,cubes)
    else:
        indice, positions = test_cube_offline(series)
    if indice.size !=0:
        return True, indice, positions
    return False, np.asarray([]), np.asarray([])
    
def test_cube_offline(series):
    """
    test_cube_offline(series):
    Input: A series
    Looks for positions in the series that are cubes. 
    """
    indice = np.arange(series.size)
    indice = indice[is_cube(series)[0]]
    return indice,is_cube(series)[1][np.logical_not(np.isnan(is_cube(series)[1]))]

