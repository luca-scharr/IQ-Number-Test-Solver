#The necessary imports
import copy
import functools
import random

import numpy as np
import numba as nb

class data:
    """
    contains:   __init__(self, positions, series)
                fill_in(self,position,values)
    """
    def __init__(self, positions, series, positive_tests, ancestor, gender):
        """
        __init__(self, positions, series)
        Input: np array of positions (int) and np array which contains the series (float, double etc. NOT int)
        Output: an Object which contains:
            1.) a np array with positions to predict (self.pos)
            2.) a np array which contains a series, used as ground-truth for predictions (self.truth)
            3.) the ground-truth masked at the given positions, denoted as np.nan (self.series)
            4.) the tests with positive outcome
            5.) the data object from which this data object got generated
            6.) the way this data object got generated
        """
        self.pos                      = positions
        self.truth                    = series
        series_work                   = copy.deepcopy(series)
        mask                          = np.full(series_work.shape,fill_value=False)
        mask[positions]               = True
        series_work[np.nonzero(mask)] = np.nan
        self.series                   = series_work
        self.positive_tests           = [positive_tests]
        self.ancestors                = [ancestor]
        self.gender                   = gender
        return
    
    def fill_in(self,position,values):
        """
        fill_in(self,position,values)
        Input: The series (from self), the positions to fill in (as np array, dtype int) 
               and the values to put (as np array, dtype castable into dtype of series)
        Output: The updated series, updated positions stored inside the object
        
        MAKE SURE that positions is a SUBSET of self.pos 
        
        """
        for v,p in zip(values,position):
            self.series[p] = v
        self.pos = np.setdiff1d(self.pos, position)
        return
    
    def enter_values(self, series):
        """
        Overwrite the series in a data object
        """
        self.series = series
        pass