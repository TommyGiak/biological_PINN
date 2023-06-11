# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:10:41 2023

@author: Tommaso Giacometti
"""
import numpy as np
from first_pinn import pde 
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

MAX = 10000

#Tests for the pde function
@given(t=st.floats(0,MAX),params=arrays(np.float64, 4, elements=st.floats(0,MAX)))
@settings(max_examples=100) 
def test_all_zero_for_no_cells(t,params):
    '''
    Test that the output is always zero if the input states are all zeros.
    '''
    init = np.zeros(4) #Input states equal to zero
    fn_result = pde(init, t, params) #Results of the differential equation
    ex_result = np.zeros(4) #We expect the result equal to zero
    assert np.array_equal(fn_result, ex_result)
    pass

@given(t=st.floats(0,MAX),val=arrays(np.float64, 4, elements=st.floats(0,MAX)))
@settings(max_examples=100) 
def test_all_zero_zero_params(t,val):
    '''
    Test that the output is always zero if the input states are all zeros.
    '''
    params = np.zeros(4) #Input states equal to zero
    fn_result = pde(val, t, params) #Results of the differential equation
    ex_result = np.zeros(4) #We expect the result equal to zero
    assert np.array_equal(fn_result, ex_result)
    pass