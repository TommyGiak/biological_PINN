# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:10:41 2023

@author: Tommaso Giacometti
"""
import numpy as np
from utils import pde_scipy, inverse_pinn_data_gen
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

MAX = 10000

#Tests for the pde function
@given(t=st.floats(0,MAX),params=arrays(np.float64, 4, elements=st.floats(0,MAX)))
@settings(max_examples=100) 
def test_pde_all_zero_for_no_cells(t,params):
    '''
    Test that the output is always zero if the input states are all zeros.
    '''
    init = np.zeros(4) #Input states equal to zero
    fn_result = pde_scipy(init, t, params) #Results of the differential equation
    ex_result = np.zeros(4) #We expect the result equal to zero
    assert np.array_equal(fn_result, ex_result)
    pass

@given(t=st.floats(0,MAX),val=arrays(np.float64, 4, elements=st.floats(0,MAX)))
@settings(max_examples=100) 
def test_pde_all_zero_zero_params(t,val):
    '''
    Test that the output is always zero if the input states are all zeros.
    '''
    params = np.zeros(4) #Input states equal to zero
    fn_result = pde_scipy(val, t, params) #Results of the differential equation
    ex_result = np.zeros(4) #We expect the result equal to zero
    assert np.array_equal(fn_result, ex_result)
    pass

#Test of data generation function
@given(t_size=st.integers(1,MAX))
@settings(max_examples=10)
def test_detagen_correct_size0(t_size):
    assert t_size > 0
    init = np.random.randint(0,MAX,size=(4,))
    time = np.linspace(0, 50, t_size)
    params = np.random.rand(4)
    data = inverse_pinn_data_gen(init, time, params, False)
    data_noise = inverse_pinn_data_gen(init, time, params, True)
    assert data.shape[0] == t_size
    assert data_noise.shape[0] == t_size
    pass

@given(t_size=st.integers(0,MAX),params=arrays(np.float64, 4, elements=st.floats(0,10)))
@settings(max_examples=10) 
def test_detagen_correct_size1(t_size, params):
    init = np.random.randint(0,MAX,size=(4,))
    time = np.linspace(0, 50, t_size)
    assert inverse_pinn_data_gen(init, time, params, False).shape[1] == 4
    assert inverse_pinn_data_gen(init, time, params, True).shape[1] == 4
    pass




