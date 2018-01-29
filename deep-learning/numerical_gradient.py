#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 07:17:32 2017

@author: stepan
"""
import numpy as np

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

def function_2(x):
    return np.sum(x**2)

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x= init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -=lr * grad
    return x
    

#print(numerical_gradient(function_2, np.array([3.0,4.0])))
#init_x = np.array([-3.0,4.0])
#print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))