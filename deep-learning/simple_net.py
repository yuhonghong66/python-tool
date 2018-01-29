#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 10:50:32 2017

@author: stepan
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
from softmax_function import softmax
from cross_entropy_error import cross_entropy_error
from numerical_gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))


t = np.array([0, 0, 1])
f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)