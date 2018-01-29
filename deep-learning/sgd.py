#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 07:42:36 2017

@author: stepan
"""

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.kyes():
            params[key] -= self.lr * grads[key]
            
            