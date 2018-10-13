#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 18:52:59 2018

@author: anon
"""
"""
http://www.machinelearning.ru/wiki/index.php?title=%D0%98%D0%BD%D1%82%D0%B5%D1%80%D0%BF%D0%BE%D0%BB%D1%8F%D1%86%D0%B8%D1%8F_%D0%BA%D1%83%D0%B1%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%BC%D0%B8_%D1%81%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD%D0%B0%D0%BC%D0%B8
https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BF%D1%80%D0%BE%D0%B3%D0%BE%D0%BD%D0%BA%D0%B8
"""
import numpy as np
from bisect import bisect_right


def tridiag_solve_out(A, B, C, F):
    n = len(F)
    
    if n < 2:
        return None
    
    if len(C) != n:
        return None
    
    if len(A) != n - 1:
        return None
    
    if len(B) != n - 1:
        return None
    
    alpha = np.zeros(A.shape)
    beta  = np.zeros(A.shape)
    
    alpha[0] = -B[0] / C[0]
    beta[0]  =  F[0] / C[0]
    
    for i in range(1, n - 1):
        d = (A[i - 1] * alpha[i - 1] + C[i])
        alpha[i] = -B[i] / d
        beta[i]  = (F[i] - A[i - 1] * beta[i - 1]) / d
        
    x = np.zeros(F.shape)
    
    x[n - 1] = (F[n - 1] - A[n - 2] * beta[n - 2])
    x[n - 1] = x[n - 1] / (A[n - 2] * alpha[n - 2] + C[n - 1]) 
    
    for i in range(n - 2, -1, -1):
        x[i] = x[i + 1] * alpha[i] + beta[i]
        
    print(alpha)
    print(beta)
    
    return x

"""
Solve Mx = F where M is tridiag(A,C,B) inplace

first:
Alpha -> C
Beta  -> F

then:
x     -> F

returns x
"""
def tridiag_solve_in(A, B, C, F):
    n = len(F)
    
    if n < 2:
        raise ValueError
    
    if len(C) != n:
        raise ValueError
    
    if len(A) != n - 1:
        raise ValueError
    
    if len(B) != n - 1:
        raise ValueError
    
    d    = C[0]
    C[0] = -B[0] / d #Alpha[0]
    F[0] =  F[0] / d #Beta[0]
    
    for i in range(1, n - 1):
        d    = (A[i - 1] * C[i - 1] + C[i])
        C[i] = -B[i] / d                        #Alpha[i]
        F[i] = (F[i] - A[i - 1] * F[i - 1]) / d #Beta[i]
        
    F[n - 1] = (F[n - 1] - A[n - 2] * F[n - 2]) / (A[n - 2] * C[n - 2] + C[n - 1])
    
    for i in range(n - 2, -1, -1):
        F[i] = F[i + 1] * C[i] + F[i]
    
    return F

"""
A = np.array([1,2,7], dtype='float')
B = np.array([3,1,1], dtype='float')
C = np.array([2,2,2,2], dtype='float')
F = np.array([1,1,1,1], dtype='float')

x = tridiag_solve_out(A,B,C,F)
print(x)
print('---------------')
x = tridiag_solve_in(A,B,C,F)
print(A)
print(B)
print(C)
print(F)
print(x)
"""

class my_spline(object):
    
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.x = None
    
    def fit(self, x, y):
        n = len(x)
        
        if len(y) != n:
            raise ValueError
            
        i = x.argsort()
        y = y[i]
        x = x[i]
        i = None
        
        self.a = y[:-1]
        self.x = x[:-1]
        
        h = x[1:] - x[:-1]
        
        self.b = (y[1:] - y[:-1])/h
        
        self.c = np.zeros((n,), dtype='float')
        self.c[1:-1] = 3.0 * (self.b[1:] - self.b[:-1])
        
        tmp = 2.0 * (h[1:] + h[:-1])
        tridiag_solve_in(h[1:-1], h[1:-1], tmp, self.c[1:-1])
        tmp = None
        
        self.b -= h * (self.c[1:] + 2.0 * self.c[:-1]) / 3.0
        self.d = (self.c[1:] - self.c[:-1]) / h / 3.0
        
        self.c = self.c[:-1]
        
    def predict(self, X):
        
        y = np.zeros(X.shape, X.dtype)
        for i in range(0, len(X)):
            j = bisect_right(self.x, X[i]) - 1
            d = X[i] - self.x[j]
            y[i] = self.a[j] + d * (self.b[j] + d * (self.c[j] + d * self.d[j]))
            
        return y

'''
X = np.arange(0.0, 10.0, 0.05)
y = np.sin(X)

i = [0]
s = 1
while True:
    i.append(i[-1] + int(s))
    s*=1.2
    if i[-1] > len(X):
        break

X_train = X[i[:-1]]
y_train = y[i[:-1]]

si = my_spline()

si.fit(X_train, y_train)

z = si.predict(X)

import matplotlib.pyplot as plt

plt.plot(X,y,X,z)
'''
