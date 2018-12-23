#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spline interpolation and least squares prototypes.
Copyright (c) 2018 Paul Beltyukov (beltyukov.p.a@gmail.com)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""
http://www.machinelearning.ru/wiki/index.php?title=%D0%98%D0%BD%D1%82%D0%B5%D1%80%D0%BF%D0%BE%D0%BB%D1%8F%D1%86%D0%B8%D1%8F_%D0%BA%D1%83%D0%B1%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%BC%D0%B8_%D1%81%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD%D0%B0%D0%BC%D0%B8
https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BF%D1%80%D0%BE%D0%B3%D0%BE%D0%BD%D0%BA%D0%B8
"""
from bisect import bisect_right
import numpy as np
import numpy.random as rnd
from my_aco import MyAntColony

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
    n = len(C)
    
    if n < 2:
        raise ValueError
    
    if F.shape[0] != n:
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

class MySpline(object):
    
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
            if j < 0:
                j = 0
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

si = MySpline()

si.fit(X_train, y_train)

z = si.predict(X)

import matplotlib.pyplot as plt

plt.plot(X,y,X,z)
'''
class MySplineLSI(object):
    
    def __init__(self, x):
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.x = None
        self._get_basic_splines(x)
    
    def _get_basic_splines(self, x):
        n = len(x)
        x = x[x.argsort()]
        #we need n+2 basic splines
        y = np.zeros((n,n + 2), dtype='float')
        y[:,1:n + 1] = np.eye(n, dtype='float')
                
        self.a = y[:-1]
        self.x = x[:-1]
        
        h = x[1:] - x[:-1]
        
        self.b = (y[1:] - y[:-1])/h[:,None]
        
        self.c = np.zeros((n,n + 2), dtype='float')
        self.c[1:-1]   = 3.0 * (self.b[1:] - self.b[:-1])
        #We have two special basic splines with zero vals in pivot points, 
        #but with nonzero second derivatives on first and last pivot points
        self.c[0,0]    = 1.0
        self.c[1,0]   -= h[0]
        self.c[-1,-1]  = 1.0
        self.c[-2,-1] -= h[-1]
        
        tmp = 2.0 * (h[1:] + h[:-1])
        tridiag_solve_in(h[1:-1], h[1:-1], tmp, self.c[1:-1])
        tmp = None
        
        self.b -= h[:,None] * (self.c[1:] + 2.0 * self.c[:-1]) / 3.0
        self.d = (self.c[1:] - self.c[:-1]) / h[:,None] / 3.0
        
        self.c = self.c[:-1]
        
    def _eval_basic_splines(self, X):
        
        n = len(X)
        y = np.zeros((n,self.a.shape[1]), X.dtype)
        for i in range(0, n):
            j = bisect_right(self.x, X[i]) - 1
            if j < 0:
                j = 0
            d = X[i] - self.x[j]
            y[i] = self.a[j] + d * (self.b[j] + d * (self.c[j] + d * self.d[j]))
            
        return y
    
    def fit(self,X,y):
        
        A = self._eval_basic_splines(X)
        w = np.linalg.pinv(A).dot(y)
        self.a = self.a.dot(w)
        self.b = self.b.dot(w)
        self.c = self.c.dot(w)
        self.d = self.d.dot(w)
        
        return self
        
    def predict(self, X):
        
        y = np.zeros(X.shape, X.dtype)
        for i in range(0, len(X)):
            j = bisect_right(self.x, X[i]) - 1
            if j < 0:
                j = 0
            d = X[i] - self.x[j]
            y[i] = self.a[j] + d * (self.b[j] + d * (self.c[j] + d * self.d[j]))
            
        return y

'''
import matplotlib.pyplot as plt

X = np.arange(0.0, 10.0, 0.05)

lss = MySplineLSI(X[::30])
s = MySpline()

y = np.sin(X)

s.fit(X[::30], y[::30])
lss.fit(X,y)

sp = s.predict(X)
lssp = lss.predict(X)

plt.plot(X,y, X,sp, X,lssp)
plt.plot(X,y-sp, X,y-lssp)
'''

class MySplineMMLSI(object):
    def __init__(self, _Q=1.0, _elite=2.0, _a=1.0, _b=1.0, _r=0.005, _N=4, _Ants=20, _epochs=100):
        self.Q     = _Q
        self.elite = _elite
        self.a = _a
        self.b = _b
        self.r = _r
        self.N = _N
        self.Ants = _Ants
        self.epochs = _epochs
        self.spline = None
        
    def predict(self, X):
        if self.spline is not None:
            return self.spline.predict(X)
    
    def fit(self, X, y):
        n = self.N
        
        graph = np.triu(np.ones(len(X)), k=1)
        
        def _start_(opt):
            return rnd.choice(np.nonzero(opt.GW[0])[0])
            
        def _stop_(path):
            #global n
            #global X
            if len(path) >= n or path[-1] >= len(X) - 1:
                return True
            else:
                return False
        
        def _weigth_(path):
            #global X
            try:
                xs = X[path]
                s = MySplineLSI(xs)
                s.fit(X,y)
                d = s.predict(X) - y
                rmax = 1.0/(np.max(np.abs(d))) #Minimize max error
                return rmax
            except Exception as e:
                #print(e)
                return 0.0
                
        aco = MyAntColony(graph, self.Q, self.elite, self.a, self.b, self.r,
                          _start_, _stop_, _weigth_, self.Ants)
        
        path,rscore = aco.run(self.epochs)
        print('The final score is:', 1.0/rscore)
        
        self.spline = MySplineLSI(X[path])
        self.spline.fit(X,y)
        
        return self

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_excel('TunelDiode.xlsx')
    df.sort_values('U , в', inplace=True)
    df = df.reset_index(drop=True)
    
    X = df['U , в'].values
    y = df['I , мА'].values
    n = 4
    
    s = MySplineMMLSI(_N=4)
    s.fit(X,y)
    
    xz = np.linspace(X[0], X[-1], len(X)*10)
    z = s.predict(xz)
    
    plt.scatter(X, y)
    plt.plot(xz, z, color='orange')
    plt.show()