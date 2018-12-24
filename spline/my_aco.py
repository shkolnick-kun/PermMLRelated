#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An ant colony optimization method prototypes.
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
import numpy as np
import numpy.random as rnd

class MyAnt(object):
    def __init__(self, _P, _Q, stop_func, weigth_func):
        self.P = _P               #Choice probabilities
        self.Q = _Q               #Feromone intensity
        self.stop   = stop_func   #Have we found some food?
        self.weigth = weigth_func #How much food have we found?
        
    def run(self, start):
        
        delta_F = np.zeros(self.P.shape)
        path = [min((start, self.P.shape[0]))]
        
        while not self.stop(path):
            i = path[-1]
            segs = np.nonzero(self.P[i])[0]
            if 0 == len(segs):
                break
            j = rnd.choice(segs, p=self.P[i][segs])
            if j not in path:
                path.append(j)
                delta_F[i,j] = self.Q
        
        w = self.weigth(path)
        delta_F *= w
        
        return w, path, delta_F

class MyAntColony(object):
    
    def _calc_probabilities(self):
        P = np.power(self.GW, self.b) * np.power(self.F, self.a)
        for i in range(0, P.shape[0]):
            s = np.sum(P[i])
            if s > 0:
                P[i] /= s
            else:
                P[i] = 0.0
        return P
        
    def __init__(self, _GW, _Q, elite, _a, _b, _r, start_func, stop_func, weigth_func, AntNum):
        self.GW = _GW #Grapt with weights
        self.Q  = elite * _Q
        self.a = _a
        self.b = _b
        self.r = _r
        self.F = (_GW > 0.0).astype(np.float)
        self.P = self._calc_probabilities()
        self.start = start_func
        self.ants = []
        for i in range(0,AntNum):
            self.ants.append(MyAnt(self.P, _Q, stop_func, weigth_func))
            
    def run(self, epochs, target = 0):
        elite_w = 0
        elite_p = []
        for i in range(0, epochs):
            weigths   = []
            paths     = []
            #
            for ant in self.ants:
                #print(ant)
                w,p,dF = ant.run(self.start(self))
                #
                weigths.append(w)
                paths.append(p)
                #
                self.F += dF
                self.F *= 1.0 - self.r
                np.copyto(self.P, self._calc_probabilities())
            #Try to find the Elite ant
            e = weigths.index(max(weigths))
            if weigths[e] > elite_w:
                elite_w = weigths[e]
                elite_p = paths[e]
            #Elite ant run
            if 0 != len(elite_p):
                delta_F = np.zeros(self.P.shape)
                cur = elite_p[0]
                for point in elite_p[1:]:
                    delta_F[cur, point] = self.Q * 10
                    cur = point
                self.F += delta_F * elite_w
                self.F *= 1.0 - self.r
                np.copyto(self.P, self._calc_probabilities())
            #
            print('Epoch:', i, 'MaxW:', max(weigths), 'ElW', elite_w)
            if target > 0 and elite_w > target:
                break
        #
        i = weigths.index(max(weigths))
        return elite_p, elite_w
    
if __name__ == "__main__":
    Graph = np.array([
            [0,1,1,0,0],
            [0,0,1,0,0],
            [0,0,0,1,1],
            [0,0,0,0,1],
            [0,0,0,0,1]
            ], dtype=np.float)
    
    def _start_(obj):
        return 0
    
    def _stop_(path):
        if path[-1] == 4:
            return True
        else:
            return False
    
    def _weigth_(path):
        return 1.0/(len(path) + 0.0001)
    
    aco = MyAntColony(Graph, 1.0, 2, 1.0, 1.0, 0.2, _start_, _stop_, _weigth_, 100)
    
    print(aco.run(50, 100))
    