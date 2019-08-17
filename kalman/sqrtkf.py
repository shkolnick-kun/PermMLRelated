# -*- coding: utf-8 -*-
"""
Square root Kalman filter prototypes.
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
#------------------------------------------------------------------------------
"""
Пока тут лежат только линейниые прототипы фильтров Калмана, 
если будет время/желание - сделаю ОФК и другие алгоритмы.
Currenty there are only linear KF prototypes here,
if I have more time and will, I'll try to implement 
EKF and other things. 
Ссылки/References:
1. Triangular Covariance Factorizations for Kalman Filtering 
   https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770005172.pdf
   
2. Solutions to the Kalman filter wordlength problem: 
   Square root and U-D covariance factorizations
   www.dtic.mil/dtic/tr/fulltext/u2/a049704.pdf
   
3. Цыганова, Ю. В. О методах реализации UD-фильтра / Ю. В. Цыганова // 
   Известия высших учебных заведений. Поволжский регион. Физико-математические 
   науки. – 2013. – No 3 (27). – С. 84–104.
   http://izvuz_fmn.pnzgu.ru/files/izvuz_fmn.pnzgu.ru/7313.pdf
   
4. БОРТОВАЯ РЕАЛИЗАЦИЯ АДАПТИВНО-РОБАСТНЫХ ОЦЕНИВАЮЩИХ ФИЛЬТРОВ: 
   ПРАКТИЧЕСКИЕ РЕЗУЛЬТАТЫ / В.Л. БУДКИН, С.Л. БУЛГАКОВ, Ю.П. МИХЕЕНКОВ, 
   А.В. ЧЕРНОДАРОВ, А.П. ПАТРИКЕЕВ // НАУЧНЫЙ ВЕСТНИК МГТУ ГА 
   серия Авионика и электротехника No 89(7) - УДК 629.7.05
   
5. Chernodarov A.V., Kovregin V.N. An H ∞ Technology for the Detection and 
   Damping of Divergent Oscillations in Updatable Inertial Systems 
   // Proc. of the International Conference “Physics and Control”.– 
   St.Petersburg, 2003, ed.1, pp. 121-126.
"""
#------------------------------------------------------------------------------
'''
See [1,2]
P = U * diag(D) * U^T decomposition
'''
def udu(p):
    
    if 2 != len(p.shape):
        return None
    
    if p.shape[0] != p.shape[1]:
        return None
    
    n = p.shape[0]
    
    u = np.zeros((n, n))
    d = np.zeros((n))
    
    d[n-1] = p[n-1,n-1]
    u[:,n-1]   = p[:,n-1] / d[n-1]

    for j in range(n-2, -1, -1):
        dd = np.diag(d[j+1:])
        c = np.dot(dd, u[j,j+1:])
        d[j] = p[j,j] - np.dot(u[j,j+1:].transpose(), c)

        if d[j] == 0:
            return None

        for i in range(j, -1, -1):
            c = np.dot(dd, u[j,j+1:])
            u[i,j] = (p[i,j] - np.dot(u[i,j+1:].transpose(), c))/d[j]
            
    return u, d
#------------------------------------------------------------------------------
'''
MWGS update see [1,2]:
U * diag(D) * U^T = w * diag(d) * w^T
    
Params:
w - is n*k float full rank
d - is k*k float diag
where k>n
return:
u - is n*n float upper triangular
D - id n*n float diag
'''
def mwgs(w,d):
    
    if 1 != len(d.shape):
        return None
    
    if 2 != len(w.shape):
        return None
    
    if w.shape[1] != d.shape[0]:
        return None
    
    if w.shape[0] >= d.shape[0]:
        return None
    
    n = w.shape[0]
    
    u = np.eye(n)
    D = np.zeros((n))
    
    for i in range(n-1, -1, -1):
        c = w[i,:] * d
        D[i] = np.dot(w[i,:], c)
        
        if D[i] <= 0:
            # How about partial reset heu_ristics here?
            return None
        
        dd = c/D[i]
        
        for j in range(0, i):
            u[j,i]  = np.dot(dd, w[j,:])
            w[j,:] -= u[j,i] * w[i,:]
    
    return u, D
#------------------------------------------------------------------------------
'''
Params:
F  - n*n transition mtx
P- = U * diag(D) * U^T state covariance
B  - p*p pertu_rbation mtx
Rq = Uq * diag(Dq) * Uq^T - prceess noise covariance
Ratu_rn:
P+ = U * diag(D) * U^T
'''
def _predict(U, D, F, B, Uq, Dq):
    
    W = np.concatenate((F.dot(U), B.dot(Uq)), axis = 1)
    D = np.concatenate((D, Dq))
    
    return mwgs(W, D)
#------------------------------------------------------------------------------
'''
Params:
x  - old state
F  - n*n transition mtx
P- = U * diag(D) * U^T state covariance
B  - p*p pertu_rbation mtx
Rq = Uq * diag(Dq) * Uq^T - prceess noise covariance
b  - transition bias vector
Ratu_rn:
x+ - predicted state
P+ = U * diag(D) * U^T predicted state covariance
'''
def _predict_lin(x, U, D, F, bf, B, Uq, Dq):
    # Predict x
    X = F.dot(x) + bf
    # Predict P
    u,d = _predict(U, D, F, B, Uq, Dq)
    
    return X, u, d
#------------------------------------------------------------------------------
'''
Linear measurement update
Params:
P- = U * D * U^T old state covariance
x- -             old state mean
H  - decorrelated observtion mtx
z  - decorrelated observation
d_r - decorrelated noise covariance as vector R = u_r * diag(d_r) * u_r^T
return:
P+ = U * D * U^T new state covariance 
x+ -             new state mean
'''
def _correct_lin(x, u, d, H, z, d_r, _scalar_correct):
    
    n = z.shape[0]
    X = x
    U = u
    D = d
    
    for i in range(0, n):
        U,D,K = _scalar_correct(U, D, H[i], d_r[i])
        #Predict measurement
        y = H[i].dot(X)
        #Correct state
        X += K * (z[i] - y)
        
    return X, U, D
#------------------------------------------------------------------------------
class _srkf_base(object):
    def __init__(self, x, P, Q, R):
        #Transition and bias
        self.x = x
        
        u,d = udu(P);
        self.U = u
        self.D = d

        u,d = udu(Q);
        self.Uq = u
        self.Dq = d

        u,d = udu(R);
        self.d_r = d
        #Decorrelation mtx
        self.dm = np.linalg.pinv(u)
#------------------------------------------------------------------------------
'''
Square root linear kalman filter base class
'''
class _srkf_lin(_srkf_base):
    
    def __init__(self, x, P, Q, R, _scalar_correct):   
        #base init
        _srkf_base.__init__(self, x, P, Q, R)        
        #Scalar correct function
        self.scalar_correct = _scalar_correct
        
    def run(self, F, B, H, z, bf = None, bh = None):
        #Constructing default biases
        if bh is None:
            bh = np.zeros(z.shape)
            
        if bf is None:
            bf = np.zeros(self.x.shape)
        #Decorrelate observations    
        z   = self.dm.dot(z - bh)
        Hd  = self.dm.dot(H)

        #Time update/Predict
        self.x, self.U, self.D = _predict_lin(self.x, self.U, self.D, F, bf, B, self.Uq, self.Dq)
        #Measu_reent update/Correct
        self.x, self.U, self.D = _correct_lin(self.x, self.U, self.D, Hd, z, self.d_r, self.scalar_correct)
        #return corrected observation
        return H.dot(self.x) + bh
#------------------------------------------------------------------------------    
'''
Params:
P+ = u * diag(d) * u^T state covariance
h  - decorrelated observation mtx collumn
x  - state mean scalar
e  = z - scalar(H(x)) - scalar error
r  - decorrelated observtion noise covariance scalar
return: 
P++ = U * diag(D) * U^T state covariance
K   - Kalman gain vector
'''
def _bierman_scalar_correct(u, d, h, r):
    
    f = h.dot(u)
    v = d * f
    
    n = d.shape[0]
    
    U = np.eye(n)
    D = np.zeros(d.shape)
    b = np.zeros(d.shape)
    
    a    = r + f[0] * v[0] # a[0]
    D[0] = d[0] * r / a
    b[0] = v[0]
    
    for k in range(1, n):
        # a_z denotes a[k-1]
        # a   denotes a[k]
        a_z    = a
        a      = a_z + f[k] * v[k]
        D[k]   = d[k] * a_z / a
        
        '''
        #This is fast but it's not for C-prototyping
        U[:,k] = u[:,k] - f[k] / a_z * b
        b     += u[:,k] * v[k]
        '''
        #This is slower but more natu_ral for C-prototyping, 
        #see USA DoD document
        b[k] = v[k]
        p    = - f[k] / a_z
        for j in range(0, k):
            U[j,k]  = u[j,k] + p * b[j]
            b[j]   += u[j,k] * v[k]
        #'''
        
    
    K = b / a #Kalman gain vector
    
    return U, D, K
#------------------------------------------------------------------------------
class bierman_lin(_srkf_lin):
    '''
    Standard SRKF in UD form
    '''
    def __init__(self, x, P, Q, R):
        _srkf_lin.__init__(self, x, P, Q, R, _bierman_scalar_correct)
#------------------------------------------------------------------------------
'''
Params:
P+ = u * diag(d) * u^T state covariance
h  - decorrelated observation mtx collumn
x  - state mean scalar
e  = z - scalar(H(x)) - scalar error
r  - decorrelated observtion noise covariance scalar
return: 
P++ = U * diag(D) * U^T state covariance
K   - Kalman gain vector
'''
def _joseph_scalar_correct(u, d, h, r):
    
    f = h.dot(u)
    v = d * f
    
    n = d.shape[0]
    
    a = r + f.dot(v)
    K = u.dot(v / a)
    
    WW = np.concatenate((np.outer(K,f) - u, K.reshape((n,1))), axis = 1)
    DD = np.concatenate((d, np.array([r])))
    
    U,D = mwgs(WW, DD)
    
    return U, D, K
#------------------------------------------------------------------------------
class joseph_lin(_srkf_lin):
    '''
    SRKF iu UD form with Joseph update:
    P+ = (K*H - I)*P*(K*H - I)^T + K*R*K^T 
    It consumes more time and memory than bierman, 
    but it makes possible to add correcctions for non-Gaussian
    noise see [3,4].
    '''
    def __init__(self, x, P, Q, R):
        _srkf_lin.__init__(self, x, P, Q, R, _joseph_scalar_correct)
#------------------------------------------------------------------------------
'''
Square root linear block Kalman filter see [3].
'''
class srkf_block(_srkf_base):
    
    def __init__(self, x, P, Q, R):   
        #base init
        _srkf_base.__init__(self, x, P, Q, R)
        #This is not true X!!!
        self.x = np.linalg.pinv(self.U).dot(self.x)/self.D
    
    def get_x(self):
        return self.U.dot(self.D*self.x)
        
    def run(self, F, B, H, z, bh = None):
        #Constructing default biases
        if bh is None:
            bh = np.zeros(z.shape)
        #Decorrelate observations
        z   = self.dm.dot(z - bh)
        Hd  = self.dm.dot(H)
        
        nq = self.Dq.shape[0]
        nz = z.shape[0]
        nx = F.shape[0]
        
        U  = self.U
        Uq = self.Uq
        
        #                       1*nq              1*nx                 1*nz
        w1 = np.concatenate((np.zeros((nq)),    self.x,         -z/self.d_r))
        #                      nx*nq              nx*nx               nx*nz 
        w2 = np.concatenate((B.dot(Uq),         F.dot(U), np.zeros((nx,nz))), axis = 1)
        #                      nz*nq              nz*nx               nz*nz
        w3 = np.concatenate((np.zeros((nz,nq)), Hd.dot(U),       np.eye(nz)), axis = 1)
        # Construct W
        W  = np.concatenate((w1.reshape((1, nq + nx + nz)), w2, w3), axis = 0)
        
        D = np.concatenate((self.Dq, self.D, self.d_r))
        
        UU, DD = mwgs(W,D)
        
        self.x = UU[         0, 1 : 1 + nx]
        self.U = UU[1 : 1 + nx, 1 : 1 + nx]
        self.D = DD[1 : 1 + nx]
        
        return H.dot(self.get_x()) + bh
    
    
#------------------------------------------------------------------------------
'''
Robust/adaptive Linear measurement update
Params:
P- = U * D * U^T old state covariance
x- -             old state mean
H  - decorrelated observtion mtx
z  - decorrelated observation
d_r - decorrelated noise covariance square root as vector
psi_func    - influence function
psidor_func - influence function first derivative
return:
P+ = U * D * U^T new state covariance 
x+ -             new state mean
'''
def _correct_ra_lin(x, u, d, H, z, d_r, psi_func, psidot_func, _scalar_correct):
    
    n = z.shape[0]
    X = x
    U = u
    D = d
    
    for i in range(0, n):
        X,U,D = _scalar_correct(X, U, D, H[i], z[i], d_r[i], psi_func, psidot_func)
                
    return X, U, D
#------------------------------------------------------------------------------
'''
Robust/adaptive square root linear kalman filter base class 
'''
class _srkf_ra_lin(_srkf_base):
    
    #Default noice model is Normal with Poisson outliers
    def _default_psi(beta):
        # +- 3*sigma
        if 3.0 >= np.abs(beta):
            return beta
        
        # +- 6*sigma - uncertain measurements
        if 6.0 >= np.abs(beta):
            return beta/3.0
        
        # outliers
        return float(np.sign(beta))
    
    def _default_psidot(beta):
        # +- 3*sigma
        if 3.0 >= np.abs(beta):
            return 1.0
        
        # +- 6*sigma - uncertain measurements
        if 6.0 >= np.abs(beta):
            return 1.0/3.0
        
        # outliers
        return 0.0
    
    def __init__(self, x, P, Q, R, psi_func, psidot_func, _scalar_correct):
        #base init
        _srkf_base.__init__(self, x, P, Q, R)
        
        # Must be sqrt(d)
        self.d_r = np.sqrt(self.d_r)
        
        #Scalar correct function
        self._scalar_correct = _scalar_correct
        
        #Influence function
        self.psi    = psi_func
        self.psidot = psidot_func
        
    def run(self, F, B, H, z, bf = None, bh = None):
        #Constructing default biases
        if bh is None:
            bh = np.zeros(z.shape)
            
        if bf is None:
            bf = np.zeros(self.x.shape)
        #Decorrelate observations    
        z   = self.dm.dot(z - bh)
        Hd  = self.dm.dot(H)

        #Time update/Predict
        self.x, self.U, self.D = _predict_lin(self.x, self.U, self.D, F, bf, B, self.Uq, self.Dq)
        #Measu_reent update/Correct
        self.x, self.U, self.D = _correct_ra_lin(self.x, self.U, self.D, Hd, z, self.d_r, self.psi, self.psidot, self._scalar_correct)
        #return corrected observation
        return H.dot(self.x) + bh
    
#------------------------------------------------------------------------------
'''
Linear robust measurement update internal part 
for robust and adaptive robust filters
Params:
P- = U * D * U^T old state covariance
x- -             old state mean
H  - decorrelated observtion mtx
z  - decorrelated observation
d_r - decorrelated noise covariance as vector R = u_r * diag(d_r) * u_r^T
return:
K  = Kalman gain vector
x+ -             new state mean
P+ = U * D * U^T new state covariance 
'''
def _intra_correct_robust_scalar(x, u, d, h, z, r, psi_func, psidot_func):
    #Innovation
    nu     = z - h.dot(x)
    #Normalized innovation
    beta   = nu / r
    #Influence function
    psi    =    psi_func(beta)
    psidot = psidot_func(beta)
    #Noise dispersion
    disp  = r * r
    
    #Now do robust Joseph-like update
    f = h.dot(u)
    v = d * f
    
    n = d.shape[0]
    
    a = disp + f.dot(v) * psidot
    K = u.dot(v / a)
     
    WW = np.concatenate((np.outer(K,f) * psidot - u, K.reshape((n,1))), axis = 1)
    DD = np.concatenate((d, np.array([disp * psidot])))
    
    U,D = mwgs(WW, DD)
        
    #Correct state
    X = x + K * psi * r
    
    return X, U, D, K, psi, psidot
#------------------------------------------------------------------------------
'''
Linear measurement update for robust filter [4]
Params:
P- = U * D * U^T old state covariance
x- -             old state mean
H  - decorrelated observtion mtx
z  - decorrelated observation
d_r - decorrelated noise covariance as vector R = u_r * diag(d_r) * u_r^T
return:
x+ -             new state mean
P+ = U * D * U^T new state covariance 
'''
def _correct_robust_scalar(x, u, d, h, z, r, psi_func, psidot_func):
    X,U,D,_,_,_ = _intra_correct_robust_scalar(x, u, d, h, z, r, psi_func, psidot_func)
    return X,U,D
    
#------------------------------------------------------------------------------
class robust_lin(_srkf_ra_lin):
    '''
    Standard SRKF in UD form [4]
    '''
    def __init__(self, x, P, Q, R, psi_func = _srkf_ra_lin._default_psi, psidot_func = _srkf_ra_lin._default_psidot):
        _srkf_ra_lin.__init__(self, x, P, Q, R, psi_func, psidot_func, _correct_robust_scalar)
#------------------------------------------------------------------------------
'''
Linear measurement update for adaptive robust filter [4,5]
WARNNG: This thing was not properly implemented due to possible typing errors 
or incomplete information in [4]!!!
Params:
P- = U * D * U^T old state covariance
x- -             old state mean
H  - decorrelated observtion mtx
z  - decorrelated observation
d_r - decorrelated noise covariance as vector R = u_r * diag(d_r) * u_r^T
return:
x+ -             new state mean
P+ = U * D * U^T new state covariance 
'''

_mu12 = 1.0 + 3.0*np.sqrt(2) # mu1^2 

def _correct_adaptive_robust_scalar(x, u, d, h, z, r, psi_func, psidot_func):
    x_j,u_j,d_j,k_j,psi_j,psidot_j = _intra_correct_robust_scalar(x, u, d, h, z, r, psi_func, psidot_func)
    
    #Residual
    nu     = z - h.dot(x_j)
    #Normalized residual
    beta   = nu / r
    #Influence function
    psi    =    psi_func(beta)
    psidot = psidot_func(beta)
    #Noise dispersion
    disp  = r * r
    #Weighted residual
    eta   = psi * r
    
    f_j = h.dot(u_j)
    v_j = d_j * f_j
    e = f_j.dot(v_j) + disp
    
    #Check for filter fail
    dg = eta * eta - e * _mu12
    
    if dg < 0:
        # Did not fail, may return robust update result
        U,D = u_j, d_j
        X = x_j
    else:    
        #Filter has failed to converge, now we must do adaptive update
        #Corrected h
        h_hat = h - h.dot(k_j) * h
    
        #Now do adaptive Joseph-like update
        f = h_hat.dot(u_j)
        v = d_j * f #f^t used 

        #Conmpute gamma^2
        c = h.dot(u_j.dot(v))
        b = f.dot(v)
        
        g2 = psidot * (c * c * _mu12 + b * dg) / (dg * disp)
        
        a = g2 * disp - b * psidot_j
        
        K = u_j.dot(v / a)
        #Compute U D adaptive update
        n = d_j.shape[0]
        WW = np.concatenate((u_j, K.reshape((n,1))), axis = 1)
        DD = np.concatenate((d_j, np.array([a * psidot])))
        
        U,D = mwgs(WW, DD)
        
        #Correct state
        X = x + (U.dot(D*U)).dot(h) * nu / disp
        #X = x_j
    
    return X,U,D
#------------------------------------------------------------------------------
'''
WARNNG: This thing was not properly implemented due to possible typing errors 
or incomplete information in [4]!!!
'''
class adaptive_robust_lin(_srkf_ra_lin):
    
    def __init__(self, x, P, Q, R, psi_func = _srkf_ra_lin._default_psi, psidot_func = _srkf_ra_lin._default_psidot):
        _srkf_ra_lin.__init__(self, x, P, Q, R, psi_func, psidot_func, _correct_adaptive_robust_scalar)
#------------------------------------------------------------------------------
'''
Linear measurement update for adaptive filter [5]
Params:
P- = U * D * U^T old state covariance
x- -             old state mean
H  - decorrelated observtion mtx
z  - decorrelated observation
d_r - decorrelated noise covariance as vector R = u_r * diag(d_r) * u_r^T
return:
P+ = U * D * U^T new state covariance 
x+ -             new state mean
'''
def _ada_scalar_correct(x, u, d, h, z, r):
    #Josef update
    u_j, d_j, K_j = _joseph_scalar_correct(u, d, h, r)
    
    #Innovation
    nu = z - h.dot(x)
    
    x_j = x + K_j * nu
    
    #residual
    eta = z - h.dot(x_j)
    
    f_j = h.dot(u_j)
    v_j = d_j * f_j
    e = f_j.dot(v_j) + r
    
    #Check for filter fail
    dg = eta * eta - e * _mu12
    
    if dg >= 0:    
        #Adaptive correction
        h_hat = h - h.dot(K_j) * h
        
        #
        f = h_hat.dot(u_j)
        v = d_j * f #f^t used 

        #Conmpute q
        c = h.dot(u_j.dot(v))
        
        q = dg / (c * c * _mu12)
        
        K = u_j.dot(v)
        
        #Compute U D adaptive update
        n = d_j.shape[0]
        WW = np.concatenate((u_j, K.reshape((n,1))), axis = 1)
        DD = np.concatenate((d_j, np.array([q])))
        
        U,D = mwgs(WW, DD)
        X = x + (U.dot(D*U)).dot(h) / r * nu
    else:
        U,D = u_j,d_j
        X = x_j
        
    return X,U,D    
        
def _ada_correct_lin(x, u, d, H, z, d_r):
    n = z.shape[0]
    X = x
    U = u
    D = d
    for i in range(0, n):
        X,U,D = _ada_scalar_correct(X, U, D, H[i], z[i], d_r[i])
        
    return X, U, D
#------------------------------------------------------------------------------
'''
Adaptive square root Kalman filter [5]
'''
class adaptive_lin(_srkf_base):
    
    def __init__(self, x, P, Q, R):   
        #base init
        _srkf_base.__init__(self, x, P, Q, R)
        
    def run(self, F, B, H, z, bf = None, bh = None):
        #Constructing default biases
        if bh is None:
            bh = np.zeros(z.shape)
            
        if bf is None:
            bf = np.zeros(self.x.shape)
        #Decorrelate observations    
        z   = self.dm.dot(z - bh)
        Hd  = self.dm.dot(H)

        #Time update/Predict
        self.x, self.U, self.D = _predict_lin(self.x, self.U, self.D, F, bf, B, self.Uq, self.Dq)
        #Measu_reent update/Correct
        self.x, self.U, self.D = _ada_correct_lin(self.x, self.U, self.D, Hd, z, self.d_r)
        #return corrected observation
        return H.dot(self.x) + bh
