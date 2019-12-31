from __future__ import print_function, division

import numpy as np
from scipy.sparse import identity, issparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg.interface import IdentityOperator
from scipy.sparse.linalg._onenormest import _onenormest_core
from scipy.sparse.linalg.matfuncs import MatrixPowerOperator
from scipy.linalg.blas import dasum, dnrm2

import scipy.io as sio

def expleja(h, A, v, tol=[0,2**-53,float('inf'),float('inf')],
                          p=5, interp_para=None):
    '''
    EXPLEJA   Matrix exponential times vector or matrix.

      EXPLEJA(H,A,V,TOL=[0,2**-53,FLOAT('INF'),FLOAT('INF')],P=5,INTERP_PARA=0)

      RETURN EXPAV, ERREST, INFO, C, M, MU, GAMMA2


      Compute EXPM(H * A) * V without forming EXP(H*A). A is a (N, N) ndarray
      or *preferably* a sparse matrix. V is a 2-dimensional ndarray.
      If len(TOL) == 1, then TOL is the absolute tolerance. If len(TOL) == 2,
      then TOL[0] is the absolute tolerance and TOL[1] the tolerance relative
      to the current approximation of EXPM(H * A)*V.
      If len(TOL) == 3, then TOL[2] specifies the norm, by default this is
      float('inf'). If len(TOL) == 4, then TOL[3] specifies the operator norm,
      by default this is float('inf'). If nothing is provided TOL = [0, 2**-53,
      float('inf'), float('inf')].
      By P one can specify the maximal power of A that is used for the
      hump reduction procedure. The default is P=5.
      On output, sum(INFO) is the total number of iterations
      (== matrix-vector products) and ERREST(j) the estimated error in step j
      (in the specified norm). C are the auxillary matrix vector products
      needed for the norm estimates, M is the selected degree of
      interpolation, MU the used shift and GAMMA2
      corresponds to the selected interpolation interval.

      The code is based on PHILEJA provided by Marco Caliari.

      Reference: M. Caliari, P. Kandolf, A. Ostermann and S. Rainer,
      The Leja method revisited: backward error analysis for the matrix
      exponential, submitted, http://arxiv.org/abs/1506.08665

      Python implementation:
      Maximilian Samsinger, April 4, 2016

      Original Matlab implementation:
      Peter Kandolf, July 8, 2015, cooperation with Marco Caliari,
      Alexander Ostermann and Stefan Rainer

      Additional notes:
      - The error is estimated during the newton interpolation, which is
      performed until
       ||errorestimate||_TOL[3] > max(TOL[1],TOL[2]*||Y||_TOL[3])
      is satisfied.

      Minimal example:
      import numpy as np
      A = np.diag(range(-10,1))+np.diag(np.ones(10),1)
      v = np.ones((11,1))
      h = 1
      y = expleja(h,A,v)

      Minimal example with select_interp_para:
      import numpy as np
      n = 10
      A = np.diag(range(-n,1))+np.diag(np.ones(n),1)
      v = np.ones((n+1,1))
      h = 1.0
      param = select_interp_para(h, A, v)
      y = expleja(h,A,v,interp_para=param)
    '''

    '''
    Check consistency of matrix and vector sizes
    '''
    n = v.shape[0]
    if A.shape != (n,n):
        print('Shape of matrix', A.shape)
        print('Shape of vector', v.shape)
        raise ValueError('Inconsistent matrix and vector sizes')


    '''
    Check trivial cases
    '''
    if not v.any(): #v is the zero vector
        return v, np.array([[0]]), np.array([[0]]), 0, 0, 0, np.array([0])
    if h == 0:
        return v, np.array([[0]]), np.array([[0]]), 0, 0, 0, np.array([0])
    if issparse(A):
        if not (A.nonzero()[0]).size: # A (sparse) has no non-zero entry
            return v, np.array([[0]]), np.array([[0]]), 0, 0, 0, np.array([0])
    elif not isinstance(A,LinearOperator):
        if not A.any(): # A (array) has no non-zero entry
            return v, np.array([[0]]), np.array([[0]]), 0, 0, 0, np.array([0])

    '''
    Set the default value to all tolerances which have not been given
    '''
    defaulttol = [0,2**-53,float('inf'),float('inf')]
    tol = tol + defaulttol[len(tol):]

    if interp_para is None:
        interp_para = select_interp_para(h, A, v, tol, m_max=99, p=p)

    return newton_wrapper(h, v, tol, *interp_para, vectornorm = tol[2])

def newton_wrapper(h, v, tol, nsteps, gamma2, xi, dd, A, mu, c, m,
                   vectornorm = float('inf')):
    '''
    Set norm for the newton interpolation
    '''
    if vectornorm == 2:
        norm = dnrm2
    elif vectornorm == 1:
        norm = dasum
    elif vectornorm == float('inf'):
        if np.isrealobj(v) and np.isrealobj(v):
            def norm(a):
                return max(abs(a.min()),abs(a.max())) #Faster than code below
        else:
            def norm(a):
                return abs(a).max() #Works for complex numbers
    else:
        raise ValueError('Specified norm is not implemented')

    expAv = v
    errest = np.zeros(int(nsteps))
    info = np.zeros(int(nsteps))

    for k in range(int(nsteps)):
        pexpAv, errest[k], info[k] = newton(h/nsteps, A, expAv, xi, dd,
                      tol[0]/nsteps, tol[1]/nsteps, norm=norm, m_max=m)
        expAv = pexpAv * np.exp(mu*h/float(nsteps))  # Compensate for shifting
    return expAv, errest, info, c, m, mu, gamma2


'''
    SUBFUNCTIONS

      SELECT_INTERP_PARA(H, A, V, TOL, M_MAX, P)

      RETURN NSTEPS, GAMMA2, XI, DD, A, MU, C, M


      Computes the parameters for the interpolation, input is the time step H
      the matrix A and vector V. TOL is used to select the correct parameters.
      M_MAX specifies the maximal degree of interpolation (should be <=99) and
      P the maximal power of A used to extimate the spectral radius.

      The output is the number of substeps NSTEPS, the interpolation interval
      GAMMA2, the scaled interpolation points XI as well as the divided
      differences DD. The (shifted) matrix A, the shift MU and the performed
      matrix vector products MV are returned as well. M is the selected
      maximal degree of interpolation.
'''
def select_interp_para(h, A, v, tol=[0,2**-53,float('inf'),float('inf')],
                                     m_max=99, p=5):

    '''
    Load interpolation parameter depending on chosen precision
    '''
    sampletol = [2**-10,2**-24,2**-53]

    if len(tol) == 1:
        t = max(tol[0],2**-53)
    else:
        t = max(min(tol[0],tol[1]),2**-53)

    if t>=sampletol[0]:
        data = sio.loadmat('data_leja_half_u.mat')
    elif t>= sampletol[1]:
        data = sio.loadmat('data_leja_single_u.mat')
    else:
        data = sio.loadmat('data_leja_double_u.mat')


    theta, xi, dd = data['theta'], data['xi'], data['dd']

    n = v.shape[0]

    if isinstance(A,LinearOperator):
        '''
        Selects the shift mu (Half of (absolutely) largest eigenvalue)
        '''
        [mu,_,mv] = largestEV(A) # Poweriteration to estimate the 2-norm of A
        mu = -mu/2.0
        A = A - mu*IdentityOperator((n,n))
        gamma2 = abs(mu)

    else:
        '''
        Selects the shift mu = trace(A)/n
        '''
        mu = sum(A.diagonal())/float(n)

        A = A - mu*identity(n)
        if not issparse(A):
            A = np.asarray(A)

        '''
        Estimate of the spectral radius
        '''
        dest = float('inf')*np.ones(p+1)
        dest[0], mv = normAmp(A,1,tol[3])

        for i in range(p): # Hump reduction procedure for p>0
            dest[i+1], costs = normAmp(A,i+2,tol[3])
            mv += costs
            if dest[i]/dest[i+1] < 1.01: # Check if estimate is good enough
                break
        gamma2 = min(dest)


    mm = min(m_max, len(theta))
    if gamma2 == 0: #Prevents division by 0
        m = 0
    else:
        m = np.argmin(np.arange(2,mm) * np.ceil((h*gamma2)/theta[2:mm]).T)
        m = m+2
    nsteps = int(np.ceil((h*gamma2)/theta[m]))

    gamma2, dd = theta[m], dd[:,m]
    xi = xi*(gamma2/2)

    return nsteps, gamma2, xi.flatten(), dd, A, mu, mv, m


def newton(h, A, v, xi, dd, abstol, reltol, norm, m_max):
    '''
    Newton

    Compute the Newton interpolation polynomial in real Leja points for the
    matrix function specified with the divided differences DD applied to the
    right hand side V of the operator A*H*V as Y=P_m(H*A)V.

    The result is stored in Y, the estimated error in NORMERREST and the
    number of steps in M.
    '''

    y = ydiff = dd[0]*v
    new_ydiff_norm = norm(ydiff)

    for m in range(m_max):
        v = (A.dot(v))*h - xi[m]*v

        ydiff = dd[m+1]*v
        y += ydiff
        old_ydiff_norm, new_ydiff_norm = new_ydiff_norm, norm(ydiff)

        normerrest = new_ydiff_norm + old_ydiff_norm
        if normerrest < reltol * norm(y) or normerrest < abstol:
            break
    return y, normerrest, m+1

#Subfunction for handlying the norms
def normAmp(A, m, p):
    #Adds the property .H to the MatrixPowerOperator
    #.H transposes and conjugates the matrix
    MatrixPowerOperator.H = property(
            lambda self: MatrixPowerOperator(self._A.conj().T, self._p))
    A = MatrixPowerOperator(A,m)
    if p == 1:
        c,_,_,mv,_ = _onenormest_core(A, A.T, t=2, itmax=5)   #ERROR WHEN A IS 2x2 (sparse?) MATRIX OR SMALLER
        return c**(1./m), mv*m
    elif p == float('inf'):
        c,_,_,mv,_ = _onenormest_core(A.T, A, t=2, itmax=5)   #ERROR WHEN A IS 2x2 (sparse?) MATRIX OR SMALLER
        return c**(1./m), mv*m
    elif p == 2:
        c,mv = normest2(A,m)
        return c, mv*m
    else:
        print('Error: p not equal to 1, 2 or inf')



def normest2(A, p=1, t=3, itermax=100, tol=1e-2):
    '''
    Created:       10.04.2016 by Maximilian Samsinger
    Last edit:     10.04.2016 by Maximilian Samsinger
    Version:       1.0 / 0.1 (Matlab)
    Author:        Marco Caliari
    Remarks:       Original (Matlab) code created by Peter Kandolf

    normest2(A, p=1, t=3, itermax=100, tol=1e-3)

    return c, mv


    estimate ||A^p||_2^(1/p), by at most itermax matrix A times t-column
    matrix products and itermax matrix A' times t-column matrix products.
    Actual used matrix times t-columns products are returned in mv.
    '''

    n = A.shape[0]
    y = np.random.randn(n,t)
    c_old = float('inf')

    for k in range(itermax):
        x = y/(abs(y).sum(axis=0)[None,:])
        y = A.dot(x)
        for j in range(1,p):
            y = A.dot(y)
        y = A.T.dot(y)
        for j in range(1,p):
            y = A.H.dot(y)
        c = max(abs(y).sum(axis=0))**(1.0/(2.*p))
        if abs(c-c_old)<tol*c or c==0:
            break
        c_old = c

    mv = 2*p*k*t
    return c, mv

def largestEV(A, powerits=100, tol=1e-2):
    ''' Estimate the absolutely largest eigenvalue using power iterations '''
    n = A.shape[0]
    x = np.random.randn(n,1)
    λ = float('inf')

    for mv in range(1,powerits+1):
        λ_old, λ = λ, np.linalg.norm(x)
        y = x/λ
        x = A.dot(y)
        if abs(λ-λ_old)<tol*λ or λ==0:
            break
    return λ, y, mv