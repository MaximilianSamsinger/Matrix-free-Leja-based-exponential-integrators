from __future__ import print_function
import numpy as np
from scipy.sparse import identity, issparse
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.sparse.linalg.interface import IdentityOperator
from expleja import expleja, normAmp
import scipy.io as sio

#functionEvaluations ~= Matrix-Vector Multiplications
#s... substeps

class Integrator:
    ''' Situation: dt u(t) = M @ u
    Calculate u(t_end), assuming M is a matrix or LinearOperator
    '''
    def __init__(self, method, inputs, reference_solution, target_errors,
                 estKwargsList, columns):
        def solve(s):
            u, functionEvaluations, misc = method(*inputs, s)

            abs_error = np.linalg.norm(u - reference_solution, 2)
            rel_error = abs_error / np.linalg.norm(u, 2)
            return u, rel_error, (functionEvaluations, misc)

        self.solve = solve
        self.method = method

        self.name = name = method.__name__

        assert(name in ['cn2','exprb2','rk2','rk4',])

        ''' Adjust class variables based on the method'''
        self.columns = columns.copy()
        if name in ['rk2','rk4']:
            self.target_errors = [min(target_errors)]
        else:
            self.target_errors = target_errors
            self.columns += ['target_error']
        if name == 'exprb2' and estKwargsList != [{}]:
            self.columns += ['safetyfactor']

        self.estKwargsList = estKwargsList if name=='exprb2' else [{}]

def rk2(M, u, t, t_end, s):
    ''' Midpoint rule '''
    τ = (t_end-t)/s # time step size
    for _ in range(s):
        k1 = M @ u
        k2 = M @ (u+τ/2.*k1)

        t,u = t + τ, u + τ*k2

    assert(abs(t - t_end) < τ)

    functionEvaluations = 2*s
    misc = 0

    return u, functionEvaluations, misc

def rk4(M, u, t, t_end, s):
    ''' Classical Runge-Kutta method '''
    τ = (t_end-t)/s # time step size
    for _ in range(s):
        k1 = M @ u
        k2 = M @ (u+τ/2.*k1)
        k3 = M @ (u+τ/2.*k2)
        k4 = M @ (u+τ*k3)

        t,u = t + τ, u + τ/6*(2*k1 + k2 + k3 + 2*k4)

    assert(abs(t - t_end) < τ)

    functionEvaluations = 4*s
    misc = 0

    return u, functionEvaluations, misc

def cn2(M, u, t, t_end, s):
    ''' Crank-Nicolson method
    The choice cn2.tol/s in gmres guarantees that the compounding error
    is low enough'''

    assert(cn2.tol is not None)

    ushape = u.shape
    N = ushape[0]

    τ = (t_end-t)/s # time step size

    def gmresiterations(rk):
        gmresiterations.counter += 1
    gmresiterations.counter = 0

    Id = (IdentityOperator((N,N)) if isinstance(M,LinearOperator)
          else identity(N)) # Appropriate Identity Opterator given M

    A = Id - τ/2.*M
    B = Id + τ/2.*M

    for _ in range(s):
        #Solve A*u = b
        b = B @ u # Counts as another function evaluation (Matrix-Vector-Mult)
        u,_ = gmres(A, b, x0=u, tol=cn2.tol/s, callback=gmresiterations,
                    M = None, atol=0)
        t += τ

    assert(abs(t - t_end) < τ)

    u = u.reshape(ushape)
    '''
    Cost of gmres per iteration:
        1 matrix-vector product
        some floating point operations (O(m*n))
    '''

    functionEvaluations = s + gmresiterations.counter # Number of mv-products
    misc = gmresiterations.counter # Used to calculate remaining costs.

    return u, functionEvaluations, misc


def exprb2(M, u, t, t_end, s):
    ''' We force s = s and m = 99 for Experiment1 '''
    assert(exprb2.tol is not None)
    if isinstance(M,LinearOperator):
        assert(bool(exprb2.estKwargs)) #Asserts dictionart is not empty
    else:
        assert(not bool(exprb2.estKwargs)) #Asserts dictionart is empty

    tol = exprb2.tol
    τ = (t_end-t)/s

    para = select_interp_para_for_fixed_m_and_s(τ, M, u, tol, s=s,
                                                estKwargs = exprb2.estKwargs)

    expAv, _, info, c, m, _, _ = expleja(t_end-t, M, u, tol=tol, p=0,
                                         interp_para=para)

    functionEvaluations = int(sum(info) + c)
    misc = c # Number of power iterations

    return expAv, functionEvaluations, misc


def largestEV(A, powerits=100, safetyfactor=1.1):
    n = A.shape[0]
    x = np.random.randn(n,1)
    λ = float('inf')

    for mv in range(1,powerits+1):
        λ_old, λ = λ, np.linalg.norm(x)
        y = x/λ
        x = A.dot(y)

    return safetyfactor*λ, mv

def select_interp_para_for_fixed_m_and_s(h, A, v, tol, s=1, m=99,
                                         normEstimator = largestEV,
                                         estKwargs = {"powerits":100}):
    '''
    The code is shortened version select_interp_para from expleja
    and forces
        a fixed interpolation degree m,
        a fixed number of substeps s,
        no hump reduction.
    '''
    n = v.shape[0]

    '''
    Load interpolation parameter depending on chosen precision
    '''
    sampletol = [2**-10,2**-24,2**-53]

    t = max(tol[0],2**-53) if len(tol) == 1 else max(min(tol[0],tol[1]),2**-53)
    if t >= sampletol[0]:
        data = sio.loadmat('data_leja_half_u.mat')
    elif t >= sampletol[1]:
        data = sio.loadmat('data_leja_single_u.mat')
    else:
        data = sio.loadmat('data_leja_double_u.mat')

    θ, ξ, dd = data['theta'], data['xi'], data['dd']

    if isinstance(A,LinearOperator):
        '''
        Selects the shift μ (Half of (absolutely) largest eigenvalue)
        '''
        [μ, mv] = normEstimator(A, **estKwargs) # Estimates 2-norm of A
        μ = -μ/2.0
        A -= μ*IdentityOperator((n,n))
        γ2 = abs(μ)

    else:
        '''
        Selects the shift μ = trace(A)/n
        '''
        μ = sum(A.diagonal())/float(n)
        A -= μ*identity(n)

        if not issparse(A):
            A = np.asarray(A)
        γ2, mv = normAmp(A,1,tol[3])

    γ2, dd = θ[m], dd[:,m]
    ξ = ξ*(γ2/2)

    return s, γ2, ξ.flatten(), dd, A, μ, mv, m