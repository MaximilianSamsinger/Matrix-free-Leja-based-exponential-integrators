from __future__ import print_function
import numpy as np
from scipy.sparse import identity, issparse
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.sparse.linalg.interface import IdentityOperator
from expleja import expleja, normAmp, powerit
import scipy.io as sio

#functionEvaluations ~= Matrix-Vector Multiplications
#s... substeps

class Integrator:
    ''' Situation: dt u(t) = M @ u  
    Calculate u(t_end), assuming M is a matrix or LinearOperator
    '''
    def __init__(self, method, reference_solution, target_errors, columns):
        def solve(M, u, t, t_end, s):
            u, functionEvaluations, otherCosts = method(M, u, t, t_end, s)
            
            abs_error = np.linalg.norm(u - reference_solution, 2)
            rel_error = abs_error / np.linalg.norm(u, 2)
            return u, rel_error, (functionEvaluations, otherCosts)
        
        self.solve = solve
        self.method = method
        
        self.name = method.__name__
        assert(self.name in ['cn2','exprb2','rk2','rk4',])
        
        self.columns = columns.copy()
        if self.name in ['rk2','rk4']:
            self.target_errors = [min(target_errors)]
        else:
            self.target_errors = target_errors
            self.columns += ['target_error']
        

def rk2(M, u, t, t_end, s):
    ''' Midpoint rule '''
    tau = (t_end-t)/s # time step size
    for _ in range(s):
        k1 = M @ u
        k2 = M @ (u+tau/2.*k1)
        
        t,u = t + tau, u + tau*k2
    
    assert(abs(t - t_end) < tau)

    functionEvaluations = 2*s         
    otherCosts = 0
    
    return u, functionEvaluations, otherCosts

def rk4(M, u, t, t_end, s):
    ''' Classical Runge-Kutta method '''
    tau = (t_end-t)/s # time step size
    for _ in range(s):
        k1 = M @ u
        k2 = M @ (u+tau/2.*k1)
        k3 = M @ (u+tau/2.*k2)
        k4 = M @ (u+tau*k3)
        
        t,u = t + tau, u + tau/6*(2*k1 + k2 + k3 + 2*k4)
     
    assert(abs(t - t_end) < tau)
        
    functionEvaluations = 4*s
    otherCosts = 0
    
    return u, functionEvaluations, otherCosts

def cn2(M, u, t, t_end, s):
    ''' Crank-Nicolson method 
    The choice cn2.tol/s in gmres guarantees that the compounding error 
    is low enough'''
    
    assert(cn2.tol is not None)   
    
    ushape = u.shape
    N = ushape[0]
    
    tau = (t_end-t)/s # time step size
    
    def gmresiterations(rk):
        gmresiterations.counter += 1
    gmresiterations.counter = 0
    
    A = identity(N) - tau/2.*M
    B = identity(N) + tau/2.*M
    
    for _ in range(s):
        #Solve A*u = b 
        b = B @ u # Counts as another function evaluation (Matrix-Vector-Mult)
        u,_ = gmres(A, b, x0=u, tol=cn2.tol/s, callback=gmresiterations,
                    M = None, atol=0)
        t += tau

    assert(abs(t - t_end) < tau)
    
    u = u.reshape(ushape)
    '''
    Cost of gmres per iteration:
        1 matrix-vector product
        some floating point operations (O(m*n))
    '''
    
    functionEvaluations = s + gmresiterations.counter # Number of mv-products
    otherCosts = gmresiterations.counter # Used to calculate remaining costs.
    
    return u, functionEvaluations, otherCosts


def exprb2(M, u, t, t_end, s):
    ''' We force s = s and m = 99 for Experiment1 '''
    assert(exprb2.tol is not None)
    tol = exprb2.tol
    
    tau = (t_end-t)/s
    
    para = select_interp_para_for_fixed_m_and_s(tau, M, u, tol, s=s, m=99)
    
    expAv, _, info, c, m, _, _ = expleja(t_end-t, M, u, tol=tol, p=0, 
                                         interp_para=para)
    
    functionEvaluations = int(sum(info) + c)
    otherCosts = 0
    
    return expAv, functionEvaluations, otherCosts
    
def select_interp_para_for_fixed_m_and_s(h, A, v, tol, s=1, m=99):    
    ''' 
    The code is shortened version select_interp_para from expleja 
    and forces 
        a fixed interpolation degree m,
        a fixed number of substeps s,
        no hump reduction.
    We assume a sparse matrix A as input. 
    '''
    
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
    #######
    if isinstance(A,LinearOperator):
        '''
        Selects the shift mu (Half of (absolutely) largest eigenvalue)
        ''' 
        [mu,c] = powerit(A) # Poweriteration to estimate the 2-norm of A
        mu = -mu/2.0
        A = A - mu*IdentityOperator((n,n))
        gamma2 = abs(mu)
        assert(isinstance(A,LinearOperator))
    
    else:
        '''
        Selects the shift mu = trace(A)/n
        '''
        mu = sum(A.diagonal())/float(n)
        
        A = A-mu*identity(n)
        if not issparse(A):
            A = np.asarray(A)
        [gamma2,c] = normAmp(A,1,tol[3])

    
    ######
    '''
    Selects the shift mu = trace(A)/n
    '''
    mu = sum(A.diagonal())/float(n)
    
    A = A-mu*identity(n) # We assume, we work with sparse matrices
    [gamma2,c] = normAmp(A,1,tol[3]) # This introduces extra mv costs
       
    gamma2, dd = theta[m], dd[:,m]
    xi = xi*(gamma2/2)
    
    return s, gamma2, xi.flatten(), dd, A, mu, c, m 