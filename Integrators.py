from __future__ import print_function
import numpy as np
from scipy.sparse import identity, issparse
from scipy.sparse.linalg import gmres
from numpy import array, asarray, transpose
from expleja import expleja, normAmp
import scipy.io as sio
import warnings

#functionEvaluations ~= Matrix-Vector Multiplications
class MatrixIntegrator:
    ''' Situation dt u(t) = f(t,u)  
    Calculate u(t_end), assuming f is a matrix-vector mult f(t,u) = M @ u
    and estimate the error
    '''
    def __init__(self, name, method, Ntlist, exact_solution):
        def solve(M, t , u, t_end, Nt):
            u, functionEvaluations, otherCosts = method(M, t, u, t_end, Nt)
            error = np.linalg.norm(u - exact_solution)
            return u, error, functionEvaluations, otherCosts
        
        self.solve = solve
        self.method = method
        self.name = name

def rk2_matrixinput(M,t,u,t_end,Nt):
    ''' Midpoint rule '''
    tau = (t_end-t)/Nt # time step size
    for _ in range(Nt):
        k1 = M @ u
        k2 = M @ (u+tau/2.*k1)
        
        t,u = t + tau, u + tau*k2
    
    assert(abs(t - t_end) < 1e-14)

    functionEvaluations = 2*Nt         
    otherCosts = 0
    
    return u, functionEvaluations, otherCosts

def rk4_matrixinput(M,t,u,t_end,Nt):
    ''' Classical Runge-Kutta method '''
    tau = (t_end-t)/Nt # time step size
    for _ in range(Nt):
        k1 = M @ u
        k2 = M @ (u+tau/2.*k1)
        k3 = M @ (u+tau/2.*k2)
        k4 = M @ (u+tau*k3)
        
        t,u = t + tau, u + tau/6*(2*k1 + k2 + k3 + 2*k4)
     
    assert(abs(t - t_end) < 1e-14)
        
    functionEvaluations = 4*Nt
    otherCosts = 0
    
    return u, functionEvaluations, otherCosts
        

def crankn(M,t,u,t_end,Nt):
    
    ushape = u.shape
    N = ushape[0]
    
    tau = (t_end-t)/Nt # time step size
    
    functionEvaluations = 0
    def gmresiterations(rk):
        gmresiterations.counter += 1
    gmresiterations.counter = 0

    for _ in range(Nt):
        #Solve A*u = b 
        A = identity(N) - tau/2.*M
        b = identity(N) + tau/2.*M
        b = b @ u # Counts as another function evaluation (Matrix-Vector-Mult)
        '''
        Why is this here? For LinearOperators maybe?
        if not issparse(A):
            A = asarray(A)
        '''
        u,_ = gmres(A,b,x0=u,tol=2**-23,callback=gmresiterations)
        t += tau

    
    assert(abs(t - t_end) < 1e-14)
    
    u = u.reshape(ushape)
    '''
    Cost of gmres per iteration:
        1 matrix-vector product
        some floating point operations (O(m*n))
    '''
    
    functionEvaluations = Nt + gmresiterations.counter # Number of mv-products
    otherCosts = gmresiterations.counter # Used to calculate remaining costs.
    
    return u, functionEvaluations, otherCosts

def expeuler(M,t,u,t_end,Nt):
    ''' We force s = Nt and m = 99 for Experiment1 '''
    
    tau = (t_end-t)/Nt
    
    para = select_interp_para_for_fixed_m_and_s(tau, M, u, s=Nt, m=99)
    
    expAv, _, info, c, m, _, _ = expleja(t_end-t, M, u, 
            tol=[0,2**-23,float('inf'),float('inf')], p=0, interp_para=para)
    
    functionEvaluations = sum(info)[0] + c
    otherCosts = 0
    
    return expAv, functionEvaluations, otherCosts
    
    ''' TODO: Runtime warnings kill our performance and the result is unusable 
    anyway. We catch them as errors and skip the result '''
    
def select_interp_para_for_fixed_m_and_s(h, A, v, s=1, m=99):    
    ''' The code is shortened version select_interp_para from expleja 
    and forces 
        a fixed interpolation degree m,
        a fixed number of substeps s,
        no hump reduction,
        32 bit precision,
    and we assume a sparse matrix A as input.
    However, we still shift the matrix A '''
    
    tol = [0,2**-23,float('inf'),float('inf')]
    data = sio.loadmat('data_leja_single_u.mat')

    theta = data['theta']
    xi = data['xi']
    dd = data['dd']
    n = v.shape[0]
    
    #Selects the shift mu = trace(A)/n
    mu = sum(A.diagonal())/float(n)
    
    A = A-mu*identity(n) # We assume, we work with sparse matrices
    [gamma2,c] = normAmp(A,1,tol[3])
       
    gamma2 = theta[m]
    dd = dd[:,m]
    xi = xi*(gamma2/2)
    return s, gamma2, xi.flatten(), dd, A, mu, c, m 
    

def rk2(f,t,u,t_end,Nt):
    ''' Midpoint rule '''
    tau = (t_end-t)/Nt # time step size
    for _ in range(Nt):
        k1 = f(t,u)
        k2 = f(t+tau/2., u+tau/2.*k1)
        
        t,u = t + tau, u + tau*k2
    
    assert(abs(t - t_end) < 1e-14)

    functionEvaluations = 2*Nt         
    return u, functionEvaluations 
        
def rk4(f,t,u,t_end,Nt):
    ''' Classical Runge-Kutta method '''
    tau = (t_end-t)/Nt # time step size
    for _ in range(Nt):
        k1 = f(t,u)
        k2 = f(t+tau/2., u+tau/2.*k1)
        k3 = f(t+tau/2., u+tau/2.*k2)
        k4 = f(t+tau, u+tau*k3)
        
        t,u = t + tau, u + tau/6*(2*k1 + k2 + k3 + 2*k4)
        
    functionEvaluations = 4*Nt
    return u, functionEvaluations
