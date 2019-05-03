from __future__ import print_function
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import gmres
from expleja import expleja, normAmp
import scipy.io as sio

#functionEvaluations ~= Matrix-Vector Multiplications
class MatrixIntegrator:
    ''' Situation dt u(t) = f(t,u)  
    Calculate u(t_end), assuming f is a matrix-vector mult f(t,u) = M @ u
    and estimate the error
    '''
    def __init__(self, name, method, exact_solution):
        def solve(M, t , u, t_end, Nt):
            u, functionEvaluations, otherCosts = method(M, t, u, t_end, Nt)
            
            abs_error_2 = np.linalg.norm(u - exact_solution, 2)
            rel_error_2 = abs_error_2/np.linalg.norm(u, 2)
            abs_error_inf = np.linalg.norm(u - exact_solution, float('inf'))
            rel_error_inf = abs_error_inf/np.linalg.norm(u, float('inf'))
            errors = (abs_error_2, rel_error_2, abs_error_inf, rel_error_inf)
            return u, errors, (functionEvaluations, otherCosts)
        
        self.name = name
        self.solve = solve
        
        method.parameter = None # Relevant for gmres in crankn
        self.method = method
        

def rk2_matrixinput(M,t,u,t_end,Nt):
    ''' Midpoint rule '''
    tau = (t_end-t)/Nt # time step size
    for _ in range(Nt):
        k1 = M @ u
        k2 = M @ (u+tau/2.*k1)
        
        t,u = t + tau, u + tau*k2
    
    assert(abs(t - t_end) < tau)

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
     
    assert(abs(t - t_end) < tau)
        
    functionEvaluations = 4*Nt
    otherCosts = 0
    
    return u, functionEvaluations, otherCosts
        

def crankn(M,t,u,t_end,Nt, usepreconditioner=False):
    ''' Crank-Nicolson method 
    Optimal gmres tolerance is unclear a priori'''
    
    if crankn.parameter is None:
        crankn.parameter = 2**-23
        
    gmres_tol = crankn.parameter
    
    
    ushape = u.shape
    N = ushape[0]
    
    tau = (t_end-t)/Nt # time step size
    
    def gmresiterations(rk):
        gmresiterations.counter += 1
    gmresiterations.counter = 0
    
    A = identity(N) - tau/2.*M
    B = identity(N) + tau/2.*M
    
    if usepreconditioner:
        import scipy.sparse.linalg as spla
        M_x = lambda x: spla.spsolve(A, x)
        preconditioner = spla.LinearOperator((N, N), M_x)
    else:
        preconditioner = None
    
    for _ in range(Nt):
        #Solve A*u = b 
        b = B @ u # Counts as another function evaluation (Matrix-Vector-Mult)
        u,_ = gmres(A, b, x0=u, tol=gmres_tol, callback=gmresiterations,
                    M = preconditioner)
        t += tau

    assert(abs(t - t_end) < tau)
    
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
    
    functionEvaluations = int(sum(info)[0] + c)
    otherCosts = 0
    
    return expAv, functionEvaluations, otherCosts
    
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
    
    assert(abs(t - t_end) < tau)

    functionEvaluations = 2*Nt    
    otherCosts = 0
     
    return u, functionEvaluations, otherCosts
        
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
    otherCosts = 0
    
    return u, functionEvaluations, otherCosts
