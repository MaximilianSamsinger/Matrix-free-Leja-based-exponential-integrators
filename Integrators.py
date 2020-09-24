from __future__ import print_function
import numpy as np
from scipy.sparse import identity, issparse
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.sparse.linalg.interface import IdentityOperator
from expleja import expleja, normAmp, newton_wrapper
import scipy.io as sio
from copy import deepcopy
try:
    import cupy as cp
except ModuleNotFoundError:
    pass

#functionEvaluations ~= Matrix-Vector Multiplications
#s... substeps

class Integrator:
    '''
    Situation: dt u(t) = F(u)
    Calculate u(t_end)
    '''
    def __init__(self, method, inputs, reference_solution, columns):        
        def solve(s, **kwargs):
            u, cost = method(*deepcopy(inputs),s,**kwargs)

            abs_error = np.linalg.norm(u - reference_solution, 2)
            rel_error = abs_error / np.linalg.norm(u, 2)

            return u, rel_error, cost

        self.method = method
        self.solve = solve
        self.columns = columns.copy()
        self.name = name = method.__name__
        assert(name in ['cn2','rk2','rk4'] or 'exprb' in name)

def rk2(F, u, t, t_end, linearCase, s):
    if len(u) >= 40000:
        try:
            u = cp.array(u)
        except NameError:
            pass
    
    ''' Midpoint rule '''
    τ = (t_end-t)/s # time step size
    for _ in range(s):
        k1 = F(u)
        k2 = F(u+τ/2.*k1)

        t,u = t + τ, u + τ*k2

    assert(abs(t - t_end) < τ)
    
    # Costs
    if linearCase:
        mv = 2*s
        cost = (mv,)
    else:
        functionEvaluations = 2*s
        derivativeEvaluations = 0
        mv = 0
        cost = (functionEvaluations, derivativeEvaluations, mv)

    if len(u) >= 40000:
        try:
            u = cp.asnumpy(u)
        except NameError:
            pass   
    return u, cost

def rk4(F, u, t, t_end, linearCase, s):
    # Using GPU for speedups
    if len(u) >= 40000:
        try:
            u = cp.array(u)
        except NameError:
            pass
    
    ''' Classical Runge-Kutta method '''
    τ = (t_end-t)/s # time step size
    for _ in range(s):
        k1 = F(u)
        k2 = F(u+τ/2.*k1)
        k3 = F(u+τ/2.*k2)
        k4 = F(u+τ*k3)

        t,u = t + τ, u + τ/6*(2*k1 + k2 + k3 + 2*k4)

    assert(abs(t - t_end) < τ)

    # Costs
    if linearCase:
        mv = 4*s
        cost = (mv,)
    else:
        functionEvaluations = 4*s
        derivativeEvaluations = 0
        mv = 0
        cost = (functionEvaluations, derivativeEvaluations, mv)

    if len(u) >= 40000:
        try:
            u = cp.asnumpy(u)
        except NameError:
            pass   
    return u, cost

def cn2(F, u, t, t_end, linearCase, s, tol=None, dF=None):
    ''' Crank-Nicolson method
    The choice tol/s in gmres guarantees that the compounding error
    is low enough'''
    
    mv = 0
    
    ushape = u.shape
    Nx = ushape[0]

    τ = (t_end-t)/s # time step size

    def gmresiterations(rk):
        gmresiterations.counter += 1
    gmresiterations.counter = 0

    if linearCase:
        M = F(u, returnMatrix=True) # Linear case
        Id = (IdentityOperator((Nx,Nx)) if isinstance(M,LinearOperator)
              else identity(Nx)) # Appropriate Identity Opterator given M
    
        A = Id - τ/2.*M
        B = Id + τ/2.*M
    
        for _ in range(s):
            #Solve A@u = b
            b = B @ u
            u,_ = gmres(A, b, x0=u, tol=tol/s, atol=0,
                        callback=gmresiterations, M = None)
            t += τ
    
        assert(abs(t - t_end) < τ)
    
        u = u.reshape(ushape)
        mv = s + gmresiterations.counter
    
        return u, (mv,)
    else:
        def newton(u0):
            Fu0 = F(u0)
            g  = lambda u1: u1 - u0 - τ/2*(Fu0 + F(u1))
            #dg = lambda u1: Id      - τ/2*      dF(u1)
            u1 = u0 + τ*Fu0
                    
            def dg(u1):
                dFu1 = dF(u1)
                mv = lambda v: v - τ/2*dFu1@v
                return LinearOperator((Nx,Nx), matvec = mv)         
            
            for k in range(9):
                if k%3 == 0:
                    A = dg(u1)
                Δu, _ = gmres(A,-g(u1),x0 = u1, tol=tol/s, atol=0,
                              callback=gmresiterations, M = None)
                u1 += Δu.reshape(ushape)
                if np.linalg.norm(Δu)/np.linalg.norm(u0) < tol/s:
                    break
            
            functionEvaluation = 1 + k
            derivativeEvaluations = 1 + k//3
            mv = k # Without gmres
            return u1, (functionEvaluation, derivativeEvaluations, mv)
        
        functionEvaluations = 0
        derivativeEvaluations = 0
        for _ in range(s):
            u, cost = newton(u)
            t += τ
            
            functionEvaluations += cost[0]
            derivativeEvaluations += cost[1]
            mv += cost[2]
            
        mv += gmresiterations.counter
        
        return u, (functionEvaluations, derivativeEvaluations, mv)

def exprb2(F, u, t, t_end, linearCase, s, 
           tol=None, normEstimator=None, dF=None):
    ''' Exponential Rosenbrock-Euler method
    If dF is False, then we assume the derivative of F is a constant Matrix M,
    which can be returned with M = F(u, returnMatrix=True)'''
    functionEvaluations = 0
    derivativeEvaluations = 0
    mv = 0
    
    u = u.copy()
    τ = (t_end-t)/s
    Nx = len(u)
    
    ''' Linear Case '''
    if linearCase:
        M = F(u, returnMatrix=True)
        ''' We force s = s and m = 99 '''
        para = select_interp_para_for_fixed_m_and_s(
                t_end-t, M, u, tol, s=s, normEstimator = normEstimator)

        expMu, _, info, c, m, _, _ = expleja(
                t_end-t, M, u, tol, p=0, interp_para=para)

        mv = int(sum(info) + c) # Total matrix-vector multiplications
        return expMu, (mv,)

    ''' Nonlinear Case '''
    v = np.vstack((np.zeros((Nx,1)),1))
    kwargs = {**normEstimator[1]}
    for k in range(s):
        X = LinOpX(u, F(u), dF(u))
        
        [λ, EV, its] = normEstimator[0](X,**kwargs)
        kwargs['x'], kwargs['λ'], kwargs['tol'] = EV, λ, 1.1     

        u, t, mvstep = exprbstep(u, t, τ, X, v, s, tol, normEstimate=λ)
        mv += mvstep + its # Number of matrix-vector multiplications
    assert(abs(t - t_end) < τ)
    functionEvaluations = s
    derivativeEvaluations = s
    
    return u, (functionEvaluations, derivativeEvaluations, mv)

def exprb3(F, u, t, t_end, linearCase, s, 
           tol=None, normEstimator=None, dF=None):
    
    if linearCase:
        raise(NotImplementedError)

    functionEvaluations = 0
    derivativeEvaluations = 0
    mv = 0
    
    u = u.copy()
    τ = (t_end-t)/s
    Nx = len(u)

    v = np.vstack((np.zeros((Nx,1)),1))
    v3 = np.vstack((np.zeros((Nx+2,1)),1))
    kwargs = {**normEstimator[1]}
    for k in range(s):
        Fu, J = F(u), dF(u)
        X = LinOpX(u, Fu, J)
        
        [λ, EV, its] = normEstimator[0](X,**kwargs)
        kwargs['x'], kwargs['λ'], kwargs['tol'] = EV, λ, 1.1        

        U2, _, mv2 = exprbstep(u, t, τ, X, v, s, tol, normEstimate=λ) 
        D2 = F(U2)-Fu - J@(U2-u)
        
        k3 = 2*D2/τ**2 # 1 function evaluation and mv
        X3 = LinOpX3(u, Fu, J, k3) 
        
        u, t, mvstep = exprbstep(u, t, τ, X3, v3, s, tol, normEstimate=λ)
        mv += mv2 + mvstep + its + 1
    assert(abs(t - t_end) < τ)
    functionEvaluations = 2*s
    derivativeEvaluations = s
    
    return u, (functionEvaluations, derivativeEvaluations, mv)

def exprb4(F, u, t, t_end, linearCase, s, 
           tol=None, normEstimator=None, dF=None):
    
    if linearCase:
        raise(NotImplementedError)
    
    functionEvaluations = 0
    derivativeEvaluations = 0
    mv = 0
    
    u = u.copy()
    τ = (t_end-t)/s
    Nx = len(u)

    v1 = np.vstack((np.zeros((Nx,1)),1))
    v4 = np.vstack((np.zeros((Nx+3,1)),1))
    kwargs = {**normEstimator[1]}
    for k in range(s):
        Fu, J = F(u), dF(u)
        X = LinOpX(u, Fu, J)
        
        [λ, EV, its] = normEstimator[0](X,**kwargs)
        kwargs['x'], kwargs['λ'], kwargs['tol'] = EV, λ, 1.1   

        U2, _, mv2 = exprbstep(u, t, τ/2, X, v1, s, tol, normEstimate=λ)       
        D2 = F(U2)-Fu - J@(U2-u) # 1 function evaluation and mv
        
        U3, _, mv3 = exprbstep(u, t,   τ, X, v1, s, tol, normEstimate=λ)       
        D3 = F(U3)-Fu - J@(U3-u) # 1 function evaluation and mv
        
        k3 = ( 16*D2 -  2*D3)/τ**2 
        k4 = (-48*D2 + 12*D3)/τ**3
        X4 = LinOpX4(u, Fu, J, k3, k4) 
        
        u, t, mvstep = exprbstep(u, t, τ, X4, v4, s, tol, normEstimate=λ)
        mv += mv2 + mvstep + its + 2
    assert(abs(t - t_end) < τ)
    functionEvaluations = 3*s
    derivativeEvaluations = s
    
    return u, (functionEvaluations, derivativeEvaluations, mv)

def exprbstep(u, t, τ, X, v, s, tol, normEstimate):
    para = select_interp_para_tmp(
        τ, X, v, tol, normEstimate = normEstimate)

    
    if para[0] > 10**10:
        raise ValueError("Too many substeps, computation will fail")

    atol, rtol, vectornorm = tol[:3]
    
    # Similar to calling expleja, but we choose a different atol and rtol
    expXv, _, info, c, _, _, _ = newton_wrapper(τ, v, *para, 
                                               atol/s, rtol/s, vectornorm)

    mv = int(sum(info))
    assert(c==0)
    return u + expXv[:len(u)], t + τ, mv

def LinOpX(u, Fu, J):
    def mv(v):
        w = np.zeros(v.shape)
        if v.ndim == 2:
            w[:-1] = J@v[:-1] + Fu*v[-1]
        else:
            w[:-1] = J@v[:-1] + Fu.flatten()*v[-1]
        return w
    return LinearOperator((len(u)+1,len(u)+1), matvec = mv)

def LinOpX3(u, Fu, J, k3):
    def mv(v):
        w = np.zeros(v.shape)
        if v.ndim == 2:
            w[:-3] = J@v[:-3] + v[-1]*Fu + v[-3]*k3 
            w[-3:-1] = v[-2:]
        else:
            w[:-3] = J@v[:-3] + v[-1]*Fu.flatten() + v[-3]*k3.flatten()
            w[-3:-1] = v[-2:]
        return w
    return LinearOperator((len(u)+3,len(u)+3), matvec = mv)

def LinOpX4(u, Fu, J, k3, k4):
    def mv(v):
        w = np.zeros(v.shape)
        if v.ndim == 2:
            w[:-4] = J@v[:-4] + v[-1]*Fu + v[-3]*k3 + v[-4]*k4 
            w[-4:-1] = v[-3:]
        else:
            w[:-4] = J@v[:-4] + (v[-1]*Fu + v[-3]*k3 + v[-4]*k4).flatten()
            w[-4:-1] = v[-3:]
        return w
    return LinearOperator((len(u)+4,len(u)+4), matvec = mv)

def largestEV(A, x=None, λ = float('inf'), powerits=100, safetyfactor=1.1,
              tol=0):
    if x is None:
        x = np.random.randn(A.shape[0],1)

    for mv in range(1,powerits+1):
        λ_old, λ = λ, np.linalg.norm(x)
        y = x/λ
        x = A.dot(y)
        if abs(λ-λ_old)<tol*λ or λ==0:
            break
    return safetyfactor*λ, y, mv

def select_interp_para_for_fixed_m_and_s(
        h, A, v, tol, s=1, m=99, normEstimate = None):
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
        if normEstimate is None:
            μ, _, mv = largestEV(A, powerits=100)
        else:
            μ = normEstimate
            mv = 0

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

def select_interp_para_tmp(h, A, v, tol, normEstimate = None, m_max=99):
    '''
    The code is shortened version select_interp_para from expleja
    and forces
        a fixed interpolation degree m,
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
        if normEstimate is None:
            μ, _, mv = largestEV(A, powerits=100)
        else:
            μ = normEstimate
            mv = 0

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
        
    mm = min(m_max, len(θ))
    if γ2 == 0: #Prevents division by 0
        m = 0
    else:
        m = np.argmin(np.arange(2,mm) * np.ceil((h*γ2)/θ[2:mm]).T)
        m = m+2
    nsteps = int(np.ceil((h*γ2)/θ[m]))

    γ2, dd = θ[m], dd[:,m]
    ξ = ξ*(γ2/2)

    return nsteps, γ2, ξ.flatten(), dd, A, μ, mv, m
