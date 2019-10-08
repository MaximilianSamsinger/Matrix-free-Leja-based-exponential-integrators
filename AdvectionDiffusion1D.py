import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import LinearOperator

def AdvectionDiffusion1D(n, adv_coeff, dif_coeff, periodic = False , h = None, 
                         asLinearOp = False):
    '''
    Input:
        n... Number of rows/columns of Matrix A
        adv_coeff... Advection coefficient
        dif_coeff... Diffusion coefficient
        periodic... If True, we assume periodic boundary conditions, 
                    otherwise we assume homogeneous Dirichlet boundary cond.
        h... float, space discretization length. If None, we discretize on the
             interval of unit length (for example [0,1])
        asLinearOp... If True, return a LinearOperator instead of a Matrix
            
    Output:
        A... NxN sparse matrix or LinearOperator  
        u... Nx1 numpy array for testing
        
    
    Situation:
    d_t u = adv_coeff*d_x u + dif_coeff d_xx u
    gets discretized to 
    d_t u = Au
    
    AdvectionDiffusion returns the NxN matrix A for the 2D case and a vector u 
    for testing.
    '''
    
    if h is None:
        h = 1./(n-1) #Assume space dimension is interval of unit length
    
    if asLinearOp:
        ''' Create linear Operator '''
        if periodic:
            def mv(v):
                # Advection (Upwind)
                adv = -v
                adv[:-1] += v[1:]
                adv[-1] += v[0]
                
                # Diffusion
                dif = -2*v
                dif[:-1] += v[1:]
                dif[1:] += v[:-1]
                dif[-1] += v[0]
                dif[0] += v[-1]
                
                return adv_coeff/h * adv + dif_coeff/h**2 * dif
        else:
            def mv(v):
                # Advection (Upwind)
                adv = -v
                adv[:-1] += v[1:]
                
                # Diffusion
                dif = -2*v
                dif[:-1] += v[1:]
                dif[1:] += v[:-1]
                
                return adv_coeff/h * adv + dif_coeff/h**2 * dif
        
        A = LinearOperator((n,n), matvec = mv)
    
    else:
        ''' Create sparse Matrix '''
        if periodic:
            adv_data = np.array([[-1],[1],[1]]).repeat(n, axis=1) #Upwind
            adv_offsets = np.array([0, 1, -(n-1)])
            
            dif_data = np.array([[-2],[1],[1],[1],[1]]).repeat(n, axis=1)
            dif_offsets = np.array([0, -1, 1, -(n-1), n-1])
            
        else:
            adv_data = np.array([[-1],[1]]).repeat(n, axis=1) #Upwind
            adv_offsets = np.array([0, 1])
            
            dif_data = np.array([[-2],[1],[1]]).repeat(n, axis=1)
            dif_offsets = np.array([0, -1, 1])
        
        #Advection
        A = adv_coeff/h * dia_matrix((adv_data, adv_offsets), shape=(n, n)) 
        #Diffusion
        A += dif_coeff/h**2 * dia_matrix((dif_data, dif_offsets), shape=(n, n)) 
    
    x = np.linspace(0,1,n+2)[1:n+1]
    u = np.exp(-80*((x-0.55)**2))
    u = u.reshape((n, 1)) #appropriate initial vector 
    
    return A, u
    

    