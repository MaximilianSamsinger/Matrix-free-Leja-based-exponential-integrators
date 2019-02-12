import numpy as np
from scipy.sparse import dia_matrix

def AdvectionDiffusion1D(N, adv_coeff, dif_coeff, periodic = True , h = None):
    '''
    Input:
        N... Number of rows and columns of Matrix A
        adv_coeff... Advection coefficient
        dif_coeff... Diffusion coefficient
        periodic... If True, we assume periodic boundary conditions, 
                    otherwise we assume homogeneous Dirichlet boundary cond.
        h... float, space discretization length. If None, we discretize on the
             interval of unit length (for example [0,1])
            
    Output:
        A... NxN sparse Advection-Diffusion matrix
        u... Nx1 numpy array for testing
        
    
    Situation:
    d_t u = adv_coeff*d_x u + dif_coeff d_xx u
    gets discretized to 
    d_t u = Au
    
    AdvectionDiffusion returns the NxN matrix A for the 2D case and a vector u 
    for testing.
    '''
    
    if h is None:
        h = 1./(N-1) #Assume space dimension is interval of unit length
    
    
    if periodic:
        adv_data = np.array([[-1],[1],[1]]).repeat(N, axis=1)/h
        adv_offsets = np.array([0, 1, -(N-1)])
        
        dif_data = np.array([[-2],[1],[1],[1],[1]]).repeat(N, axis=1)/h**2
        dif_offsets = np.array([0, -1, 1, -(N-1), N-1])
        
    else:
        adv_data = np.array([[-1],[1]]).repeat(N, axis=1)/h #Upwind
        adv_offsets = np.array([0, 1])
        
        dif_data = np.array([[-2],[1],[1]]).repeat(N, axis=1)/h**2
        dif_offsets = np.array([0, -1, 1])
        
        
    #Advection
    A = adv_coeff * dia_matrix((adv_data, adv_offsets), shape=(N, N)) 
    #Diffusion
    A += dif_coeff * dia_matrix((dif_data, dif_offsets), shape=(N, N)) 
    
    x = np.linspace(0,1,N+2)[1:N+1]
    u = np.exp(-80*((x-0.55)**2))
    u = u.reshape((N, 1)) #appropriate initial vector 
    
    return A, u