from __future__ import print_function
import numpy as np
from scipy.sparse import identity, issparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg.interface import IdentityOperator
from expleja import expleja
import scipy.io as sio
from AdvectionDiffusion1D import AdvectionDiffusion1D
from copy import deepcopy

def select_interp_parax(h, A, v, tol=[0,2**-53,float('inf'),float('inf')],
                                     m_max=99):

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

    n = v.shape[0]

    if isinstance(A,LinearOperator):
        '''
        Selects the shift μ (Half of (absolutely) largest eigenvalue)
        '''
        [μ,_,mv] = largestEV(A)
        μ = -μ/2.0
        A -= μ*IdentityOperator((n,n))
        γ2 = abs(μ)

    else:
        μ = sum(A.diagonal())/float(n)
        A -= μ*identity(n)
        [γ2,_,mv] = largestEV(A)
        γ2 = abs(γ2)
        if not issparse(A):
            A = np.asarray(A)

    mm = min(m_max, len(θ))
    if γ2 == 0: #Prevents division by 0
        m = 0
    else:
        m = np.argmin(np.arange(2,mm) * np.ceil((h*γ2)/θ[2:mm]).T)
        m = m+2
    s = int(np.ceil((h*γ2)/θ[m]))

    γ2, dd = θ[m], dd[:,m]
    ξ = ξ*(γ2/2)

    return s, γ2, ξ.flatten(), dd, A, μ, mv, m

def largestEV(A, x=None, λ = float('inf'), powerits=20, safetyfactor=1.,
              tol=1e-3):
    if x is None:
        x = np.random.randn(A.shape[0],1)

    for mv in range(1,powerits+1):
        λ_old, λ = λ, np.linalg.norm(x)
        y = x/λ
        x = A.dot(y)
        if abs(λ-λ_old)<tol*λ or λ==0:
            break
    return safetyfactor*λ, y, mv

N = 400
h = 0.1
tol = [0,2**-24,2,2]
A, u = AdvectionDiffusion1D(N, 1, 0, asLinearOp = False)
u *= 1
A = np.random.randn(N,N)
sol = expleja(h, A, u, tol=[0,2**-53,2,2])[0]
expAv1, _, info1, c1, m1, _, _ = expleja(h, A, u, tol=tol, p=0)
expAv2, _, info2, c2, m2, _, _ = expleja(h, A, u, tol=tol, interp_para = 
                 select_interp_parax(h, A, u, tol=tol))
err1 = np.linalg.norm(sol-expAv1)
err2 = np.linalg.norm(sol-expAv2)

print('      ','left = old expleja','      ','right = new expleja')
print('Errors:',err1, err2)
print('Cost:',int(sum(info1)+c1),int(sum(info2)+c2))
print('Interpolation degree:', m1, m2)
print('Substeps:', len(info1), len(info2))

