import numpy as np
import scipy as sp
from AdvectionDiffusion1D import AdvectionDiffusion1D
from expleja import expleja

N = 200
adv_coeff = 1
dif_coeff = 1
periodic = False
h = 1
asLinearOp = True

A, u = AdvectionDiffusion1D(N, adv_coeff, dif_coeff, periodic, h, asLinearOp)


'''
Test if AdvectionDiffusion Matrix delivers plausible results
'''
#print(A.toarray())

'''
Test if the periodic and non-periodic case have the same result, except for 
the border
'''
A1, u = AdvectionDiffusion1D(N, adv_coeff, dif_coeff, True, h)
A2, u = AdvectionDiffusion1D(N, adv_coeff, dif_coeff, False, h)
Truth_vector = np.isclose(A1 @ u,A2 @ u)[1:-1]
assert(np.all(Truth_vector))

'''
expleja: Is the result plausible?
'''

tau = 1.

A_sparse, u = AdvectionDiffusion1D(N, adv_coeff, dif_coeff, periodic, h, False)
expAv1 = sp.sparse.linalg.expm_multiply(tau*A_sparse, u)
expAv2 = expleja(tau, A, u)[0]
Truth_vector = np.isclose(expAv1, expAv2)
assert(np.all(Truth_vector))

'''
expleja: Test with zero vector
'''

A1 = AdvectionDiffusion1D(N, adv_coeff, dif_coeff, periodic, h, False)[0] # Sparse Matrix
A2 = A1.toarray() # 2D Numpy Array
A3 = AdvectionDiffusion1D(N, adv_coeff, dif_coeff, periodic, h, True)[0] # Linear Operator 

tau = 1.

def test_with_0_vector(tau, matrix):
    v = np.zeros((N,1))
    assert(expleja(tau, matrix, v)[0] is v)
    v = v.flatten()
    assert(expleja(tau, matrix, v)[0] is v)

test_with_0_vector(tau, A1)
test_with_0_vector(tau, A2)
test_with_0_vector(tau, A3)

print('Everything is ok')

'''
%timeit A@u
%timeit A_sparse@u
%memit A@u
%memit
'''
