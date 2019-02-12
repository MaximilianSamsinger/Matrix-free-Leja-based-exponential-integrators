import numpy as np
import scipy as sp
from AdvectionDiffusion1D import AdvectionDiffusion1D
from expleja import expleja

N = 100
adv_coeff = 2.
dif_coeff = 0.99
periodic = True
h = None

'''
Test if AdvectionDiffusion Matrix delivers plausible results
'''
A, u = AdvectionDiffusion1D(N, adv_coeff, dif_coeff, periodic, h)
print(A.toarray())

'''
Test if the periodic and non-periodic case have the same result, except for 
the border
'''
A1, u = AdvectionDiffusion1D(N, adv_coeff, dif_coeff, True, h)
A2, u = AdvectionDiffusion1D(N, adv_coeff, dif_coeff, False, h)
Truth_vector = np.isclose(A1 @ u,A2 @ u)[1:-1]
assert(np.all(Truth_vector))

'''
Expleja: Is the result plausible
'''

tau = 1.

expAv1 = sp.sparse.linalg.expm_multiply(tau*A, u)
expAv2 = expleja(tau, A, u)[0]
Truth_vector = np.isclose(expAv1, expAv2)
assert(np.all(Truth_vector))

print('Everything is ok')