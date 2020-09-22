from expleja import expleja
from AdvectionDiffusion import AdvectionDiffusion1D, AdvectionDiffusion2D
import numpy as np
import scipy as sp
import pandas as pd
from scipy import integrate
from scipy.sparse import diags
from Integrators import Integrator, rk2, rk4, cn2, exprb2, exprb3, exprb4
from time import time, sleep
from itertools import product
from multiprocessing import Process, Lock
from solve_ODE import solve_ODE
import os
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
except:
    pass


'''
Linear Case: dt u = adv dxx u + dif dx u
On the interval [0,1] with homogeneous Dirichlet boundary conditions
for t in [0,0.1]

For different fixed time stepsizes and space dimension we calculate the error
and the cost for each integration method
'''

#!ptrepack -o Experiment1.h5 Experiment11.h5

'''
Nx... discretize x-axis into Nx-1 parts
s... number of substeps used when integrating
'''

def Nonlinear_Advection_Diffusion_Equation2D(Nx, param, asLinearOp, filename, 
                                             lock):
    t = 0 # Start time
    t_end = 1e-1 # Final time


    Adv, u = AdvectionDiffusion2D(Nx, 1, 0)
    Dif, u = AdvectionDiffusion2D(Nx, 0, 1)
    
    def F_cpu(u, param=param):
        f = 0.5*Dif@(u*(u+2))
        g = 2*u*(Adv@u)
        h = u*(u-0.5)
        return param[0]*f + param[1]*g + param[2]*h
    
    if Nx >= 300:
        try:
            Adv_gpu = csr_gpu(Adv)
            Dif_gpu = csr_gpu(Dif)
        
            def F_gpu(u, param=param):
                f = 0.5*Dif_gpu.dot(u*(u+2))
                g = 2*u*(Adv_gpu.dot(u))
                h = u*(u-0.5)
                return param[0]*f + param[1]*g + param[2]*h
        
            def F(u, param=param):
                if type(u) == cp.core.core.ndarray:
                    return F_gpu(u, param)
                elif Nx >= 200:
                    return cp.asnumpy(F_gpu(cp.array(u), param))
                else:
                    return F_cpu(u,param)
            
        except:
            F = F_cpu
    else:
        F = F_cpu
            
    def dF(u, param=param):
        u = u.flatten()
        df = Dif@diags((u+1))
        dg = 2*(diags(Adv@u) + diags(u)@Adv)
        dh = diags(2*u-0.5)
        return sp.sparse.csr_matrix(param[0]*df + param[1]*dg + param[2]*dh)
    

    ''' Compute reference solution '''
    solver = sp.integrate.ode(lambda t,u: F(u))
    solver.set_integrator('vode', method='bdf', order=5, atol=1e-16,
                          rtol=1e-16, nsteps=100000, with_jacobian=False)
    solver.set_initial_value(u.copy(), 0)

    while solver.successful() and solver.t<t_end:
        solver.integrate(t_end)
    reference_solution = solver.y

    ''' Get Settings '''
    Settings, methods = SettingsNonlinear(dF)

    columns = ['substeps', 'Nx', 'α','β','γ', 'relerror',
               'Feval', 'dFeval', 'mv']

    inputs = [F, u, t, t_end, False]
    Integrators = [Integrator(
            method, inputs, reference_solution,
            columns + list(Settings[method.__name__][0].keys()))
            for method in methods]

    add_to_row = [Nx, *param]
    solve_ODE(Integrators, Settings, add_to_row, filename, lock=lock)


def Nonlinear_Advection_Diffusion_Equation(Nx, param, asLinearOp, filename, lock):
    t = 0 # Start time
    t_end = 1e-1 # Final time

    Adv, u = AdvectionDiffusion1D(Nx, 1, 0)
    Dif, u = AdvectionDiffusion1D(Nx, 0, 1)

    h = 1/(Nx-1)
    def dx(u):
        # Upwind
        dxu = -u
        dxu[:-1] += u[1:]
        return dxu/h

    def F(u, param=param):
        dxu = dx(u)
        f = dxu**2 + (u+1)*Dif.dot(u)
        g = 2*u*dxu
        h = u*(u-0.5)
        return param[0]*f + param[1]*g + param[2]*h


    def dF(u, param=param):

        #df = 2*diags(dx(u)).dot(Adv) + diags(Dif.dot(u)) + diags(u+1).dot(Dif)
        #dg = 2* ( diags(dx(u)) + diags(u).dot(Adv) )
        #dh = diags(2*u - .5)
        dxu = dx(u)
        data = np.zeros((3,Nx))
        data[0,1:] = (
                param[0]*(2*dxu[:-1]
                + (u[:-1]+1)/h)/h
                + param[1]*(2*u[:-1]/h)
                ).flatten()
        data[1] = (
                param[0]*(-2*dxu/h + Dif.dot(u) -2*(u+1)/h**2)
                + param[1]*(-2*u/h + 2*dxu)
                + param[2]*(2*u - .5)
                ).flatten()
        data[2,:-1] = (
                param[0]*(
                        (u[1:]+1)/h**2)
                ).flatten()
        
        return sp.sparse.linalg.aslinearoperator(sp.sparse.csr_matrix(
                sp.sparse.dia_matrix((data,[1,0,-1]), shape=(Nx, Nx))
                ))
        


    ''' Compute reference solution '''
    solver = sp.integrate.ode(lambda t,u: F(u))
    solver.set_integrator('vode', method='bdf', order=5, atol=1e-16,
                          rtol=1e-16, nsteps=100000, with_jacobian=False)
    solver.set_initial_value(u.copy(), 0)

    while solver.successful() and solver.t<t_end:
        solver.integrate(t_end)
    reference_solution = solver.y

    ''' Get Settings '''
    Settings, methods = SettingsNonlinear(dF)

    columns = ['substeps', 'Nx', 'α','β','γ', 'relerror',
               'Feval', 'dFeval', 'mv']

    inputs = [F, u, t, t_end, False]
    Integrators = [Integrator(
            method, inputs, reference_solution,
            columns + list(Settings[method.__name__][0].keys()))
            for method in methods]

    add_to_row = [Nx, *param]
    solve_ODE(Integrators, Settings, add_to_row, filename, lock=lock)

def SettingsNonlinear(dF):
    Settings = {}
    Settings['all'] = {'tol':[2**-24], "dF": dF} 
    Settings['rk2'] = [{}]
    Settings['rk4'] = [{}]
    Settings['cn2'] = [{'tol': te} for te in Settings['all']['tol']]
    Settings['exprb2'] = [
            {'tol':te, 'powerits': 4, 'safetyfactor': 1.1}
            for te in Settings['all']['tol']]
    Settings['exprb4'] = Settings['exprb3'] = Settings['exprb2']
    substeps = (1.10**np.array(range(1,122))).astype('int')
    Settings['all']['substeps'] = np.unique(substeps)
    Settings['all']['dftype'] = {
        'substeps':np.int32, 'Nx':np.int32, 
        'α':np.float64, 'β':np.float64, 'γ':np.float64, 
        'Feval':np.int32, 'dFeval':np.int32, 'mv':np.int32,
    }

    ''' Define Integrators '''
    methods = [exprb2]
    
    return Settings, methods

def SettingsLinear(asLinearOp):
    Settings = {}
    Settings['all'] = {'tol':[2**-10,2**-24], "dF": False}
    Settings['rk2'] = [{}]
    Settings['rk4'] = [{}]
    Settings['cn2'] = [{'tol': te} for te in Settings['all']['tol']]
    Settings['all']['dftype'] = {
            'substeps':np.int32,'Nx':np.int32, 'dif':np.float64,
            'mv':np.int32,
    }

    if asLinearOp:
        ''' Define Integrators '''
        methods = [exprb2]
        substeps = [250, 500, 750, 1000]
        normEstimatorParams = [{"powerits": it, "safetyfactor": sf}
                                for it in [2,3,4,6,8,10,25,50]
                                for sf in [0.5,0.75,0.9,1,1.1,1.5,2]]
        Settings['exprb2'] = [{'tol':te, **ne}
                        for te in Settings['all']['tol']
                        for ne in normEstimatorParams]
    else:
        ''' Define Integrators '''
        methods = [exprb2, cn2, rk2, rk4]
        substeps = (1.10**np.array(range(1,122))).astype('int')
        Settings['exprb2'] = [{'tol':te} for te in Settings['all']['tol']]

    Settings['all']['substeps'] = np.unique(substeps)

    return Settings, methods


def Linear_Advection_Diffusion_Equation(Nx, dif, asLinearOp, filename, lock):    
    Settings, methods = SettingsLinear(asLinearOp)

    t = 0 # Start time
    t_end = 1e-1 # Final time

    A, u = AdvectionDiffusion1D(Nx, 1, dif, asLinearOp = False) # Dont change
    reference_solution, tmp = expleja(t_end, A, u)[:2] # Double precision

    A, u = AdvectionDiffusion1D(Nx, 1, dif, asLinearOp = asLinearOp)
    columns = ['substeps', 'Nx', 'dif', 'relerror', 'mv']

    def F(v, returnMatrix=False):
        return A if returnMatrix else A@v

    inputs = [F, u, t, t_end, True]
    Integrators = [Integrator(
            method, inputs, reference_solution,
            columns + list(Settings[method.__name__][0].keys()))
            for method in methods]

    add_to_row = [Nx, dif]

    solve_ODE(Integrators, Settings, add_to_row, filename, lock=lock)


if __name__ == '__main__':
    multiprocessing = True # On Windows use a seperate console, not IPython
    asLinearOp = True  # Calculations use LinearOperators instead of matrices
    LinearCase = False # Switches between linear and nonlinear case
    isproblem2D = True # Switches between 1D and 2D case
   
    if isproblem2D:
        assert(not LinearCase)
        assert(asLinearOp)
        Nxlist = list(range(50,401,50))
        params = [[α, β, γ] for α in 
                  [0.1, 0.01] for β in [1, 0.1, 0.01] for γ in [1]]
        problem_to_solve = Nonlinear_Advection_Diffusion_Equation2D
        filename = 'Experiment_2D.h5'

    elif LinearCase:
        Nxlist = list(range(50,401,50))
        params = [1e0, 1e-1, 1e-2] # Diffusion coefficient. Should be <= 1
        problem_to_solve = Linear_Advection_Diffusion_Equation
        filename = 'Experiment1LinOp.h5' if asLinearOp else 'Experiment1.h5'
    
    else:
        assert(asLinearOp) # Probably unnecessary
        Nxlist = list(range(50,401,50))
        params = [[α, β, γ] for α in 
                  [0.1, 0.01] for β in [1, 0.1, 0.01] for γ in [1]]
        problem_to_solve = Nonlinear_Advection_Diffusion_Equation
        filename = 'Experiment2.h5'
    filelocation = 'HDF5-files' + os.sep + filename

    begin = time()
    if multiprocessing:
        lock = Lock()
        all_processes = [
            Process(target = problem_to_solve,
            args=(Nx, param, asLinearOp, filelocation, lock)) 
            for Nx in Nxlist for param in params
            ]

        for p in all_processes: p.start()
        for p in all_processes: p.join()

    else:
        for Nx, param in product(Nxlist, params):
            problem_to_solve(Nx, param, asLinearOp, filelocation, None)

    pd.set_option('io.hdf.default_format','table')
    with pd.HDFStore(filelocation) as hdf:
        for key in hdf.keys():
            hdf[key] = hdf[key].drop_duplicates()
            print('Key',key)
            print(hdf[key])
    print('Total time:', time()-begin)
    sleep(120)