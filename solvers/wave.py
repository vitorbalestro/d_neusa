"""This code implements a pseudo-spectral method which gives a numerical solution
to a wave equation having the form

u_tt = c^2 * D2u,

where D2 denotes the spatial Laplacian, and c is the velocity field.

The parameters are the velocity field c, the initial condition u0, the basis of
the spectral decomposition and the time domain discretization ts.

The solution is a tensor where the first coordinate is the time variable, and
the other ones are the spatial derivative. For example, in the 2d case, the solution
is a tensor [t,x,y].

We assume that the first order initial condition is

u_t(0,x) = 0,

and the boundary conditions are hard constraints given by the choice of the basis. The
default first order initial condition is

v0 = torch.zeros_like(u0),

but a different v0 may be passed as an argument"""

import torch

def wave_solver(c,u0,basis,ts,device,v0=None):
    
    c = c.to(device)
    u0 = u0.to(device)
    basis = basis.to(device)
    ts = ts.to(device)

    if v0 == None:
        v0 = torch.zeros_like(u0)
    else:
        v0 = v0.to(device)
   
    D2 = basis.D2

    def rk4(u, v, dt, f, g):
        k1 = dt * f(u, v)
        l1 = dt * g(u, v)

        k2 = dt * f(u + 0.5 * k1, v + 0.5 * l1)
        l2 = dt * g(u + 0.5 * k1, v + 0.5 * l1)

        k3 = dt * f(u + 0.5 * k2, v + 0.5 * l2)
        l3 = dt * g(u + 0.5 * k2, v + 0.5 * l2)

        k4 = dt * f(u + k3, v + l3)
        l4 = dt * g(u + k3, v + l3)

        u_next = u + (k1 + 2*k2 + 2*k3 + k4) / 6
        v_next = v + (l1 + 2*l2 + 2*l3 + l4) / 6

        return u_next, v_next


    def fu(u, v):

        return v

    def fv(u, v):
        u_hat = basis.dbt(u)

        D2u_hat = D2 * u_hat
        D2u = basis.idbt(D2u_hat)
       
        return c**2 * D2u


    T = [ts[0]]
    U = [u0]
    V = [v0]
    u = u0
    v = v0
    dt = ts[1] - ts[0]
    
    for t in ts[1:]:
        
        u,v = rk4(u,v,dt,fu,fv)

        T.append(t)
        U.append(u)
        V.append(v)

    U_tensor = torch.stack(U)
    u_ref = U_tensor.detach().cpu().numpy()

    return u_ref
    