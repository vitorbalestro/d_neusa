from basis.fourier1d import FourierBasis1D
import numpy as np
import torch
from math import pi
import os


def rk4(u, dt, f):
    k1 = dt * f(u)
    k2 = dt * f(u + 0.5 * k1)
    k3 = dt * f(u + 0.5 * k2)
    k4 = dt * f(u + k3)

    u_next = u + (k1 + 2*k2 + 2*k3 + k4) / 6
    return u_next

def allen_cahn_1d_solver(n_x,n_t,domain_x,domain_t,diffusion=0.0001,u0_pert=None):
    # output tensor has shape [n_t,n_x]
    device = 'cuda:0'
    ts_factor = 100
    n_t = ts_factor * (n_t-1) + 1
    ts = torch.linspace(*domain_t,n_t,device=device)

    basis = FourierBasis1D(n_x,domain_x,device=device)
    D2 = basis.omegas.to(device)

    # initial condition
    if u0_pert == None:
        u0 = (basis.x**2) * torch.cos((torch.pi) * basis.x)
    else:
        u0 = u0_pert.to(device)
        

    def f(u):
        u_hat = basis.dbt(u)
        D2u_hat = u_hat * D2
        D2u = basis.idbt(D2u_hat)
        

        return diffusion * D2u - 5*u**3 + 5*u


    T = [ts[0]]
    U = [u0]
    u = u0
    dt = ts[1]-ts[0]

    for t in ts[1:]:
        u = rk4(u,dt,f)
        T.append(t)
        U.append(u)

    U_tensor = torch.stack(U)

    u_gt = U_tensor.detach().cpu().numpy()

    u_gt = u_gt[::ts_factor,:]

    
    return u_gt


