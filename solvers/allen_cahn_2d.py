import torch

def rk4(u, dt, f):
    k1 = dt * f(u)
    k2 = dt * f(u + 0.5 * k1)
    k3 = dt * f(u + 0.5 * k2)
    k4 = dt * f(u + k3)

    u_next = u + (k1 + 2*k2 + 2*k3 + k4) / 6
    return u_next

def allen_cahn_2d_solver(basis,n_t,domain_t,u0,diffusion=1e-4,beta=10,device='cuda:0'):
    
    ts = torch.linspace(*domain_t,n_t,device=device)

    basis = basis.to(device)
    D2 = basis.D2
    
    D2 = D2.to(device)

    u0 = u0.to(device)

    def f(u):
        u_hat = basis.dbt(u)
        D2u_hat = u_hat * D2
        D2u = basis.idbt(D2u_hat)

        return diffusion * D2u - beta*u**3 + beta*u
    
    T = [ts[0]]
    U = [u0.cpu()]
    u = u0
    dt = (ts[1]-ts[0]).to(device)

    with torch.no_grad():
        for t in ts[1:]:
            u = rk4(u, dt, f)
            U.append(u.cpu())
        
    U_tensor = torch.stack(U)
    u_ref = U_tensor.numpy()

    

    return u_ref
