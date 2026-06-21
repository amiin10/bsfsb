"""
Problem definition and data generation for the Poisson boundary-value
problem (2D or 3D), matching:

    Delta u = f(x)   on  Omega = [-5, 5]^d        (d = 2 or 3)
    u = g(x)         on  d(Omega)                 (Dirichlet)

    2D:  d^2u/dx^2 + d^2u/dy^2 = -2 sin(x) sin(y)        u* = sin(x) sin(y)
    3D:  d^2u/dx^2 + d^2u/dy^2 + d^2u/dz^2 = -3 sin(x) sin(y) sin(z)
                                              u* = sin(x) sin(y) sin(z)
"""
import numpy as np
import torch


def analytical_solution(z, dim):
    """Exact solution u*(x) = prod_i sin(x_i)."""
    out = torch.sin(z[:, 0:1])
    for i in range(1, dim):
        out = out * torch.sin(z[:, i:i + 1])
    return out


def forcing(z, dim):
    """RHS of the Poisson PDE: Delta u* = -d * prod_i sin(x_i)."""
    return -float(dim) * analytical_solution(z, dim)


def laplacian(u, z, dim):
    """d^2u/dx_1^2 + ... + d^2u/dx_d^2 via two-pass autograd."""
    grads = torch.autograd.grad(
        u, z, grad_outputs=torch.ones_like(u),
        retain_graph=True, create_graph=True
    )[0]                                                # [N, d]
    lap = torch.zeros_like(u)                           # [N, 1]
    for i in range(dim):
        u_xi = grads[:, i:i + 1]
        u_xixi = torch.autograd.grad(
            u_xi, z, grad_outputs=torch.ones_like(u_xi),
            retain_graph=True, create_graph=True
        )[0][:, i:i + 1]
        lap = lap + u_xixi
    return lap


def sample_points(N_b, N_r, dim, lb, ub, seed=0):
    """Generate boundary points (with Dirichlet values from u*) and interior
    collocation points uniformly in [lb, ub]^d.

    For 2D the boundary is 4 edges (N_b/4 points per edge).
    For 3D the boundary is 6 faces (N_b/6 points per face).

    Returns:
        Z_b : [N_b, d]   boundary coordinates
        u_b : [N_b, 1]   boundary Dirichlet values g = u*|d(Omega)
        Z_r : [N_r, d]   interior collocation coordinates
    """
    rng = np.random.default_rng(seed)
    if dim == 2:
        n_per = N_b // 4
        edges = []
        ys = rng.uniform(lb[1], ub[1], n_per)
        edges.append(np.column_stack([np.full_like(ys, lb[0]), ys]))
        ys = rng.uniform(lb[1], ub[1], n_per)
        edges.append(np.column_stack([np.full_like(ys, ub[0]), ys]))
        xs = rng.uniform(lb[0], ub[0], n_per)
        edges.append(np.column_stack([xs, np.full_like(xs, lb[1])]))
        n_top = N_b - 3 * n_per
        xs = rng.uniform(lb[0], ub[0], n_top)
        edges.append(np.column_stack([xs, np.full_like(xs, ub[1])]))
        Z_b = np.vstack(edges).astype(np.float32)
    elif dim == 3:
        n_per = N_b // 6
        faces = []
        for axis in range(3):
            for side in (lb[axis], ub[axis]):
                this_n = N_b - 5 * n_per if (axis == 2 and side == ub[2]) else n_per
                pts = np.empty((this_n, 3), dtype=np.float32)
                for j in range(3):
                    if j == axis:
                        pts[:, j] = side
                    else:
                        pts[:, j] = rng.uniform(lb[j], ub[j], this_n)
                faces.append(pts)
        Z_b = np.vstack(faces).astype(np.float32)
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    z_b_t = torch.from_numpy(Z_b)
    u_b = analytical_solution(z_b_t, dim).numpy().astype(np.float32)

    Z_r = np.column_stack([
        rng.uniform(lb[k], ub[k], N_r) for k in range(dim)
    ]).astype(np.float32)

    return Z_b, u_b, Z_r


def make_test_grid(dim, lb, ub, n_per_axis):
    """Cartesian grid for evaluation (n_per_axis along each dim)."""
    axes = [np.linspace(lb[k], ub[k], n_per_axis) for k in range(dim)]
    mesh = np.meshgrid(*axes, indexing='ij')
    Z = np.column_stack([m.flatten() for m in mesh]).astype(np.float32)
    return Z, mesh
