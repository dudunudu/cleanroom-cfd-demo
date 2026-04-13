import numpy as np


def apply_scalar_bc(a):
    a[0, :] = a[1, :]
    a[-1, :] = a[-2, :]
    a[:, 0] = a[:, 1]
    a[:, -1] = a[:, -2]
    return a


def apply_velocity_bc(u, v, flow_obstacle):
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0

    # top/bottom approximately open
    u[0, :] = u[1, :]
    v[0, :] = v[1, :]
    u[-1, :] = u[-2, :]
    v[-1, :] = v[-2, :]

    u[flow_obstacle] = 0.0
    v[flow_obstacle] = 0.0
    return u, v


def laplacian(a, dx):
    out = np.zeros_like(a)
    out[1:-1, 1:-1] = (
        a[2:, 1:-1] + a[:-2, 1:-1] +
        a[1:-1, 2:] + a[1:-1, :-2] -
        4.0 * a[1:-1, 1:-1]
    ) / dx**2
    return out


def advect_upwind(phi, u, v, dt, dx):
    out = phi.copy()

    center = phi[1:-1, 1:-1]
    u_c = u[1:-1, 1:-1]
    v_c = v[1:-1, 1:-1]

    dphidx = np.where(
        u_c >= 0,
        (center - phi[1:-1, :-2]) / dx,
        (phi[1:-1, 2:] - center) / dx
    )

    dphidy = np.where(
        v_c >= 0,
        (center - phi[:-2, 1:-1]) / dx,
        (phi[2:, 1:-1] - center) / dx
    )

    out[1:-1, 1:-1] -= dt * (u_c * dphidx + v_c * dphidy)
    return out