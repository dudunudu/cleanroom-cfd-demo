import numpy as np
from .numerics import apply_velocity_bc


def project_incompressible(u, v, p, dx, dt, rho, pressure_iters, flow_obstacle):
    div = np.zeros_like(p)
    div[1:-1, 1:-1] = (
        (u[1:-1, 2:] - u[1:-1, :-2]) +
        (v[2:, 1:-1] - v[:-2, 1:-1])
    ) / (2.0 * dx)
    div[flow_obstacle] = 0.0

    rhs = (rho / dt) * div
    p_old = p.copy()
    p_new = p.copy()

    for _ in range(pressure_iters):
        p_new[1:-1, 1:-1] = 0.25 * (
            p_old[1:-1, 2:] + p_old[1:-1, :-2] +
            p_old[2:, 1:-1] + p_old[:-2, 1:-1] -
            dx**2 * rhs[1:-1, 1:-1]
        )

        p_new[:, 0] = p_new[:, 1]
        p_new[:, -1] = p_new[:, -2]
        p_new[0, :] = p_new[1, :]
        p_new[-1, :] = 0.0
        p_new[flow_obstacle] = 0.0

        p_old, p_new = p_new, p_old

    p = p_old

    u[1:-1, 1:-1] -= (dt / (2.0 * rho * dx)) * (p[1:-1, 2:] - p[1:-1, :-2])
    v[1:-1, 1:-1] -= (dt / (2.0 * rho * dx)) * (p[2:, 1:-1] - p[:-2, 1:-1])

    u, v = apply_velocity_bc(u, v, flow_obstacle)
    return u, v, p