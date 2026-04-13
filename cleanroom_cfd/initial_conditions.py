import numpy as np
from .numerics import apply_scalar_bc


def warmup_bubble(grid_h, grid_w, is_obstacle, hs_x, hs_y, alpha, dx, warmup_steps, bubble_r):
    warm_dt = (dx**2) / (4 * alpha) * 0.5
    T = np.full((grid_h, grid_w), 20.0)

    for _ in range(warmup_steps):
        Tn = T.copy()
        lap = (
            T[2:, 1:-1] + T[:-2, 1:-1] +
            T[1:-1, 2:] + T[1:-1, :-2] -
            4.0 * T[1:-1, 1:-1]
        )
        Tn[1:-1, 1:-1] += (alpha * warm_dt / dx**2) * lap
        Tn[is_obstacle] = 18.0
        Tn[hs_y-bubble_r:hs_y+bubble_r, hs_x-bubble_r:hs_x+bubble_r] = 80.0
        Tn = apply_scalar_bc(Tn)
        T = Tn

    return T


def initialize_state(grid_h, grid_w):
    thermal_grid = np.full((grid_h, grid_w), 20.0)
    sock_tracer = np.zeros((grid_h, grid_w))
    u_vel = np.zeros((grid_h, grid_w))
    v_vel = np.zeros((grid_h, grid_w))
    p = np.zeros((grid_h, grid_w))
    return thermal_grid, sock_tracer, u_vel, v_vel, p