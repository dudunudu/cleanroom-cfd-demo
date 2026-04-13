import numpy as np
from .numerics import advect_upwind, laplacian, apply_scalar_bc, apply_velocity_bc
from .pressure import project_incompressible


def compute_time_step(dx, sock_speed, alpha_heat, nu_eff):
    dt_adv = dx / max(abs(sock_speed), 1e-12)
    dt_diff_T = dx**2 / (4 * alpha_heat)
    dt_diff_u = dx**2 / (4 * nu_eff)
    return min(dt_adv, dt_diff_T, dt_diff_u) * 0.20


def make_flow_obstacle(is_obstacle, src_y0, src_y1):
    flow_obstacle = is_obstacle.copy()
    flow_obstacle[src_y0:src_y1, 1:-1] = False
    return flow_obstacle


def cfd_step(
    T, sock_tracer, u, v, p, *,
    dx, dt, is_obstacle, flow_obstacle,
    src_y0, src_y1,
    hs_x, hs_y, bubble_r,
    alpha_heat, nu_eff, rho, beta_b, g, T_ref,
    sock_speed, supply_temp, smoke_diff,
    pressure_iters, max_speed
):
    """
    CFD step

    - full-width sock
    - downward-only release
    - no upper/upward emission
    """

    # Force the sock direction to match the original notebook: downward only
    v_sock_target = -abs(sock_speed)

    # 1) Advect momentum
    u_star = advect_upwind(u, u, v, dt, dx)
    v_star = advect_upwind(v, u, v, dt, dx)

    # 2) Original-style sock forcing:
    # full-width band, downward only
    inject_strength = 0.55
    v_star[src_y0:src_y1, 1:-1] = (
        (1.0 - inject_strength) * v_star[src_y0:src_y1, 1:-1]
        + inject_strength * v_sock_target
    )
    u_star[src_y0:src_y1, 1:-1] *= (1.0 - inject_strength)

    # 3) Optional buoyancy
    if beta_b != 0.0:
        v_star[1:-1, 1:-1] += dt * (g * beta_b * (T[1:-1, 1:-1] - T_ref))

    # 4) Diffuse momentum
    u_star += dt * nu_eff * laplacian(u_star, dx)
    v_star += dt * nu_eff * laplacian(v_star, dx)

    # 5) Apply velocity BCs before pressure solve
    u_star, v_star = apply_velocity_bc(u_star, v_star, flow_obstacle)

    # Keep the sock open and active after BCs too
    v_star[src_y0:src_y1, 1:-1] = (
        (1.0 - inject_strength) * v_star[src_y0:src_y1, 1:-1]
        + inject_strength * v_sock_target
    )
    u_star[src_y0:src_y1, 1:-1] *= (1.0 - inject_strength)

    # 6) Pressure projection
    u_next, v_next, p_next = project_incompressible(
        u_star, v_star, p, dx, dt, rho, pressure_iters, flow_obstacle
    )

    # 7) Re-impose the sock once after pressure solve
    v_next[src_y0:src_y1, 1:-1] = (
        (1.0 - inject_strength) * v_next[src_y0:src_y1, 1:-1]
        + inject_strength * v_sock_target
    )
    u_next[src_y0:src_y1, 1:-1] *= (1.0 - inject_strength)

    # 8) Clamp speed for stability
    speed = np.sqrt(u_next * u_next + v_next * v_next)
    mask = speed > max_speed
    if np.any(mask):
        u_next[mask] *= max_speed / speed[mask]
        v_next[mask] *= max_speed / speed[mask]

    u_next, v_next = apply_velocity_bc(u_next, v_next, flow_obstacle)

    # Final re-impose so the sock remains continuously active
    v_next[src_y0:src_y1, 1:-1] = (
        (1.0 - inject_strength) * v_next[src_y0:src_y1, 1:-1]
        + inject_strength * v_sock_target
    )
    u_next[src_y0:src_y1, 1:-1] *= (1.0 - inject_strength)

    # 9) Temperature transport
    T_next = advect_upwind(T, u_next, v_next, dt, dx)
    T_next[1:-1, 1:-1] += dt * alpha_heat * laplacian(T_next, dx)[1:-1, 1:-1]

    # Inject cold supply air in the same full-width sock band
    temp_inject_strength = 0.35
    T_next[src_y0:src_y1, 1:-1] = (
        (1.0 - temp_inject_strength) * T_next[src_y0:src_y1, 1:-1]
        + temp_inject_strength * supply_temp
    )

    # Original hotspot / furniture behavior
    T_next[is_obstacle] = 18.0
    T_next[hs_y-bubble_r:hs_y+bubble_r, hs_x-bubble_r:hs_x+bubble_r] = 80.0
    T_next = apply_scalar_bc(T_next)

    # 10) Tracer for the sock air
    tracer_next = advect_upwind(sock_tracer, u_next, v_next, dt, dx)
    tracer_next[1:-1, 1:-1] += dt * smoke_diff * laplacian(tracer_next, dx)[1:-1, 1:-1]

    # Full-width, downward-only sock tracer source
    tracer_next[src_y0:src_y1, 1:-1] = np.maximum(
        tracer_next[src_y0:src_y1, 1:-1], 0.9
    )

    tracer_next[is_obstacle] = 0.0
    tracer_next = np.clip(tracer_next, 0.0, 1.0)
    tracer_next = apply_scalar_bc(tracer_next)

    return T_next, tracer_next, u_next, v_next, p_next