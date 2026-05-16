import numpy as np
from .numerics import advect_upwind, laplacian, apply_scalar_bc, apply_velocity_bc
from .pressure import project_incompressible


def compute_time_step(dx, sock_speed, alpha_heat, nu_eff):
    dt_adv = dx / max(abs(sock_speed), 1e-12)
    dt_diff_T = dx**2 / (4 * alpha_heat)
    dt_diff_u = dx**2 / (4 * nu_eff)
    return min(dt_adv, dt_diff_T, dt_diff_u) * 0.30


def cfd_step(
    T, sock_tracer, u, v, p, *,
    dx, dt, entities_list, res, is_obstacle,
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

    p_next = p.copy()
    v_sock_target = -abs(sock_speed)
    inject_strength = 0.55

    # 1. Dynamic flow_obstacle (SVG + Entity)
    flow_obstacle = is_obstacle.copy()
    # Open the space of the air sock (direction of make_flow_obstacle)
    flow_obstacle[src_y0:src_y1, 1:-1] = False

    for ent in entities_list:
        if ent.is_active:
            s_y, s_x = ent.get_mask(res, T.shape[0], T.shape[1])
            flow_obstacle[s_y, s_x] = ent.blocks_airflow

    # 2. Advection (velocity transport and self-advection)
    # v_forced = v + dt * g * beta_b * (T - T_ref)
    # u_forced = u.copy()

    u_star = advect_upwind(u, u, v, dt, dx)
    v_star = advect_upwind(v, u, v, dt, dx)

    # 3. Sock injection
    v_star[src_y0:src_y1, 1:-1] = (1.0 - inject_strength) * v_star[src_y0:src_y1, 1:-1] + inject_strength * v_sock_target
    u_star[src_y0:src_y1, 1:-1] *= (1.0 - inject_strength)

    # 4. Gravity and buoyancy (Boussinesq approximation)
    if beta_b != 0.0:
        v_star[1:-1, 1:-1] += dt * (g * beta_b * (T[1:-1, 1:-1] - T_ref))

    # 5. Difussion (Laplacian)
    u_next = u_star + dt * nu_eff * laplacian(u_star, dx)
    v_next = v_star + dt * nu_eff * laplacian(v_star, dx)

    # 6. Apply mechanic entities (Mechanics: air conditioning, friction)
    for ent in entities_list:
        ent.apply_mechanics(u_next, v_next, p_next, res, T.shape[0], T.shape[1])

    # 7. Apply BCs (Velocities: no-slip on walls, sock injection)
    # u_star, v_star = apply_velocity_bc(u_star, v_star, flow_obstacle)
    # v_star[src_y0:src_y1, 1:-1] = (1.0 - inject_strength) * v_star[src_y0:src_y1, 1:-1] + inject_strength * v_sock_target

    # 8. Pressure projection maintain incompressibility and apply BCs
    darcy_mask = np.zeros_like(flow_obstacle)  # bool array
    darcy_coeff_grid = np.zeros_like(p)

    for ent in entities_list:
        if ent.is_active and not ent.blocks_airflow and ent.friction > 0.0:
            s_y, s_x = ent.get_mask(res, T.shape[0], T.shape[1])
            darcy_mask[s_y, s_x] = True
            darcy_coeff_grid[s_y, s_x] = ent.friction * ent.darcy_resistance

    # 8. Pressure projection con Darcy integrado
    u_next, v_next, p_next = project_incompressible(
        u_next, v_next, p_next, dx, dt, rho, pressure_iters, flow_obstacle,
        darcy_mask=darcy_mask,
        darcy_coeff=darcy_coeff_grid
    )

    # 9. Sock, again (3rd time) after pressure projection to maintain constant flow
    v_next[src_y0:src_y1, 1:-1] = (1.0 - inject_strength) * v_next[src_y0:src_y1, 1:-1] + inject_strength * v_sock_target

    # Clamp speed
    speed = np.sqrt(u_next**2 + v_next**2)
    mask = speed > max_speed
    if np.any(mask):
        u_next[mask] *= max_speed / speed[mask]
        v_next[mask] *= max_speed / speed[mask]

    u_next, v_next = apply_velocity_bc(u_next, v_next, flow_obstacle)

    # 10. Temperature transport
    T_next = advect_upwind(T, u_next, v_next, dt, dx)
    T_next[1:-1, 1:-1] += dt * alpha_heat * laplacian(T_next, dx)[1:-1, 1:-1]

    # Obstacles have conductivity
    T_neighbors = (
    np.roll(T_next, 1, axis=0) + np.roll(T_next, -1, axis=0) +
    np.roll(T_next, 1, axis=1) + np.roll(T_next, -1, axis=1)) / 4.0

    h_wall = 15 # conductivity coeffcient
    T_next[is_obstacle] = T[is_obstacle] + dt * h_wall * (T_neighbors[is_obstacle] - T[is_obstacle])

    # T_next[is_obstacle] = T_ref # obstacles constant temperature (alternative)

    # Sock temperature 
    T_next[src_y0:src_y1, 1:-1] = (
        (1.0 - inject_strength) * T_next[src_y0:src_y1, 1:-1] 
        + inject_strength * supply_temp
    )

    # Apply entities (Thermal: heat sources)
    for ent in entities_list:
        ent.apply_thermal(T_next, dt, res, T.shape[0], T.shape[1], h_conv=15.0)

    T_next = apply_scalar_bc(T_next)

    # 11. Air trace (smoke)
    tracer_next = advect_upwind(sock_tracer, u_next, v_next, dt, dx)
    tracer_next[1:-1, 1:-1] += dt * smoke_diff * laplacian(tracer_next, dx)[1:-1, 1:-1]
    tracer_next[src_y0:src_y1, 1:-1] = np.maximum(tracer_next[src_y0:src_y1, 1:-1], 0.9)
    tracer_next[is_obstacle] = 0.0
    tracer_next = np.clip(tracer_next, 0.0, 1.0)
    tracer_next = apply_scalar_bc(tracer_next)

    return T_next, tracer_next, u_next, v_next, p_next