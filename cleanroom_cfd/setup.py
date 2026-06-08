import numpy as np
from .geometry import load_room_from_svg
from .entities import Entity


def build_room_setup(cfg, svg_path):
    geom = load_room_from_svg(svg_path, cfg.res)

    svg_width_m = geom["svg_width_m"]
    svg_height_m = geom["svg_height_m"]
    grid_w = geom["grid_w"]
    grid_h = geom["grid_h"]
    is_obstacle = geom["is_obstacle"]

    dx = 1.0 / cfg.res

    hs_x = int(cfg.hotspot_x_m * cfg.res)
    hs_y = int(cfg.hotspot_y_m * cfg.res)

    y_start = max(1, int((cfg.y_sock_m - cfg.sock_thickness_m) * cfg.res))
    y_end = min(grid_h - 2, int((cfg.y_sock_m + cfg.sock_thickness_m) * cfg.res))
    y_mid = (y_start + y_end) // 2

    src_half = cfg.source_half_thickness_cells
    src_y0 = max(1, y_mid - src_half)
    src_y1 = min(grid_h - 1, y_mid + src_half)

    return {
        "svg_width_m": svg_width_m,
        "svg_height_m": svg_height_m,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "is_obstacle": is_obstacle,
        "dx": dx,
        "hs_x": hs_x,
        "hs_y": hs_y,
        "src_y0": src_y0,
        "src_y1": src_y1,
        "entity_factory": Entity,
    }


def build_initial_fields(grid_h, grid_w, is_obstacle, T_ref, supply_temp, src_y0, src_y1):
    thermal_grid = np.full((grid_h, grid_w), T_ref)
    thermal_grid[is_obstacle] = T_ref

    thermal_grid[src_y0:src_y1, 1:-1] = (
        0.65 * thermal_grid[src_y0:src_y1, 1:-1] + 0.35 * supply_temp
    )

    sock_tracer = np.zeros((grid_h, grid_w))
    sock_tracer[src_y0:src_y1, 1:-1] = 0.9

    u_vel = np.zeros((grid_h, grid_w))
    v_vel = np.zeros((grid_h, grid_w))
    p = np.zeros((grid_h, grid_w))

    return thermal_grid, sock_tracer, u_vel, v_vel, p