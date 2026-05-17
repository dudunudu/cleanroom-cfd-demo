from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # Grid / geometry
    res: int = 20  # cells per meter

    # Thermal setup
    room_temp: float = 20.0
    obstacle_temp: float = 18.0
    hotspot_temp: float = 80.0
    hotspot_x_m: float = 1.5
    hotspot_y_m: float = 2.0
    bubble_r: int = 5

    # Sock setup
    y_sock_m: float = 8.0
    sock_thickness_m: float = 0.5
    sock_speed: float = 0.45
    supply_temp: float = 18.0

    # Physics
    alpha_heat: float = 1.0e-2
    nu_eff: float = 5.0e-3
    rho: float = 1.0
    beta_b: float = 0.0
    g: float = 9.81
    T_ref: float = 20.0

    # Numerics
    pressure_iters: int = 30
    max_speed: float = 1.5
    smoke_diff: float = 8.0e-4
    source_half_thickness_cells: int = 2

    # Timing
    target_time: float = 18.0
    frames: int = 50
    warmup_steps: int = 3000

    # Plotting
    temp_vmin: float = 15.0
    temp_vmax: float = 45.0