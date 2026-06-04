import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _entity_bounds(ent, res, grid_h, grid_w):
    x0 = max(0, int(ent.x * res))
    y0 = max(0, int(ent.y * res))
    x1 = min(grid_w, int((ent.x + ent.width) * res))
    y1 = min(grid_h, int((ent.y + ent.height) * res))
    return x0, x1, y0, y1


def _entity_box_mask(ent, res, grid_h, grid_w):
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    x0, x1, y0, y1 = _entity_bounds(ent, res, grid_h, grid_w)
    mask[y0:y1, x0:x1] = True
    return mask


def _entity_circle_mask(ent, res, grid_h, grid_w, radius_m):
    yy, xx = np.indices((grid_h, grid_w))
    cx = (ent.x + 0.5 * ent.width) * res
    cy = (ent.y + 0.5 * ent.height) * res
    r = radius_m * res
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _entity_surround_circle_mask(ent, res, grid_h, grid_w, inner_radius_m, outer_radius_m):
    outer = _entity_circle_mask(ent, res, grid_h, grid_w, outer_radius_m)
    inner = _entity_circle_mask(ent, res, grid_h, grid_w, inner_radius_m)
    return outer & (~inner)


def compute_metrics(
    T, u, v, is_obstacle, machine_entities, res,
    surround_pad_cells=3,
    use_circular_machine_metrics=True,
    default_machine_radius_m=0.30,
    surround_outer_radius_m=0.60
):
    air_mask = ~is_obstacle
    speed = np.sqrt(u * u + v * v)

    air_T = T[air_mask]
    air_speed = speed[air_mask]

    metrics = {
        "room_avg_temp": float(np.mean(air_T)),
        "room_max_temp": float(np.max(air_T)),
        "room_min_temp": float(np.min(air_T)),
        "room_std_temp": float(np.std(air_T)),
        "room_avg_speed": float(np.mean(air_speed)),
        "room_max_speed": float(np.max(air_speed)),
    }

    grid_h, grid_w = T.shape

    for ent in machine_entities:
        if use_circular_machine_metrics:
            inner_radius_m = ent.thermal_radius_m if ent.thermal_radius_m is not None else default_machine_radius_m
            box_mask = _entity_circle_mask(ent, res, grid_h, grid_w, inner_radius_m) & air_mask
            surround_mask = _entity_surround_circle_mask(
                ent, res, grid_h, grid_w,
                inner_radius_m=inner_radius_m,
                outer_radius_m=surround_outer_radius_m
            ) & air_mask
        else:
            box_mask = _entity_box_mask(ent, res, grid_h, grid_w) & air_mask

            x0, x1, y0, y1 = _entity_bounds(ent, res, grid_h, grid_w)
            surround_mask = np.zeros((grid_h, grid_w), dtype=bool)
            ox0 = max(0, x0 - surround_pad_cells)
            ox1 = min(grid_w, x1 + surround_pad_cells)
            oy0 = max(0, y0 - surround_pad_cells)
            oy1 = min(grid_h, y1 + surround_pad_cells)
            surround_mask[oy0:oy1, ox0:ox1] = True
            surround_mask[y0:y1, x0:x1] = False
            surround_mask &= air_mask

        if np.any(box_mask):
            metrics[f"machine_{ent.id_name}_box_temp"] = float(np.mean(T[box_mask]))
        else:
            metrics[f"machine_{ent.id_name}_box_temp"] = np.nan

        if np.any(surround_mask):
            metrics[f"machine_{ent.id_name}_surround_temp"] = float(np.mean(T[surround_mask]))
        else:
            metrics[f"machine_{ent.id_name}_surround_temp"] = np.nan

    return metrics


def run_simulation_and_collect(
    thermal_grid, sock_tracer, u_vel, v_vel, p,
    *,
    step_fn,
    frames,
    substeps_per_frame,
    dt,
    is_obstacle,
    machine_entities,
    res,
    real_start_timestamp=None,
    surround_pad_cells=3,
    use_circular_machine_metrics=True,
    default_machine_radius_m=0.30,
    surround_outer_radius_m=0.60
):
    state = {
        "T": thermal_grid.copy(),
        "tracer": sock_tracer.copy(),
        "u": u_vel.copy(),
        "v": v_vel.copy(),
        "p": p.copy(),
    }

    records = []

    for frame in range(frames):
        for _ in range(substeps_per_frame):
            state["T"], state["tracer"], state["u"], state["v"], state["p"] = step_fn(
                state["T"], state["tracer"], state["u"], state["v"], state["p"]
            )

        sim_t = (frame + 1) * substeps_per_frame * dt

        metrics = compute_metrics(
            state["T"],
            state["u"],
            state["v"],
            is_obstacle,
            machine_entities,
            res,
            surround_pad_cells=surround_pad_cells,
            use_circular_machine_metrics=use_circular_machine_metrics,
            default_machine_radius_m=default_machine_radius_m,
            surround_outer_radius_m=surround_outer_radius_m
        )

        metrics["frame"] = frame + 1
        metrics["sim_time_s"] = sim_t
        metrics["sim_time_min"] = sim_t / 60.0

        if real_start_timestamp is not None:
            metrics["real_datetime"] = (
                pd.Timestamp(real_start_timestamp) + pd.Timedelta(seconds=sim_t)
            )

        records.append(metrics)

    df_metrics = pd.DataFrame(records)
    return df_metrics, state


def build_base_ambient_mask(
    is_obstacle,
    persistent_hot_entities,
    res,
    machine_exclusion_radius_m=0.60,
    exclude_above_y_m=None,
):
    """
    Build a static ambient mask:
    - starts from all air cells
    - removes persistent hot entities (machines, screens, etc.)
    - optionally removes the room region above a chosen y-level
    """
    grid_h, grid_w = is_obstacle.shape
    ambient_mask = (~is_obstacle).copy()

    yy, xx = np.indices((grid_h, grid_w))

    if exclude_above_y_m is not None:
        y_m = yy / res
        ambient_mask &= (y_m < exclude_above_y_m)

    for ent in persistent_hot_entities:
        inner_r = ent.thermal_radius_m if ent.thermal_radius_m is not None else machine_exclusion_radius_m
        exclusion_r = max(inner_r, machine_exclusion_radius_m)

        ent_mask = _entity_circle_mask(ent, res, grid_h, grid_w, exclusion_r)
        ambient_mask &= (~ent_mask)

    return ambient_mask


def apply_dynamic_human_exclusion(
    base_ambient_mask,
    humans,
    res,
    human_exclusion_radius_m=0.30,
):
    """
    Starting from the static ambient mask, remove humans for the current frame.
    """
    ambient_mask = base_ambient_mask.copy()
    grid_h, grid_w = ambient_mask.shape

    if humans is None:
        return ambient_mask

    for ent in humans:
        human_mask = _entity_circle_mask(ent, res, grid_h, grid_w, human_exclusion_radius_m)
        ambient_mask &= (~human_mask)

    return ambient_mask


def compute_ambient_metrics(T, ambient_mask):
    """
    Compute mean / median ambient temperature over currently valid ambient cells.
    """
    vals = T[ambient_mask]

    if vals.size == 0:
        return {
            "sim_ambient_mean": np.nan,
            "sim_ambient_median": np.nan,
            "ambient_cell_count": 0,
        }

    return {
        "sim_ambient_mean": float(np.mean(vals)),
        "sim_ambient_median": float(np.median(vals)),
        "ambient_cell_count": int(vals.size),
    }


def run_simulation_and_collect_all(
    thermal_grid, sock_tracer, u_vel, v_vel, p,
    *,
    step_fn,
    frames,
    substeps_per_frame,
    dt,
    room,
    cfg,
    is_obstacle,
    persistent_hot_entities,
    persistent_machine_entities,
    res,
    dynamic_entities_state,
    real_start_timestamp=None,
    machine_exclusion_radius_m=0.60,
    human_exclusion_radius_m=0.30,
    surround_pad_cells=3,
    use_circular_machine_metrics=True,
    default_machine_radius_m=0.30,
    surround_outer_radius_m=0.60,
    exclude_region_above_sock=True,
    fixed_ambient_top_cutoff_m=None,
):
    state = {
        "T": thermal_grid.copy(),
        "tracer": sock_tracer.copy(),
        "u": u_vel.copy(),
        "v": v_vel.copy(),
        "p": p.copy(),
    }

    records = []

    exclude_above_y_m = None
    if fixed_ambient_top_cutoff_m is not None:
        exclude_above_y_m = fixed_ambient_top_cutoff_m
    elif exclude_region_above_sock:
        exclude_above_y_m = cfg.y_sock_m + cfg.sock_thickness_m

    base_ambient_mask = build_base_ambient_mask(
        is_obstacle=is_obstacle,
        persistent_hot_entities=persistent_hot_entities,
        res=res,
        machine_exclusion_radius_m=machine_exclusion_radius_m,
        exclude_above_y_m=exclude_above_y_m,
    )

    for frame in range(frames):
        for _ in range(substeps_per_frame):
            state["T"], state["tracer"], state["u"], state["v"], state["p"] = step_fn(
                state["T"], state["tracer"], state["u"], state["v"], state["p"]
            )

        sim_t = (frame + 1) * substeps_per_frame * dt
        current_humans = dynamic_entities_state.get("humans", [])

        metrics = compute_metrics(
            state["T"],
            state["u"],
            state["v"],
            is_obstacle,
            persistent_machine_entities,
            res,
            surround_pad_cells=surround_pad_cells,
            use_circular_machine_metrics=use_circular_machine_metrics,
            default_machine_radius_m=default_machine_radius_m,
            surround_outer_radius_m=surround_outer_radius_m
        )

        ambient_mask = apply_dynamic_human_exclusion(
            base_ambient_mask=base_ambient_mask,
            humans=current_humans,
            res=res,
            human_exclusion_radius_m=human_exclusion_radius_m,
        )

        ambient_metrics = compute_ambient_metrics(state["T"], ambient_mask)

        metrics["sim_ambient_mean"] = ambient_metrics["sim_ambient_mean"]
        metrics["sim_ambient_median"] = ambient_metrics["sim_ambient_median"]
        metrics["ambient_cell_count"] = ambient_metrics["ambient_cell_count"]
        metrics["human_count"] = len(current_humans)

        metrics["frame"] = frame + 1
        metrics["sim_time_s"] = sim_t
        metrics["sim_time_min"] = sim_t / 60.0

        if real_start_timestamp is not None:
            metrics["real_datetime"] = (
                pd.Timestamp(real_start_timestamp) + pd.Timedelta(seconds=sim_t)
            )

        records.append(metrics)

    df_all_metrics = pd.DataFrame(records)
    return df_all_metrics, state


def save_metric_plots(df_metrics, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "metrics.csv")
    df_metrics.to_csv(csv_path, index=False)

    use_real_time = ("real_datetime" in df_metrics.columns) and df_metrics["real_datetime"].notna().any()

    if use_real_time:
        x = pd.to_datetime(df_metrics["real_datetime"])
        x_label = "Real time"
        rotate_xticks = True
    else:
        x = df_metrics["sim_time_min"]
        x_label = "Simulation time (minutes)"
        rotate_xticks = False

    # Room temperature metrics
    plt.figure(figsize=(10, 6))
    plt.plot(x, df_metrics["room_avg_temp"], label="Room avg temp")
    plt.plot(x, df_metrics["room_max_temp"], label="Room max temp")
    plt.plot(x, df_metrics["room_min_temp"], label="Room min temp")
    plt.xlabel(x_label)
    plt.ylabel("Temperature (°C)")
    plt.title("Room temperature metrics over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if rotate_xticks:
        plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "room_temperature_metrics.png"), dpi=150)
    plt.close()

    # Room std only
    plt.figure(figsize=(10, 6))
    plt.plot(x, df_metrics["room_std_temp"], label="Room temp std")
    plt.xlabel(x_label)
    plt.ylabel("Temperature std (°C)")
    plt.title("Room temperature standard deviation over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if rotate_xticks:
        plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "room_variability_and_speed.png"), dpi=150)
    plt.close()

    # Machine surround temperatures
    surround_cols = [c for c in df_metrics.columns if c.endswith("_surround_temp")]
    if surround_cols:
        plt.figure(figsize=(12, 7))
        for col in surround_cols:
            plt.plot(x, df_metrics[col], label=col)
        plt.xlabel(x_label)
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature around each machine over time")
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        if rotate_xticks:
            plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "machine_surround_temperatures.png"), dpi=150)
        plt.close()

    # Machine hotspot temperatures
    box_cols = [c for c in df_metrics.columns if c.endswith("_box_temp")]
    if box_cols:
        plt.figure(figsize=(12, 7))
        for col in box_cols:
            plt.plot(x, df_metrics[col], label=col)
        plt.xlabel(x_label)
        plt.ylabel("Temperature (°C)")
        plt.title("Machine hotspot temperatures over time")
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        if rotate_xticks:
            plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "machine_box_temperatures.png"), dpi=150)
        plt.close()

    return csv_path