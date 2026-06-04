import pandas as pd
from .entities import Entity


def load_timeline_window(csv_path, table_window_minutes, window_start_timestamp=None):
    df_all = pd.read_csv(csv_path)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)

    table_window_seconds = table_window_minutes * 60

    if window_start_timestamp is None:
        first_ts = df_all["timestamp"].min()
    else:
        first_ts = pd.Timestamp(window_start_timestamp)

    end_ts = first_ts + pd.Timedelta(minutes=table_window_minutes)

    df_window = df_all[
        (df_all["timestamp"] >= first_ts) &
        (df_all["timestamp"] < end_ts)
    ].copy()

    unique_times = df_window["timestamp"].drop_duplicates().sort_values().to_list()
    rows_by_time = {
        t: grp.copy()
        for t, grp in df_window.groupby("timestamp")
    }

    return {
        "df_all": df_all,
        "df_window": df_window,
        "first_ts": first_ts,
        "end_ts": end_ts,
        "unique_times": unique_times,
        "rows_by_time": rows_by_time,
        "table_window_minutes": table_window_minutes,
        "table_window_seconds": table_window_seconds,
    }


def clamp_xy(x, y, w, h, svg_width_m, svg_height_m):
    x = min(max(float(x), 0.0), svg_width_m - w)
    y = min(max(float(y), 0.0), svg_height_m - h)
    return x, y


def machine_size_from_label(label):
    label = str(label).strip().lower()
    if label == "machine":
        return 1.0, 0.6
    if label == "screen":
        return 0.8, 0.3
    if label == "cableduct":
        return 0.8, 0.3
    if label == "window":
        return 0.6, 0.4
    return 0.8, 0.5

def screen_size_from_label(label):
    return 0.35, 0.20


def build_machine_state(df_window, svg_width_m, svg_height_m, machine_radius_m=0.30):
    machine_df = df_window[
        df_window["label"].astype(str).str.strip().str.lower() == "machine"
    ].copy()

    machine_groups = machine_df.groupby("object_id")

    persistent_machine_entities = []
    machine_entity_map = {}

    for machine_id, grp in machine_groups:
        label = grp["label"].mode().iloc[0]

        x_series = grp["display_x"].where(grp["display_x"].notna(), grp["projected_x"])
        y_series = grp["display_y"].where(grp["display_y"].notna(), grp["projected_y"])

        x = float(x_series.median())
        y = float(y_series.median())

        w, h = machine_size_from_label(label)
        x, y = clamp_xy(x, y, w, h, svg_width_m, svg_height_m)

        ent = Entity(
            "heat_source_short",
            x_m=x,
            y_m=y,
            width_m=w,
            height_m=h,
            id_name=str(machine_id),
        )

        ent.thermal_shape = "circle"
        ent.thermal_radius_m = machine_radius_m

        if grp["temp_mean_c"].notna().any():
            ent.temp_target = float(grp["temp_mean_c"].median())

        persistent_machine_entities.append(ent)
        machine_entity_map[machine_id] = ent

    machine_temp_lookup = {}
    for ts, grp in machine_df.groupby("timestamp"):
        temp_map = {}
        for machine_id, subgrp in grp.groupby("object_id"):
            if subgrp["temp_mean_c"].notna().any():
                temp_map[machine_id] = float(subgrp["temp_mean_c"].mean())
        machine_temp_lookup[ts] = temp_map

    return {
        "machine_df": machine_df,
        "persistent_machine_entities": persistent_machine_entities,
        "machine_entity_map": machine_entity_map,
        "machine_temp_lookup": machine_temp_lookup,
    }

def build_screen_state(df_window, svg_width_m, svg_height_m, screen_radius_m=0.18):
    screen_df = df_window[
        df_window["label"].astype(str).str.strip().str.lower() == "screen"
    ].copy()

    screen_groups = screen_df.groupby("object_id")

    persistent_screen_entities = []
    screen_entity_map = {}

    for screen_id, grp in screen_groups:
        label = grp["label"].mode().iloc[0]

        x_series = grp["display_x"].where(grp["display_x"].notna(), grp["projected_x"])
        y_series = grp["display_y"].where(grp["display_y"].notna(), grp["projected_y"])

        x = float(x_series.median())
        y = float(y_series.median())

        w, h = screen_size_from_label(label)
        x, y = clamp_xy(x, y, w, h, svg_width_m, svg_height_m)

        ent = Entity(
            "screen_source",
            x_m=x,
            y_m=y,
            width_m=w,
            height_m=h,
            id_name=str(screen_id),
        )

        ent.thermal_shape = "circle"
        ent.thermal_radius_m = screen_radius_m

        if grp["temp_mean_c"].notna().any():
            ent.temp_target = float(grp["temp_mean_c"].median())

        persistent_screen_entities.append(ent)
        screen_entity_map[screen_id] = ent

    screen_temp_lookup = {}
    for ts, grp in screen_df.groupby("timestamp"):
        temp_map = {}
        for screen_id, subgrp in grp.groupby("object_id"):
            if subgrp["temp_mean_c"].notna().any():
                temp_map[screen_id] = float(subgrp["temp_mean_c"].mean())
        screen_temp_lookup[ts] = temp_map

    return {
        "screen_df": screen_df,
        "persistent_screen_entities": persistent_screen_entities,
        "screen_entity_map": screen_entity_map,
        "screen_temp_lookup": screen_temp_lookup,
    }


def _get_rows_for_sim_time(sim_time_s, first_ts, unique_times, rows_by_time):
    if not unique_times:
        return None

    current_ts = first_ts + pd.Timedelta(seconds=sim_time_s)
    eligible_times = [t for t in unique_times if t <= current_ts]

    if not eligible_times:
        return rows_by_time[unique_times[0]]

    return rows_by_time[eligible_times[-1]]


def update_machine_entities_for_time(
    sim_time_s,
    first_ts,
    unique_times,
    machine_entity_map,
    machine_temp_lookup,
    grace_period_minutes=3,
):
    if not unique_times:
        return

    current_ts = first_ts + pd.Timedelta(seconds=sim_time_s)
    eligible_times = [t for t in unique_times if t <= current_ts]

    if not eligible_times:
        ts_to_use = unique_times[0]
    else:
        ts_to_use = eligible_times[-1]

    temp_map = machine_temp_lookup.get(ts_to_use, {})
    grace_period = pd.Timedelta(minutes=grace_period_minutes)

    for machine_id, ent in machine_entity_map.items():
        if machine_id in temp_map:
            ent.is_active = True
            ent.temp_target = temp_map[machine_id]
            ent.last_seen_ts = ts_to_use
        else:
            if not hasattr(ent, "last_seen_ts"):
                ent.is_active = False
                continue

            if current_ts - ent.last_seen_ts <= grace_period:
                ent.is_active = True
            else:
                ent.is_active = False

def update_screen_entities_for_time(
    sim_time_s,
    first_ts,
    unique_times,
    screen_entity_map,
    screen_temp_lookup,
    grace_period_minutes=3,
):
    if not unique_times:
        return

    current_ts = first_ts + pd.Timedelta(seconds=sim_time_s)
    eligible_times = [t for t in unique_times if t <= current_ts]

    if not eligible_times:
        ts_to_use = unique_times[0]
    else:
        ts_to_use = eligible_times[-1]

    temp_map = screen_temp_lookup.get(ts_to_use, {})
    grace_period = pd.Timedelta(minutes=grace_period_minutes)

    for screen_id, ent in screen_entity_map.items():
        if screen_id in temp_map:
            ent.is_active = True
            ent.temp_target = temp_map[screen_id]
            ent.last_seen_ts = ts_to_use
        else:
            if not hasattr(ent, "last_seen_ts"):
                ent.is_active = False
                continue

            if current_ts - ent.last_seen_ts <= grace_period:
                ent.is_active = True
            else:
                ent.is_active = False

def build_human_entities_for_time(
    sim_time_s,
    first_ts,
    unique_times,
    rows_by_time,
    svg_width_m,
    svg_height_m,
    human_size_m=0.4,
    human_radius_m=0.20,
):
    rows = _get_rows_for_sim_time(sim_time_s, first_ts, unique_times, rows_by_time)
    if rows is None:
        return []

    human_rows = rows[
        rows["label"].astype(str).str.strip().str.lower() == "person"
    ].copy()

    human_entities = []
    for i, row in human_rows.iterrows():
        x = row["projected_x"]
        y = row["projected_y"]
        w, h = human_size_m, human_size_m
        x, y = clamp_xy(x, y, w, h, svg_width_m, svg_height_m)

        ent = Entity(
            "human",
            x_m=x,
            y_m=y,
            width_m=w,
            height_m=h,
            id_name=str(row["object_id"]) if pd.notna(row.get("object_id")) else f"human_{i}",
        )

        ent.thermal_shape = "circle"
        ent.thermal_radius_m = human_radius_m

        if pd.notna(row.get("temp_mean_c")):
            ent.temp_target = float(row["temp_mean_c"])

        human_entities.append(ent)

    return human_entities