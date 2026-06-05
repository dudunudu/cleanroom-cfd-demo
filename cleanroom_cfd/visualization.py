import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, RegularPolygon
from IPython.display import HTML, display
import pandas as pd


def animate_simulation(
    thermal_grid, sock_tracer, u_vel, v_vel, p,
    *,
    step_fn, frames, substeps_per_frame,
    svg_width_m, svg_height_m,
    is_obstacle, y_sock_m, sock_thickness_m,
    hs_x_m, hs_y_m, v_sock_target, T_supply, dt,
    entities_list=None,
    real_start_timestamp=None,
    dynamic_entities_state=None,
    save_gif_path=None,
    gif_fps=8,
    show_inline=False
):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Only show actual temperature
    temp_im = ax.imshow(
        thermal_grid,
        cmap='magma',
        origin='lower',
        extent=[0, svg_width_m, 0, svg_height_m],
        vmin=18,
        vmax=30
    )

    ax.contour(
        is_obstacle.astype(float),
        levels=[0.5],
        colors='white',
        alpha=0.55,
        extent=[0, svg_width_m, 0, svg_height_m]
    )

    sock_patch = Rectangle(
        (0, y_sock_m - sock_thickness_m),
        svg_width_m,
        2 * sock_thickness_m,
        facecolor='none',
        edgecolor='cyan',
        linewidth=2
    )
    ax.add_patch(sock_patch)

    # Draw persistent entities
    if entities_list is not None:
        for ent in entities_list:
            # Outlets as circles
            if ent.type == "pressure_outlet":
                cx = ent.x + ent.width / 2
                cy = ent.y + ent.height / 2
                circle = plt.Circle(
                    (cx, cy),
                    radius=0.18,
                    facecolor='none',
                    edgecolor='lime',
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=5
                )
                ax.add_patch(circle)

    # ax.plot(hs_x_m, hs_y_m, marker='o', markersize=5, color='white')

    plt.colorbar(temp_im, label='Temperature (°C)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')

    info_text = ax.text(
        0.02, 0.98, '',
        transform=ax.transAxes,
        ha='left', va='top',
        color='white',
        bbox=dict(facecolor='black', alpha=0.45, edgecolor='none')
    )

    state = {
        "T": thermal_grid,
        "tracer": sock_tracer,
        "u": u_vel,
        "v": v_vel,
        "p": p,
    }

    human_patches = []

    def draw_humans():
        nonlocal human_patches

        for patch in human_patches:
            patch.remove()
        human_patches = []

        if dynamic_entities_state is None:
            return

        current_humans = dynamic_entities_state.get("humans", [])

        for ent in current_humans:
            cx = ent.x + ent.width / 2
            cy = ent.y + ent.height / 2

            tri = RegularPolygon(
                (cx, cy),
                numVertices=3,
                radius=0.22,
                orientation=np.pi / 2,
                facecolor='cyan',
                edgecolor='black',
                linewidth=1.5,
                alpha=0.95,
                zorder=9
            )
            ax.add_patch(tri)
            human_patches.append(tri)

    def update(frame):
        for _ in range(substeps_per_frame):
            state["T"], state["tracer"], state["u"], state["v"], state["p"] = step_fn(
                state["T"], state["tracer"], state["u"], state["v"], state["p"]
            )

        temp_im.set_data(state["T"])
        draw_humans()

        sim_t = (frame + 1) * substeps_per_frame * dt
        lines = [f"Sim time: {sim_t:.2f} s"]

        if real_start_timestamp is not None:
            current_real_time = pd.Timestamp(real_start_timestamp) + pd.Timedelta(seconds=sim_t)
            lines.append(f"Real date/time: {current_real_time.strftime('%Y-%m-%d %H:%M:%S')}")

        lines.extend([
            f"Sock speed: {abs(v_sock_target):.2f} m/s",
            f"Supply temp: {T_supply:.1f} °C",
        ])

        ax.set_title('Temperature field')
        info_text.set_text("\n".join(lines))

        return [temp_im, info_text] + human_patches

    anim = FuncAnimation(fig, update, frames=frames, interval=120, blit=False)

    if save_gif_path is not None:
        save_dir = os.path.dirname(save_gif_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        anim.save(save_gif_path, writer="pillow", fps=gif_fps)

    if show_inline:
        display(HTML(anim.to_jshtml()))

    plt.close(fig)
    return anim