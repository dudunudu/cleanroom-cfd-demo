import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import HTML, display


def tracer_display(a, threshold=0.005):
    return np.ma.masked_where(a < threshold, a)


def tracer_alpha(a, threshold=0.005):
    return np.clip((a - threshold) * 1.6, 0.0, 0.90)


def animate_simulation(
    thermal_grid, sock_tracer, u_vel, v_vel, p,
    *,
    step_fn, frames, substeps_per_frame,
    svg_width_m, svg_height_m,
    is_obstacle, y_sock_m, sock_thickness_m,
    hs_x_m, hs_y_m, v_sock_target, T_supply, dt
):
    fig, ax = plt.subplots(figsize=(12, 8))

    temp_im = ax.imshow(
        thermal_grid,
        cmap='magma',
        origin='lower',
        extent=[0, svg_width_m, 0, svg_height_m],
        vmin=8,
        vmax=80
    )

    smoke_im = ax.imshow(
        tracer_display(sock_tracer),
        cmap='Blues',
        origin='lower',
        extent=[0, svg_width_m, 0, svg_height_m],
        vmin=0,
        vmax=0.15,
        alpha=tracer_alpha(sock_tracer)
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

    ax.plot(hs_x_m, hs_y_m, marker='o', markersize=5, color='white')

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

    def update(frame):
        for _ in range(substeps_per_frame):
            state["T"], state["tracer"], state["u"], state["v"], state["p"] = step_fn(
                state["T"], state["tracer"], state["u"], state["v"], state["p"]
            )

        temp_im.set_data(state["T"])
        smoke_im.set_data(tracer_display(state["tracer"]))
        smoke_im.set_alpha(tracer_alpha(state["tracer"]))

        sim_t = (frame + 1) * substeps_per_frame * dt
        max_v = np.max(np.sqrt(state["u"] * state["u"] + state["v"] * state["v"]))

        ax.set_title('Temperature + air from air sock')
        info_text.set_text(
            f'Sim time: {sim_t:.2f} s\n'
            f'Sock speed: {abs(v_sock_target):.2f} m/s\n'
            f'Supply temp: {T_supply:.1f} °C\n'
            f'Max air speed now: {max_v:.2f} m/s'
        )

        return temp_im, smoke_im, info_text

    anim = FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
    plt.close(fig)
    display(HTML(anim.to_jshtml()))
    return anim