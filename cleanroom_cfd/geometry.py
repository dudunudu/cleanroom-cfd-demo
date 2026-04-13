import xml.dom.minidom as minidom
import numpy as np


def load_room_from_svg(svg_file: str, res: int):
    cm_to_m = 0.01

    doc = minidom.parse(svg_file)
    svg_root = doc.getElementsByTagName("svg")[0]

    # ViewBox and physical dimensions
    viewBox = svg_root.getAttribute("viewBox").split()
    vb_x, vb_y, vb_width, vb_height = map(float, viewBox)

    svg_width_cm = float(svg_root.getAttribute("width").replace("cm", ""))
    svg_height_cm = float(svg_root.getAttribute("height").replace("cm", ""))

    svg_width_m = svg_width_cm * cm_to_m
    svg_height_m = svg_height_cm * cm_to_m

    scale_x = svg_width_m / vb_width
    scale_y = svg_height_m / vb_height

    # Grid size
    grid_w = int(svg_width_m * res)
    grid_h = int(svg_height_m * res)

    is_obstacle = np.zeros((grid_h, grid_w), dtype=bool)

    for rect in doc.getElementsByTagName("rect"):
        x = float(rect.getAttribute("x"))
        y = float(rect.getAttribute("y"))
        width = float(rect.getAttribute("width"))
        height = float(rect.getAttribute("height"))
        style = rect.getAttribute("style").lower()

        fill = "gray"
        if "fill:" in style:
            fill = style.split("fill:")[1].split(";")[0]
            if fill in ["none", "transparent"]:
                continue

        # Convert from SVG/viewBox coordinates to meters
        x_m = x * scale_x
        y_m = svg_height_m - (y + height) * scale_y
        width_m = width * scale_x
        height_m = height * scale_y

        # Convert meters to grid indices
        x_idx = int(x_m * res)
        y_idx = int(y_m * res)
        w_idx = int(width_m * res)
        h_idx = int(height_m * res)

        is_obstacle[y_idx:y_idx + h_idx, x_idx:x_idx + w_idx] = True

    doc.unlink()

    return {
        "svg_width_m": svg_width_m,
        "svg_height_m": svg_height_m,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "is_obstacle": is_obstacle,
    }