ENTITY_CATALOGE = {
    "heat_source": {
        "base_temperature": 45.0,
        "blocks_airflow": True,
        "heat emission": True,
        "state": True,
        "color_plot": "red",
    },
    "air_coditioning": {
        "base_temperature": 18.0,
        "velocity": (0.0, -1.5),
        "color_plot": "blue",
    },
    "ventilation_walls": {
        "blocks_airflow": False,
        "air_resistence": 0.5,
    },
    "human": {
        "base_temperature": 37.0,
        "blocks_airfllow": True,
        "movil": True,
        "heat_emission": True,
        "state": True,
    },
}

class Entity:
    def __init__(self, type_id, x_m, y_m, width_m, height_m, id_name=None):
        config = ENTITY_CATALOGE[type_id]
        self.type = type_id
        self.id_name = id_name or f"{type_id}_{id(self)}"

        # Geometry positioin
        self.x = x_m
        self.y = y_m
        self.width = width_m
        self.height = height_m

        # Propities from the cataloge
        self.temp = config.get("base_temperature", None)
        self.blocks_airflow = config.get("blocks_airflow", False)
        self.velocity = config.get("velocity", (0.0, 0.0))
        self.state = True

    def get_mask(self, res, grid_h, grid_w):
        """Return the boolean mask of the entity in the grid"""

        x_idx = int(self.x * res)
        y_idx = int(self.y * res)
        w_idx = int(self.width * res)
        h_idx = int(self.height * res)

        # Ensure limits
        return slice(max(0, y_idx), min(grid_h, y_idx + h_idx)), \
                slice(max(0, x_idx), min(grid_w, x_idx + w_idx))
