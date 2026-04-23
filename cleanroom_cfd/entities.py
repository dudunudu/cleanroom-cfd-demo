ENTITY_CATALOGE = {
    "heat_source_full": {
        "base_temperature": 45.0,
        "blocks_airflow": True,
        "friction": 1.0,
        "is_active": True,
    },
    "heat_source_short": {
        "base_temperature": 45.0,
        "blocks_airflow": False,
        "friction": 0.5,
        "is_active": True,
    },
    "air_coditioning": {
        "base_temperature": 18.0,
        "velocity": (0.0, -1.5),
        "is_active": True,
        "blocks_airflow": False,
    },
    "pressure_outlet": {
        "blocks_airflow": False,
        "is_outlet": True,
        "is_active": True,
    },
    "human": {
        "base_temperature": 37.0,
        "blocks_airfllow": True,
        "heat_emission": True,
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

        # State attributes
        self.is_active = config.get("is_active", True)
        self.temp_target = config.get("base_temperature", None)

        # Physics attributes
        self.blocks_airflow = config.get("blocks_airflow", False)
        self.friction = config.get("friction", 0.0)
        self.vel_x, self.vel_y = config.get("velocity", (0.0, 0.0))

        def get_mask(self, res, grid_h, grid_w):
            """Return the boolean mask of the entity in the grid"""

            x_idx = int(self.x * res)
            y_idx = int(self.y * res)
            w_idx = int(self.width * res)
            h_idx = int(self.height * res)

            # Ensure limits
            return slice(max(0, y_idx), min(grid_h, y_idx + h_idx)), \
                    slice(max(0, x_idx), min(grid_w, x_idx + w_idx))
        


    def apply_to_grid(self, T, u, v, res):

        if not self.is_active:
            return
        
        s_y, s_x = self.get_mask(res, T.shape[0], T.shape[1])

        # Termic effect
        if self.temp_target is not None:
            T[s_y, s_x] = self.temp_target

        # Mechanic effect
        if self.vel_x != 0 or self.vel_y != 0:
            u[s_y, s_x] = self.vel_x
            v[s_y, s_x] = self.vel_y

        # Obstacle effect
        if self.blocks_airflow:
            u[s_y, s_x] = 0.0
            v[s_y, s_x] = 0.0
        elif self.friction > 0.0:
            u[s_y, s_x] *= (1.0 - self.friction)
            v[s_y, s_x] *= (1.0 - self.friction)





