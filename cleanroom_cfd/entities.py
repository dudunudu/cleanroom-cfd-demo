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
        "friction": 0.7,
        "darcy_resistance": 33.0,
        "is_active": True,
    },
    "air_conditioning": {
        "base_temperature": 18.0,
        "velocity": (0.0, -1.5),
        "is_active": True,
        "blocks_airflow": False,
    },
    "pressure_outlet": {
        "blocks_airflow": False,
        "is_outlet": True,
        "is_active": True,
        "drain_strength": 0.01, #amo fraction of air drained
    },
    "human": {
        "base_temperature": 37.0,
        "blocks_airflow": True,
        "heat_emission": True,
    },
    "furniture_passthrough": {
    "blocks_airflow": False,
    "friction": 0.7,        
    "base_temperature": None,
    "darcy_resistance": 20.0, # less dense than machines
    "is_active": True,
    },
}

class Entity:
    def __init__(self, type_id, x_m, y_m, width_m, height_m, id_name=None):
        config = ENTITY_CATALOGE[type_id]
        self.type = type_id
        self.id_name = id_name or f"{type_id}_{id(self)}"

        # Geometry position
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
        self.is_outlet = config.get("is_outlet", False)
        self.darcy_resistance = config.get("darcy_resistance", 0.0)
        self.drain_strength = config.get("drain_strength", 0.0)

    def get_mask(self, res, grid_h, grid_w):
        """Return the boolean mask of the entity in the grid"""

        x_idx = int(self.x * res)
        y_idx = int(self.y * res)
        w_idx = int(self.width * res)
        h_idx = int(self.height * res)

        # Ensure limits
        return slice(max(0, y_idx), min(grid_h, y_idx + h_idx)), \
                slice(max(0, x_idx), min(grid_w, x_idx + w_idx))
        
    def apply_mechanics(self, u, v, p, res, grid_h, grid_w):

        if not self.is_active:
            return
        
        s_y, s_x = self.get_mask(res, grid_h, grid_w)

        # Mechanic effect
        if self.vel_x != 0 or self.vel_y != 0:
            u[s_y, s_x] = self.vel_x
            v[s_y, s_x] = self.vel_y

        # Obstacle effect
        if self.blocks_airflow:
            u[s_y, s_x] = 0.0
            v[s_y, s_x] = 0.0

        # Outlet: gradual absorption of air (direction -y)
        if self.is_outlet and self.drain_strength > 0.0:
            # Reforce componenet v towards -y gradually
            v[s_y, s_x] = (1.0 - self.drain_strength) * v[s_y, s_x] \
                        + self.drain_strength * (-abs(v[s_y, s_x]) - 0.1)
            # pressure 0 = open outlet
            p[s_y, s_x] = 0.0

        if self.is_outlet:
            p[s_y, s_x] = 0.0

    def apply_thermal(self, T, dt, res, grid_h, grid_w, h_conv=15.0):

        if not self.is_active or self.temp_target is None:
            return
        
        s_y, s_x = self.get_mask(res, grid_h, grid_w)
        T[s_y, s_x] += dt * h_conv * (self.temp_target - T[s_y, s_x])

    def get_current_temp(self):
        # Max T and amount time 
        return self.temp_target if self.is_active else None
    