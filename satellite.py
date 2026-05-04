import numpy as np
from mechanics import step_orbit

class Satellite:
    def __init__(self, pos, vel, dry_mass=100.0, fuel_mass=50.0, max_fuel_rate=0.1, exhaust_velocity=3000.0):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.dry_mass = float(dry_mass)
        self.fuel_mass = float(fuel_mass)
        self.max_fuel_rate = float(max_fuel_rate)
        self.exhaust_velocity = float(exhaust_velocity)

    @property
    def wet_mass(self):
        return self.dry_mass + self.fuel_mass

    def apply_thrust(self, throttle: float, attitude: np.ndarray, dt: float):
        """
        Applies thrust based on an ego-centric local frame (RSW/RTN):
        attitude[0]: Transverse/In-track (Forward/Backward) -> Changes orbit size
        attitude[1]: Normal/Cross-track (Left/Right) -> Changes inclination
        attitude[2]: Radial (Up/Outward) -> Changes eccentricity
        """
        throttle = np.clip(throttle, 0.0, 1.0)
        if throttle <= 0.0 or self.fuel_mass <= 0.0:
            return

        att = np.array(attitude, dtype=np.float64)
        norm = np.linalg.norm(att)
        if norm < 1e-8:
            return
        local_dir = att / norm

        # Build ego-centric rotation matrix (Local to Global)
        r_hat = self.pos / np.linalg.norm(self.pos)
        h_vec = np.cross(self.pos, self.vel)
        h_hat = h_vec / np.linalg.norm(h_vec)
        t_hat = np.cross(h_hat, r_hat)
        
        # Matrix columns are the local basis vectors in global coordinates
        # X: Transverse (Forward), Y: Normal (Left), Z: Radial (Up)
        local_to_global = np.column_stack((t_hat, h_hat, r_hat))
        
        # Rotate the local direction vector into the global frame
        global_direction = local_to_global.dot(local_dir)

        mass_to_burn = throttle * self.max_fuel_rate * dt
        mass_to_burn = min(mass_to_burn, self.fuel_mass)

        if mass_to_burn > 0:
            m0 = self.wet_mass
            m1 = m0 - mass_to_burn
            dv = self.exhaust_velocity * np.log(m0 / m1)
            self.vel += global_direction * dv
            self.fuel_mass -= mass_to_burn

    def step(self, dt: float, mu: float = 3.986004418e14):
        self.pos, self.vel = step_orbit(self.pos, self.vel, dt, mu)
