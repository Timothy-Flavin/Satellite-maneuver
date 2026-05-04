import numpy as np
from mechanics import (
    initialize_coverage_arrays,
    generate_temporal_orbit_points,
    generate_orbital_band_points,
    splat_to_healpix,
    extract_agent_observation,
    visualize_coverage_scene,
    InteractiveCoverageRenderer
)
from satellite import Satellite

class OrbitalEnv:
    def __init__(self, num_satellites=2, nside=16, mu_earth=3.986004418e14):
        self.num_satellites = num_satellites
        self.nside = nside
        self.mu_earth = mu_earth
        
        self.satellites = []
        np.random.seed(42)
        for _ in range(num_satellites):
            pos_dir = np.random.randn(3)
            pos_dir /= np.linalg.norm(pos_dir)
            pos = pos_dir * (6371e3 + 600e3 + np.random.rand() * 200e3)
            
            vel_dir = np.cross(pos_dir, np.random.randn(3))
            vel_dir /= np.linalg.norm(vel_dir)
            vel_mag = np.sqrt(self.mu_earth / np.linalg.norm(pos))
            vel = vel_dir * vel_mag
            vel += np.random.randn(3) * 100.0
            
            sat = Satellite(pos, vel)
            self.satellites.append(sat)
            
        self.reset_maps()
        self.orbit_data_cache = []

    def reset_maps(self):
        self.steady_state, self.current_cov, self.priority = initialize_coverage_arrays(self.nside)

    def step(self, actions: list, dt: float):
        for i, action in enumerate(actions):
            sat = self.satellites[i]
            sat.apply_thrust(action.get('throttle', 0.0), action.get('attitude', [1, 0, 0]), dt)
            sat.step(dt, self.mu_earth)
            
        return self.observe()

    def observe(self):
        self.reset_maps()
        self.orbit_data_cache = []
        
        for sat in self.satellites:
            orbit_pts = generate_temporal_orbit_points(sat.pos, sat.vel, num_points=256, mu=self.mu_earth)
            band_pts = generate_orbital_band_points(orbit_pts, N_deg=15.0, K_deg=3.0)
            
            # Splat orbit path into Steady State Map
            coverage_weights = np.full(len(orbit_pts), 0.5, dtype=np.float32)
            splat_to_healpix(
                points=orbit_pts / np.linalg.norm(orbit_pts, axis=1, keepdims=True),
                point_weights=coverage_weights,
                target_map=self.steady_state,
                nside=self.nside
            )
            
            # Splat current position into Current Coverage Map
            current_pos_normalized = sat.pos / np.linalg.norm(sat.pos)
            splat_to_healpix(
                points=current_pos_normalized[None, :],
                point_weights=np.array([2.0], dtype=np.float32),
                target_map=self.current_cov,
                nside=self.nside
            )
            
            self.orbit_data_cache.append({
                'orbit_points': orbit_pts,
                'band_points': band_pts,
                'current_pos': np.copy(sat.pos)
            })
            
        observations = []
        for sat_data in self.orbit_data_cache:
            obs = extract_agent_observation(
                band_points=sat_data['band_points'],
                orbit_points=sat_data['orbit_points'],
                steady_state=self.steady_state,
                current_cov=self.current_cov,
                priority=self.priority,
                nside=self.nside,
                num_orbit_points=256
            )
            observations.append(obs)
            
        return observations

    def render(self):
        visualize_coverage_scene(self.nside, self.orbit_data_cache)


if __name__ == '__main__':
    import time
    env = OrbitalEnv(num_satellites=2, nside=16)
    
    # Initialize interactive renderer map
    renderer = InteractiveCoverageRenderer(nside=16)
    
    # Initialize observation mapping
    _ = env.observe()
    
    # Step through time to show dynamic orbital calculations
    dt = 60.0 * 0.5  # Propagate 1/2 minutes per step
    for step_num in range(50):
        actions = []
        
        # Satellite 0 does nothing
        actions.append({'throttle': 0.0, 'attitude': [1.0, 0.0, 0.0]})
        
        # Satellite 1 thrusts randomly
        random_attitude = np.random.randn(3)
        actions.append({'throttle': 0.5, 'attitude': [1.0, 0.0, 0.0]})
        
        obs = env.step(actions, dt)
        
        print('Step', step_num+1, '| Sat 0 Vel:', round(np.linalg.norm(env.satellites[0].vel), 2), 'm/s')
        print('Step', step_num+1, '| Sat 1 Vel:', round(np.linalg.norm(env.satellites[1].vel), 2), 'm/s', '| Fuel left:', round(env.satellites[1].fuel_mass, 2), 'kg')
        
        renderer.update(env.orbit_data_cache)
        time.sleep(0.1) # Briefly pause loop to animate cleanly
