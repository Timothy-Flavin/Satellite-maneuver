import numpy as np
import pyvista as pv
import healpy as hp

def generate_healpix_centers(nside: int) -> np.ndarray:
    """
    Generates the 3D unit vectors for the centers of all HEALPix pixels.
    
    Args:
        nside: The resolution parameter. Must be a power of 2. 
               Total number of pixels = 12 * nside**2.
               
    Returns:
        np.ndarray of shape (N_pixels, 3) containing XYZ coordinates.
    """
    npix = hp.nside2npix(nside)
    # nest=True aligns with CNN hierarchical pooling structures
    x, y, z = hp.pix2vec(nside, np.arange(npix), nest=True)
    return np.column_stack((x, y, z))

def map_xyz_to_healpix(points: np.ndarray, nside: int) -> np.ndarray:
    """
    Maps an array of 3D Cartesian points to their containing HEALPix pixel indices.
    
    Args:
        points: np.ndarray of shape (N, 3) representing [x, y, z] locations.
        nside: The HEALPix resolution parameter.
        
    Returns:
        np.ndarray of shape (N,) containing the integer pixel indices.
    """
    # hp.vec2pix automatically handles normalization of the input vectors
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return hp.vec2pix(nside, x, y, z, nest=True)

def generate_temporal_orbit_points(
    pos: np.ndarray, 
    vel: np.ndarray, 
    num_points: int = 256, 
    mu: float = 3.986004418e14
) -> np.ndarray:
    """
    Generates temporally equally spaced 3D points along a Keplerian orbit.
    
    Args:
        pos: (3,) array of current position in meters [X, Y, Z].
        vel: (3,) array of current velocity in m/s [Vx, Vy, Vz].
        num_points: Number of temporal points to sample.
        mu: Standard gravitational parameter (default is Earth).
        
    Returns:
        np.ndarray of shape (num_points, 3) representing the orbit path.
    """
    r0 = np.asarray(pos, dtype=np.float64)
    v0 = np.asarray(vel, dtype=np.float64)
    
    r_norm = np.linalg.norm(r0)
    v_norm = np.linalg.norm(v0)
    
    # Specific angular momentum vector
    h = np.cross(r0, v0)
    h_norm = np.linalg.norm(h)
    
    # Eccentricity vector
    e_vec = np.cross(v0, h) / mu - (r0 / r_norm)
    e = np.linalg.norm(e_vec)
    
    if e >= 1.0:
        raise ValueError("Orbit is not closed (e >= 1). Cannot generate periodic points.")
        
    # Specific orbital energy and semi-major axis
    epsilon = (v_norm**2) / 2.0 - (mu / r_norm)
    a = -mu / (2.0 * epsilon)
    
    # Perifocal coordinate frame basis vectors
    if e > 1e-8:
        p_hat = e_vec / e
        w_hat = h / h_norm
        q_hat = np.cross(w_hat, p_hat)
    else:
        # Fallback for perfectly circular orbits
        w_hat = h / h_norm
        p_hat = r0 / r_norm
        q_hat = np.cross(w_hat, p_hat)
        
    # Temporally spaced points mean equally spaced Mean Anomaly (M)
    M = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Vectorized Newton-Raphson to solve Kepler's Equation: M = E - e*sin(E)
    E = M.copy()
    for _ in range(10):
        # 10 iterations securely converges for e < 1.0 at float64 precision
        f = E - e * np.sin(E) - M
        f_prime = 1.0 - e * np.cos(E)
        E = E - f / f_prime
        
    # Perifocal plane coordinates (x_omega, y_omega)
    p_coord = a * (np.cos(E) - e)
    q_coord = a * np.sqrt(1.0 - e**2) * np.sin(E)
    
    # Transform from 2D perifocal plane to 3D Cartesian space
    orbit_points = np.outer(p_coord, p_hat) + np.outer(q_coord, q_hat)
    
    return orbit_points

def generate_orbital_band_points(
    orbit_points: np.ndarray, 
    N_deg: float, 
    K_deg: float
) -> np.ndarray:
    """
    Projects 3D orbit points onto a unit sphere and generates a cross-track 
    band of points representing +/- N degrees of inclination changes.
    
    Args:
        orbit_points: (256, 3) array of Cartesian orbital points.
        N_deg: Maximum lateral angle in degrees (e.g., 30 for +/- 30 deg).
        K_deg: Step size in degrees between lateral points.
        
    Returns:
        np.ndarray of shape (num_band_points, 3) containing the red band points.
    """
    # 1. Project the orbit onto the unit sphere
    norms = np.linalg.norm(orbit_points, axis=1, keepdims=True)
    u_points = orbit_points / norms
    
    # 2. Calculate the orbital normal vector (h_hat)
    # Using the first two points to define the orbital plane
    h_vec = np.cross(orbit_points[0], orbit_points[1])
    h_hat = h_vec / np.linalg.norm(h_vec)
    
    # 3. Define the lateral angles in radians
    angles_deg = np.arange(-N_deg, N_deg + K_deg, K_deg)
    # Remove 0.0 to avoid overlapping the central blue orbit points
    angles_deg = angles_deg[np.abs(angles_deg) > 1e-6] 
    angles_rad = np.radians(angles_deg)
    
    # 4. Generate the band using geodesic interpolation
    band_points = []
    for phi in angles_rad:
        # Vectorized generation of the entire ring at angle phi
        row_points = u_points * np.cos(phi) + h_hat * np.sin(phi)
        band_points.append(row_points)
        
    return np.vstack(band_points)

def step_orbit(pos: np.ndarray, vel: np.ndarray, dt: float, mu: float = 3.986004418e14):
    """
    Steps the satellite along its Keplerian orbit by dt seconds.
    Returns the new (position, velocity).
    Uses Universal Variables for robust propagation.
    """
    r0 = np.asarray(pos, dtype=np.float64)
    v0 = np.asarray(vel, dtype=np.float64)
    r_norm = np.linalg.norm(r0)
    v_norm = np.linalg.norm(v0)
    
    alpha = 2.0 / r_norm - (v_norm**2) / mu
    
    chi = np.sqrt(mu) * np.abs(alpha) * dt if alpha > 0 else np.sign(dt) * np.sqrt(-mu * alpha)
    
    for _ in range(20):
        chi2 = chi**2
        if alpha > 1e-6:
            z = alpha * chi2
            s = (np.sqrt(z) - np.sin(np.sqrt(z))) / (z**(1.5))
            c = (1.0 - np.cos(np.sqrt(z))) / z
        elif alpha < -1e-6:
            z = -alpha * chi2
            s = (np.sinh(np.sqrt(z)) - np.sqrt(z)) / (z**(1.5))
            c = (np.cosh(np.sqrt(z)) - 1.0) / z
        else:
            z = 0.0
            s = 1.0/6.0
            c = 1.0/2.0
            
        #r_mag = chi**2 * c + r0.dot(v0) / np.sqrt(mu) * chi * (1 - z*s) + r_norm * (1 - z*c)
        f_chi = v0.dot(r0) / np.sqrt(mu) * chi**2 * c + (1 - alpha * r_norm) * chi**3 * s + r_norm * chi - np.sqrt(mu) * dt
        f_prime = chi**2 * c + r0.dot(v0) / np.sqrt(mu) * chi * (1 - z*s) + r_norm * (1 - z*c)
        
        ratio = f_chi / f_prime
        chi = chi - ratio
        if np.abs(ratio) < 1e-6:
            break
            
    f = 1.0 - chi**2 / r_norm * c
    g = dt - chi**3 / np.sqrt(mu) * s
    r_new = f * r0 + g * v0
    r_new_norm = np.linalg.norm(r_new)
    
    f_dot = np.sqrt(mu) / (r_norm * r_new_norm) * chi * (z * s - 1)
    g_dot = 1.0 - chi**2 / r_new_norm * c
    v_new = f_dot * r0 + g_dot * v0
    
    return r_new, v_new

def visualize_coverage_scene(
    nside: int, 
    satellites_data: list
):
    """
    Uses PyVista to plot the HEALPix sphere, projected orbits, coverage bands,
    and current satellite positions.
    """
    # 1. Generate the white sphere points (HEALPix centers)
    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix), nest=True)
    sphere_points = np.column_stack((x, y, z))
    
    # 2. Initialize PyVista Plotter
    plotter = pv.Plotter()

    # Add HEALPix sphere points (White)
    sphere_cloud = pv.PolyData(sphere_points)
    plotter.add_mesh(
        sphere_cloud, 
        color="white", 
        point_size=4, 
        render_points_as_spheres=True,
        opacity=0.3
    )

    for sat in satellites_data:
        orbit_points = sat['orbit_points']
        band_points = sat['band_points']
        current_pos = sat['current_pos']
        
        # Scale to Earth radii for visualization (unit sphere = Earth's surface)
        r_earth = 6371e3
        scaled_orbit = orbit_points / r_earth
        
        # Close the loop by appending the first point
        closed_orbit = np.vstack([scaled_orbit, scaled_orbit[0]])
        
        # Create polyline for the orbit
        orbit_poly = pv.PolyData(closed_orbit)
        orbit_poly.lines = np.hstack([[len(closed_orbit)], np.arange(len(closed_orbit))])

        plotter.add_mesh(
            orbit_poly, 
            color="blue", 
            line_width=3
        )

        plotter.add_mesh(
            pv.PolyData(band_points), 
            color="red", 
            point_size=5, 
            render_points_as_spheres=True
        )
        
        # Plot current pos at true height
        current_scaled = current_pos / r_earth
        plotter.add_mesh(
            pv.PolyData(current_scaled[None, :]),
            color="green",
            point_size=15,
            render_points_as_spheres=True
        )

    # Camera and rendering setup
    plotter.set_background("black")
    plotter.add_axes()
    plotter.show()

class InteractiveCoverageRenderer:
    """
    A PyVista-based interactive renderer for the orbital environment.
    Retains the window state and efficiently updates only the changing actors each step.
    """
    def __init__(self, nside: int):
        self.nside = nside
        self.plotter = pv.Plotter()
        self.dynamic_actors = []
        
        # Static HEALPix sphere
        npix = hp.nside2npix(nside)
        x, y, z = hp.pix2vec(nside, np.arange(npix), nest=True)
        sphere_points = np.column_stack((x, y, z))
        sphere_cloud = pv.PolyData(sphere_points)
        self.plotter.add_mesh(
            sphere_cloud, 
            color="white", 
            point_size=4, 
            render_points_as_spheres=True,
            opacity=0.3
        )
        
        self.plotter.set_background("black")
        self.plotter.add_axes()
        self.plotter.show(interactive_update=True)

    def update(self, satellites_data: list):
        # Remove previous satellite meshes
        for actor in self.dynamic_actors:
            self.plotter.remove_actor(actor)
        self.dynamic_actors.clear()

        r_earth = 6371e3
        for sat in satellites_data:
            orbit_points = sat['orbit_points']
            band_points = sat['band_points']
            current_pos = sat['current_pos']
            
            # Orbit path (Blue Line)
            scaled_orbit = orbit_points / r_earth
            closed_orbit = np.vstack([scaled_orbit, scaled_orbit[0]])
            orbit_poly = pv.PolyData(closed_orbit)
            orbit_poly.lines = np.hstack([[len(closed_orbit)], np.arange(len(closed_orbit))])
            actor1 = self.plotter.add_mesh(orbit_poly, color="blue", line_width=3)
            self.dynamic_actors.append(actor1)

            # Observation Band (Red dots)
            actor2 = self.plotter.add_mesh(
                pv.PolyData(band_points), 
                color="red", 
                point_size=5, 
                render_points_as_spheres=True
            )
            self.dynamic_actors.append(actor2)
            
            # Current Satellite Position (Green dot)
            current_scaled = current_pos / r_earth
            actor3 = self.plotter.add_mesh(
                pv.PolyData(current_scaled[None, :]),
                color="green",
                point_size=15,
                render_points_as_spheres=True
            )
            self.dynamic_actors.append(actor3)

        self.plotter.update()

def initialize_coverage_arrays(nside: int):
    """
    Initializes the three distinct HEALPix channels for the CNN.
    
    Args:
        nside: The resolution parameter.
        
    Returns:
        Three 1D np.ndarray arrays of length 12 * nside**2, initialized to zero.
    """
    npix = hp.nside2npix(nside)
    
    # Using float32 is recommended to save memory and align with PyTorch/CUDA defaults
    steady_state_map = np.zeros(npix, dtype=np.float32)
    current_coverage_map = np.zeros(npix, dtype=np.float32)
    priority_map = np.zeros(npix, dtype=np.float32)
    
    # Example: Pre-fill the priority map with a baseline weight of 1.0
    priority_map.fill(1.0)
    
    return steady_state_map, current_coverage_map, priority_map

def splat_to_healpix(
    points: np.ndarray, 
    point_weights: np.ndarray, 
    target_map: np.ndarray, 
    nside: int
) -> None:
    """
    Projects 3D points onto the HEALPix sphere and distributes their weights
    to the 4 nearest pixel centers using spherical interpolation weights.
    
    Args:
        points: (N, 3) array of Cartesian [X, Y, Z] points.
        point_weights: (N,) array of scalar values to add (e.g., coverage intensity).
        target_map: (N_pix,) the HEALPix array to be modified in-place.
        nside: The HEALPix resolution parameter.
    """
    # 1. Convert Cartesian [X, Y, Z] to Spherical [Theta, Phi]
    # Theta (colatitude) in [0, pi], Phi (longitude) in [0, 2pi]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    theta, phi = hp.vec2ang(np.vstack((x, y, z)), lonlat=False)
    
    # 2. Get the 4 nearest neighbors and their spherical interpolation weights
    # neighbors shape: (4, N) - The integer indices of the pixels
    # interp_weights shape: (4, N) - The distribution ratios summing to 1.0 per point
    neighbors, interp_weights = hp.get_interp_weights(nside, theta, phi, nest=True)
    
    # 3. Scale the interpolation weights by the actual value of each point
    # point_weights shape (N,) broadcasts across the 4 rows
    splat_values = interp_weights * point_weights
    
    # 4. Vectorized accumulation into the target array
    # np.add.at is required instead of target_map[neighbors] += splat_values
    # because multiple points might splat onto the exact same pixel in a single batch,
    # and standard indexing would overwrite rather than accumulate.
    np.add.at(target_map, neighbors, splat_values)

def extract_agent_observation(
    band_points: np.ndarray,
    orbit_points: np.ndarray,
    steady_state: np.ndarray,
    current_cov: np.ndarray,
    priority: np.ndarray,
    nside: int,
    num_orbit_points: int = 256
) -> np.ndarray:
    """
    Samples the 3 global maps and combines them with localized height 
    to form a 4-channel ego-centric image observation.
    
    Returns:
        np.ndarray of shape (4, num_rows, num_cols) where:
        Ch0: Steady State
        Ch1: Current Coverage
        Ch2: Priority
        Ch3: Satellite Altitude (km)
    """
    # 1. Get pixel indices for the projected band points
    pixels = map_xyz_to_healpix(band_points, nside)
    
    # 2. Sample the 3 global HEALPix maps (flattened)
    c0 = steady_state[pixels]
    c1 = current_cov[pixels]
    c2 = priority[pixels]
    
    # Calculate how many lateral rows exist in the band
    num_rows = len(band_points) // num_orbit_points
    
    # 3. Reshape sampled globals into 2D (rows, cols)
    img_c0 = c0.reshape(num_rows, num_orbit_points)
    img_c1 = c1.reshape(num_rows, num_orbit_points)
    img_c2 = c2.reshape(num_rows, num_orbit_points)
    
    # 4. Calculate the 4th channel (Satellite Height/Altitude)
    # The height changes over the orbit sequence, but is constant across cross-track rows.
    altitudes_km = (np.linalg.norm(orbit_points, axis=1) - 6371e3) / 1000.0
    img_c3 = np.tile(altitudes_km, (num_rows, 1)).astype(np.float32)
    
    # Combine into a 4-channel tensor
    return np.stack((img_c0, img_c1, img_c2, img_c3), axis=0)


# ==========================================
# Example Execution Pipeline
# ==========================================
if __name__ == "__main__":
    NSIDE = 16  # 3072 total pixels
    mu_earth = 3.986004418e14
    
    # Initialize the 3 global CNN channels
    steady_state, current_cov, priority = initialize_coverage_arrays(NSIDE)
    
    # Generate two random satellite orbits
    np.random.seed(42)
    satellites = []
    
    for i in range(2):
        # Random LEO-ish positions (around 7000 km altitude)
        pos_dir = np.random.randn(3)
        pos_dir /= np.linalg.norm(pos_dir)
        pos = pos_dir * (6371e3 + 600e3 + np.random.rand() * 200e3)
        
        # Circular velocity direction (perpendicular to pos)
        vel_dir = np.cross(pos_dir, np.random.randn(3))
        vel_dir /= np.linalg.norm(vel_dir)
        vel_mag = np.sqrt(mu_earth / np.linalg.norm(pos))
        vel = vel_dir * vel_mag
        
        # Slightly perturb velocity for some small eccentricity
        vel += np.random.randn(3) * 100.0
        
        satellites.append({
            'pos': pos,
            'vel': vel
        })

    salellite_scene_data = []

    for idx, sat in enumerate(satellites):
        # Move the satellite by a delta time to demonstrate step_orbit
        dt = 60.0 * 10.0 # Propagate 10 minutes forward
        r_new, v_new = step_orbit(sat['pos'], sat['vel'], dt, mu_earth)
        sat['pos'] = r_new
        sat['vel'] = v_new
        
        # Generate the physical orbit points
        orbit_pts = generate_temporal_orbit_points(sat['pos'], sat['vel'], num_points=256)
        
        # Generate band for satellite
        band_pts = generate_orbital_band_points(orbit_pts, N_deg=15.0, K_deg=3.0)
        
        # Splat orbit into Steady State
        coverage_weights = np.full(len(orbit_pts), 0.5, dtype=np.float32)
        splat_to_healpix(
            points=orbit_pts / np.linalg.norm(orbit_pts, axis=1, keepdims=True),
            point_weights=coverage_weights,
            target_map=steady_state,
            nside=NSIDE
        )
        
        # Splat current pos into Current Coverage
        current_pos_normalized = sat['pos'] / np.linalg.norm(sat['pos'])
        splat_to_healpix(
            points=current_pos_normalized[None, :],
            point_weights=np.array([2.0], dtype=np.float32),
            target_map=current_cov,
            nside=NSIDE
        )
        
        salellite_scene_data.append({
            'orbit_points': orbit_pts,
            'band_points': band_pts,
            'current_pos': sat['pos']
        })

    # Extract dynamic 4-channel ego-centric observations for each satellite
    for sat in salellite_scene_data:
        obs = extract_agent_observation(
            band_points=sat['band_points'],
            orbit_points=sat['orbit_points'],
            steady_state=steady_state,
            current_cov=current_cov,
            priority=priority,
            nside=NSIDE,
            num_orbit_points=256
        )
        sat['observation'] = obs

    print(f"Max steady state coverage pixel value: {steady_state.max():.3f}")
    print(f"Max current coverage pixel value: {current_cov.max():.3f}")
    print(f"Agent 0 Ego-centric observation shape: {salellite_scene_data[0]['observation'].shape}")
    
    # Visualize scene
    visualize_coverage_scene(nside=NSIDE, satellites_data=salellite_scene_data)