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

def visualize_coverage_band(
    nside: int, 
    orbit_points: np.ndarray, 
    band_points: np.ndarray
):
    """
    Uses PyVista to plot the HEALPix sphere, projected orbit, and coverage band.
    """
    # 1. Generate the white sphere points (HEALPix centers)
    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix), nest=True)
    sphere_points = np.column_stack((x, y, z))
    
    # 2. Project the blue orbit points onto the unit sphere
    norms = np.linalg.norm(orbit_points, axis=1, keepdims=True)
    projected_orbit = orbit_points / norms

    # 3. Initialize PyVista Plotter
    plotter = pv.Plotter()

    # Add HEALPix sphere points (White)
    sphere_cloud = pv.PolyData(sphere_points)
    plotter.add_mesh(
        sphere_cloud, 
        color="white", 
        point_size=2, 
        render_points_as_spheres=True,
        opacity=0.3 # Dimmed slightly so the orbit pops
    )

    # Add projected orbit points (Blue)
    orbit_cloud = pv.PolyData(projected_orbit)
    plotter.add_mesh(
        orbit_cloud, 
        color="blue", 
        point_size=8, 
        render_points_as_spheres=True
    )

    # Add band points (Red)
    band_cloud = pv.PolyData(band_points)
    plotter.add_mesh(
        band_cloud, 
        color="red", 
        point_size=5, 
        render_points_as_spheres=True
    )

    # Camera and rendering setup
    plotter.set_background("black")
    plotter.add_axes()
    plotter.show()

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


# ==========================================
# Example Execution Pipeline
# ==========================================
if __name__ == "__main__":
    NSIDE = 16  # 3072 total pixels
    
    # 1. Initialize the 3 CNN channels
    steady_state, current_cov, priority = initialize_coverage_arrays(NSIDE)
    
    # 2. Simulate 256 points from an orbit (from our previous step)
    # Here, we just generate random points normalized to the unit sphere for the example
    N_points = 256
    random_points = np.random.randn(N_points, 3)
    orbit_points = random_points / np.linalg.norm(random_points, axis=1, keepdims=True)
    
    # 3. Simulate the coverage weight each point provides
    # (e.g., higher altitude = broader but weaker weight)
    coverage_weights = np.full(N_points, 0.5, dtype=np.float32)
    
    # 4. Splat the orbit into the Steady State map
    splat_to_healpix(
        points=orbit_points,
        point_weights=coverage_weights,
        target_map=steady_state,
        nside=NSIDE
    )
    
    # 5. Simulate 5 active agents for the Current Coverage map
    agent_points = np.random.randn(5, 3)
    agent_points = agent_points / np.linalg.norm(agent_points, axis=1, keepdims=True)
    agent_weights = np.full(5, 2.0, dtype=np.float32) # Agents drop heavy, immediate weight
    
    # 6. Splat the agents into the Current Coverage map
    splat_to_healpix(
        points=agent_points,
        point_weights=agent_weights,
        target_map=current_cov,
        nside=NSIDE
    )
    
    print(f"Max steady state coverage pixel value: {steady_state.max():.3f}")
    print(f"Max current coverage pixel value: {current_cov.max():.3f}")

if __name__ == "__main__":
    # Assume generate_temporal_orbit_points is defined from the previous step
    
    # Example state: LEO-ish orbit
    pos = np.array([7000e3, 0.0, 0.0])         # 7000 km on X axis
    vel = np.array([0.0, 5000.0, 5500.0])      # Inclined velocity vector
    
    # Generate the 256 physical orbit points
    orbit_pts = generate_temporal_orbit_points(pos, vel, num_points=256)
    
    # Generate a +/- 15 degree band, with points every 3 degrees
    band_pts = generate_orbital_band_points(orbit_pts, N_deg=15.0, K_deg=3.0)
    
    # Visualize against a HEALPix sphere with nside=16 (3072 points)
    visualize_coverage_band(nside=16, orbit_points=orbit_pts, band_points=band_pts)