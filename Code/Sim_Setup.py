from tvb.simulator.lab import models, connectivity, coupling, monitors, simulator
import tvb_data
import os
import numpy as np

def generate_time_series():
    # Get the connectivity file path
    connectivity_path = os.path.join(tvb_data.__path__[0], "connectivity/connectivity_76.zip")

    if not os.path.exists(connectivity_path):
        raise FileNotFoundError(f"Connectivity file not found at {connectivity_path}. Ensure tvb-data is correctly installed.")

    # Load connectivity
    conn = connectivity.Connectivity.from_file(connectivity_path)

    # Check if regions are properly loaded
    if conn.number_of_regions == 0:
        print("Warning: 'hemispheres' data is missing. Using 'areas' for configuration.")
        conn.number_of_regions = conn.weights.shape[0]
        conn.hemispheres = np.zeros(conn.number_of_regions, dtype=bool)  # Default: all in the left hemisphere
        conn.region_labels = np.array([f"Region-{i}" for i in range(conn.number_of_regions)])

        # Optional: Assign right hemisphere for the second half
        conn.hemispheres[conn.number_of_regions // 2:] = True
        conn.configure()


    # Validate connectivity data
    num_nodes = conn.number_of_regions
    if num_nodes != 76:
        raise ValueError(f"Expected 76 nodes, but connectivity has {num_nodes} nodes.")

    # Configure simulation
    conn.speed = np.array([3.0])
    jansen_rit = models.JansenRit()
    linear_coupling = coupling.Linear(a=np.array([0.005]))
    integrator = simulator.integrators.HeunDeterministic(dt=1.0)
    raw_monitor = monitors.Raw()

    sim = simulator.Simulator(
        model=jansen_rit,
        connectivity=conn,
        coupling=linear_coupling,
        integrator=integrator,
        monitors=[raw_monitor]
    )
    sim.configure()

    # Run the simulation for 1000 time points
    simulation_length = 1000
    raw_data, = sim.run(simulation_length=simulation_length)

    # Extract voltage state variables
    time_series = raw_data[1][0, :, :].T  # Transpose to get shape (76, 1000)
    print(f"Time series shape: {time_series.shape}")
    print(raw_data)

    return time_series, conn

