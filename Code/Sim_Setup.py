from tvb.simulator.lab import models, connectivity, coupling, monitors, simulator
import tvb_data
import os
import numpy as np

def generate_time_series():
    # Get the connectivity file path from the tvb_data package
    connectivity_path = os.path.join(tvb_data.__path__[0], "connectivity/connectivity_76.zip")

    if not os.path.exists(connectivity_path):
        raise FileNotFoundError(f"Connectivity file not found at {connectivity_path}. Ensure tvb-data is correctly installed.")

    # Load connectivity
    conn = connectivity.Connectivity.from_file(connectivity_path)

    # Configure simulation
    num_nodes = 76
    conn.speed = np.array([3.0])
    conn.weights[:num_nodes, :num_nodes]
    jansen_rit = models.JansenRit()
    linear_coupling = coupling.Linear(a=np.array([0.005]))
    integrator = simulator.integrators.HeunDeterministic(dt=0.01)
    raw_monitor = monitors.Raw()


    sim = simulator.Simulator(
        model=jansen_rit,
        connectivity=conn,
        coupling=linear_coupling,
        integrator=integrator,
        monitors=[raw_monitor]
    )
    sim.configure()

    # Run the simulation
    simulation_length = 1000
    raw_data, = sim.run(simulation_length=simulation_length)

    # Display results
    time_series = raw_data[1][:, :, 0]
    print(f"Time series shape: {time_series.shape}")

    return raw_data, conn



