import numpy as np
from tvb.simulator.lab import models, connectivity, coupling, monitors, simulator

# Set the path to the downloaded connectivity file
conn = connectivity.Connectivity.from_file("path/to/connectivity_76.zip")

# Set conduction speed as an array (one value for each region or a single value for all)
conn.speed = np.array([3.0])  # Propagation speed for all regions

# Configure the model, coupling, and integrator
jansen_rit = models.JansenRit()
linear_coupling = coupling.Linear(a=0.015)
integrator = simulator.integrators.HeunDeterministic(dt=0.1)
raw_monitor = monitors.Raw()

# Set up and configure the simulator
sim = simulator.Simulator(
    model=jansen_rit,
    connectivity=conn,
    coupling=linear_coupling,
    integrator=integrator,
    monitors=[raw_monitor]
)
sim.configure()

# Run simulation for 1000 ms
simulation_length = 1000  # Duration in ms
(raw_data,), = sim.run(simulation_length=simulation_length)

# Extract time series data
time_series = raw_data[1][:, :, 0]
print(f"Time series shape: {time_series.shape}")

