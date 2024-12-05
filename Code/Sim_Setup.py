from tvb.simulator.lab import models, connectivity, coupling, monitors, simulator
import numpy as np

# Load connectivity and set up model and simulator
conn = connectivity.Connectivity.from_file()
conn.speed = 3.0  # Propagation speed

# Set up a Jansen-Rit neural mass model for each region
jansen_rit = models.JansenRit()

# Define coupling and integrator
linear_coupling = coupling.Linear(a=0.015)
integrator = simulator.integrators.HeunDeterministic(dt=0.1)

# Set up a monitor to collect raw output data
raw_monitor = monitors.Raw()

# Configure and run the simulation
sim = simulator.Simulator(
    model=jansen_rit,
    connectivity=conn,
    coupling=linear_coupling,
    integrator=integrator,
    monitors=[raw_monitor]
)
sim.configure()

# Run simulation for a specified time duration in ms
simulation_length = 1000  # in ms
(raw_data,), = sim.run(simulation_length=simulation_length)
time_series = raw_data[1][:, :, 0]  # Extract activity for each node over time
