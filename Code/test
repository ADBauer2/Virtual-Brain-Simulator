import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.widgets import Button
import time


pressed = None
on_time = None
off_time = None
import numpy as np

# Define 3D coordinates for brain regions (nodes)
nodes = np.array([
    [20, -30, 60],   # Primary Somatosensory Cortex (Lateral, near central sulcus)
    [-20, -100, 10],  # Primary Visual Cortex (Occipital lobe)
    [-60, -20, 5],    # Primary Auditory Cortex (Temporal lobe)
    [30, -20, 60],    # Primary Motor Cortex (Near central sulcus)
    [40, 30, 50],     # Prefrontal Cortex (Anterior frontal lobe)
    [20, -50, 50],    # Parietal Association Cortex (Near Parieto-occipital sulcus)
    [-60, -40, 20],   # Temporal Association Cortex (Temporal lobe)
    [10, -10, 20],    # Basal Ganglia (Subcortical, near internal capsule)
    [0, -20, 10],     # Thalamus (Midline, near 3rd ventricle)
    [-25, -30, -15],  # Hippocampus (Medial temporal lobe, near midline)
    [-20, -30, -10],  # Amygdala (Medial temporal lobe, near hippocampus)
    [30, -80, -35],   # Cerebellum (Posterior fossa)
    [0, 30, 40],      # Cingulate Cortex (Medial frontal lobe, superior to corpus callosum)
    [0, 0, -50]       # Brainstem (Midline, near pons and medulla)
])

# Calculate the center of the current brain regions
current_center = np.mean(nodes, axis=0)

# Desired center position for the brain (50, 50, 50)
desired_center = np.array([50, 50, 50])

# Shift the brain regions to center them at the desired position
adjusted_nodes = nodes - current_center + desired_center
print(adjusted_nodes)

# Define connections (edges) between nodes (indices correspond to the nodes array)
connections = [
    (1, 5),  # Primary Visual Cortex ↔ Parietal Association Cortex
    (2, 6),  # Primary Auditory Cortex ↔ Temporal Association Cortex
    (4, 5),  # Prefrontal Cortex ↔ Parietal Association Cortex
    (3, 12), # Primary Motor Cortex ↔ Cerebellum
    (0, 3),  # Primary Somatosensory Cortex ↔ Primary Motor Cortex
    (4, 7),  # Prefrontal Cortex ↔ Basal Ganglia
    (3, 7),  # Primary Motor Cortex ↔ Basal Ganglia
    (9, 4),  # Hippocampus ↔ Prefrontal Cortex
    (10, 4), # Amygdala ↔ Prefrontal Cortex
    (8, 1),  # Thalamus ↔ Primary Visual Cortex
    (8, 3),  # Thalamus ↔ Primary Motor Cortex
    (8, 5),  # Thalamus ↔ Parietal Association Cortex
    (7, 8),  # Basal Ganglia ↔ Thalamus
    (7, 3),  # Basal Ganglia ↔ Primary Motor Cortex
    (7, 13), # Basal Ganglia ↔ Cingulate Cortex
    (9, 10), # Hippocampus ↔ Amygdala
    (13, 4), # Cingulate Cortex ↔ Prefrontal Cortex
    (13, 5), # Cingulate Cortex ↔ Parietal Association Cortex
    (12, 8), # Brainstem ↔ Thalamus
    (12, 11), # Brainstem ↔ Cerebellum
    (12, 4),  # Brainstem ↔ Prefrontal Cortex
    (12, 2),  # Brainstem ↔ Primary Auditory Cortex
    (12, 10), # Brainstem ↔ Amygdala
    (11, 5),  # Cerebellum ↔ Parietal Association Cortex
    (11, 4),  # Cerebellum ↔ Prefrontal Cortex
    (11, 3),  # Cerebellum ↔ Primary Motor Cortex
    (11, 7),  # Cerebellum ↔ Basal Ganglia
    (11, 0),  # Cerebellum ↔ Primary Somatosensory Cortex
    (11, 9),  # Cerebellum ↔ Hippocampus
    (4, 1),   # Prefrontal Cortex ↔ Primary Visual Cortex
    (2, 1),   # Prefrontal Cortex ↔ Primary Auditory Cortex
]

# Dictionary mapping node indices to brain area names
node_names = {
    0: "Primary Somatosensory Cortex",
    1: "Primary Visual Cortex",
    2: "Primary Auditory Cortex",
    3: "Primary Motor Cortex",
    4: "Prefrontal Cortex",
    5: "Parietal Association Cortex",
    6: "Temporal Association Cortex",
    7: "Basal Ganglia",
    8: "Thalamus",
    9: "Hippocampus",
    10: "Amygdala",
    11: "Cerebellum",
    12: "Cingulate Cortex",
    13: "Brainstem"
}



# Function to simulate activity over time (you can replace this with your model)
def simulate_activity(t):
    return np.abs(np.sin(0.1 * t + np.linspace(0, 2 * np.pi, len(adjusted_nodes))))  # Activity between 0 and 1

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a colormap using matplotlib
cmap = cm.viridis  # You can change this to any colormap you prefer

# Add connections (edges) as lines (Initially, show all connections)
edges = []
for start_idx, end_idx in connections:
    start_point = adjusted_nodes[start_idx]
    end_point = adjusted_nodes[end_idx]
    edge, = ax.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    [start_point[2], end_point[2]], color='gray', lw=1)
    edges.append(edge)

# Plot the nodes (with initial color)
node_scatter = ax.scatter([], [], [], c=[], cmap=cmap, s=100)

# Set axis limits for better visualization
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_zlim([0, 100])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Variables to control the animation
is_playing = False
frame = 0
selected_node = None  # No node is selected initially

# Text Box for the selected node activity
activity_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

# Function to update the activity display in the legend or text box
def update_activity_display(node_idx, activity_data):
    activity_text.set_text(f"Node {node_names[node_idx]} Activity: {activity_data[node_idx]:.2f}")
    fig.canvas.draw()

# Function to project 3D nodes to 2D based on the current view
def project_to_2d(adjusted_nodes, ax):
    """Projects 3D points to 2D based on the current view angle of the plot."""
    # Get the projection matrix
    proj_matrix = ax.get_proj()

    # Add a 1 to the node coordinates for homogeneous coordinates
    ones = np.ones((adjusted_nodes.shape[0], 1))
    nodes_homogeneous = np.hstack((adjusted_nodes, ones))

    # Apply the projection matrix to the nodes (this gives us the 2D projected coordinates)
    projected = nodes_homogeneous @ proj_matrix.T

    # Normalize by the z coordinate to perform the perspective divide (homogeneous to 2D)
    projected = projected[:, :2] / projected[:, 2:3]

    # Invert the X and Y axes for proper Cartesian coordinate system behavior
    projected[:, 0] = -projected[:, 0]  # Flip the X-axis
    projected[:, 1] = -projected[:, 1]  # Flip the Y-axis

    return projected


# Function to get mouse position
def get_mouse_position(event):
    if event.xdata is not None and event.ydata is not None:
        return np.array([event.xdata, event.ydata])
    return None

# Function to find the closest node based on 2D distance
def closest_node(coord_pair, projected_nodes):
    distances = np.linalg.norm(projected_nodes - coord_pair, axis=1)  # Compute distance in 2D space
    closest_index = np.argmin(distances)
    return closest_index, distances

# Function to update the node and connection visibility based on the selection
def update_connections(t):
    global selected_node

    # Get the activity data for the current time step
    activity_data = simulate_activity(t)
    
    # Update the node colors based on the activity
    node_scatter._offsets3d = (adjusted_nodes[:, 0], adjusted_nodes[:, 1], adjusted_nodes[:, 2])  # Update node positions
    node_scatter.set_array(activity_data)  # Update node colors based on activity

    # If a node is selected, show only its connections
    if selected_node is not None:
        # Hide all edges
        for edge in edges:
            edge.set_visible(False)
        
        # Show only the selected node's connections
        for start_idx, end_idx in connections:
            if start_idx == selected_node or end_idx == selected_node:
                # If the edge contains the selected node, show it
                edges[connections.index((start_idx, end_idx))].set_visible(True)

        # Update the activity display for the selected node
        update_activity_display(selected_node, activity_data)

    else:
        # Show all edges if no node is selected
        for edge in edges:
            edge.set_visible(True)

# Function to check if a mouse click is within a node's coordinates in 2D space
def check_node_click(event, projected_nodes):
    coords = get_mouse_position(event)
    if coords is None:
        return None
    
    node_to_select, distances = closest_node(coords, projected_nodes)
    # If the distance is within a threshold, consider it a click
    if distances[node_to_select] < 0.1:  # You can adjust the threshold
        return node_to_select
    else:
        return None

# Mouse click event handler to select a node
def on_click(event):
    
    global selected_node, pressed
    print(off_time-on_time)
    if pressed == False and off_time-on_time < 0.3:
        # Project the nodes to 2D space based on the current view
        projected_nodes = project_to_2d(adjusted_nodes, ax)

        selected_node = check_node_click(event, projected_nodes)
        
        if selected_node is not None:
            update_activity_display(selected_node, simulate_activity(0))  # Display initial activity
        else:
            print("No node selected")
    else:
        print(pressed)

# Mouse press event handler
def on_press(event):
    global pressed, on_time
    on_time = time.time()
    pressed = True

# Mouse release event handler
def on_release(event):
    global pressed, off_time
    off_time = time.time()
    pressed = False

# Add mouse click event to select nodes
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('button_release_event', on_click)

# Create the animation
ani = animation.FuncAnimation(fig, update_connections, frames=100, interval=50, blit=False)

# Variables to control the animation
is_playing = False
frame = 0

# Function to start the animation
def start_animation(event):
    global is_playing
    is_playing = True
    ani.event_source.start()

# Function to stop the animation
def stop_animation(event):
    global is_playing
    is_playing = False
    ani.event_source.stop()

# Function to step forward in the animation
def step_forward(event):
    global frame
    if frame < 100:
        update_connections(frame)  # Update once for the current frame
        frame += 1
        plt.draw()

# Add buttons for control
ax_start = plt.axes([0.1, 0.01, 0.1, 0.075])  # Button placement
ax_stop = plt.axes([0.25, 0.01, 0.1, 0.075])
ax_step = plt.axes([0.4, 0.01, 0.1, 0.075])

button_start = Button(ax_start, 'Start')
button_stop = Button(ax_stop, 'Stop')
button_step = Button(ax_step, 'Step Forward')

# Attach the button actions
button_start.on_clicked(start_animation)
button_stop.on_clicked(stop_animation)
button_step.on_clicked(step_forward)

# Display the plot with buttons
plt.show()
