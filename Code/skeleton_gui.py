import sys
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Set up the PyQt5 GUI with matplotlib embedded
class App(QMainWindow):
    def __init__(self, time_series, conn):
        super().__init__()
        self.setWindowTitle("Real-Time Brain Simulation")
        self.setGeometry(100, 100, 800, 800)
        
        # Store simulation data and connectivity information
        self.time_series = time_series
        self.conn = conn
        self.num_nodes = len(conn.region_labels)
        self.current_frame = 0

        # Initialize NetworkX graph from connectivity
        self.G = nx.Graph()
        for i in range(self.num_nodes):
            self.G.add_node(i, label=conn.region_labels[i])
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if conn.weights[i, j] > 0:
                    self.G.add_edge(i, j, weight=conn.weights[i, j])

        # Set up the matplotlib canvas for 3D
        self.canvas = plt.Figure()
        self.ax = self.canvas.add_subplot(111, projection='3d')  # Use 3D axes
        self.figure_widget = FigureCanvas(self.canvas)
        
        # Layout for embedding the matplotlib figure
        layout = QVBoxLayout()
        layout.addWidget(self.figure_widget)
        
        # Container widget and main window layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Timer for real-time updates
        self.timer = QTimer()
        self.timer.setInterval(50)  # Update every 50 ms
        self.timer.timeout.connect(self.update_plot)
        
    def start_simulation(self):
        self.timer.start()  # Start the timer to update the plot

    def update_plot(self):
        # Loop the frames by resetting to the beginning if we reach the end of the time series
        if self.current_frame >= len(self.time_series):
            self.current_frame = 0  # Reset to the beginning for looped playback
        
        # Clear current plot
        self.ax.clear()

        # Extract current activity data for the current frame
        activity = self.time_series[self.current_frame]

        # Ensure activity has the correct number of nodes
        if len(activity) != self.num_nodes:
            print(f"Warning: Activity data length {len(activity)} does not match number of nodes {self.num_nodes}")
            return  # Skip this frame if the lengths don't match
        
        # Create 3D position for the nodes (using spring_layout but 3D)
        pos = nx.spring_layout(self.G, dim=3, seed=42)  # Get 3D positions for nodes (with a fixed seed for reproducibility)
        
        # Set node colors and sizes based on activity data
        node_colors = np.array([activity[i] for i in range(self.num_nodes)])
        node_sizes = np.array([50 + 500 * abs(activity[i]) for i in range(self.num_nodes)])

        # Ensure node_sizes is a 1D array (if not already)
        if node_sizes.ndim > 1:
            node_sizes = node_sizes.flatten()

        # Make sure node_sizes and node_colors are the same length as the number of nodes
        assert len(node_sizes) == len(node_colors) == self.num_nodes, "Node sizes and colors must match number of nodes"

        # Draw nodes in 3D
        self.ax.scatter([pos[i][0] for i in range(self.num_nodes)], 
                        [pos[i][1] for i in range(self.num_nodes)], 
                        [pos[i][2] for i in range(self.num_nodes)], 
                        s=node_sizes, c=node_colors, cmap=plt.cm.viridis, alpha=0.8)

        # Draw edges (in 3D)
        for edge in self.G.edges:
            x_vals = [pos[edge[0]][0], pos[edge[1]][0]]
            y_vals = [pos[edge[0]][1], pos[edge[1]][1]]
            z_vals = [pos[edge[0]][2], pos[edge[1]][2]]
            self.ax.plot(x_vals, y_vals, z_vals, color='gray', alpha=0.5)

        # Update title and refresh the canvas
        self.ax.set_title(f"Neural Activity at time {self.current_frame * 0.1:.1f} ms")
        
        # Adjust the 3D plot view angle for better visualization (you can adjust as needed)
        self.ax.view_init(elev=30, azim=30)  # Change the view angle
        
        self.figure_widget.draw()  # Refresh the plot
        self.current_frame += 1  # Move to the next frame

