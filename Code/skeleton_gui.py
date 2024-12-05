import sys
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import numpy as np

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

        # Set up layout and initial positions
        self.pos = nx.spring_layout(self.G)

        # Set up the matplotlib canvas
        self.canvas = plt.Figure()
        self.ax = self.canvas.add_subplot(111)
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
        self.timer.start()

    def update_plot(self):
        if self.current_frame >= len(self.time_series):
            self.timer.stop()  # Stop after all frames have been displayed
            return

        # Clear current plot
        self.ax.clear()

        # Extract current activity data
        activity = self.time_series[self.current_frame]

        # Set node colors and sizes based on activity data
        node_colors = [activity[i] for i in range(self.num_nodes)]
        node_sizes = [50 + 500 * abs(activity[i]) for i in range(self.num_nodes)]

        # Draw nodes, edges, and labels
        nx.draw_networkx_nodes(self.G, self.pos, node_size=node_sizes, node_color=node_colors,
                               cmap=plt.cm.viridis, alpha=0.8, ax=self.ax)
        nx.draw_networkx_edges(self.G, self.pos, edgelist=self.G.edges, width=0.5, alpha=0.5, ax=self.ax)
        nx.draw_networkx_labels(self.G, self.pos, labels={i: self.conn.region_labels[i] for i in range(self.num_nodes)}, ax=self.ax)

        # Update title and refresh the canvas
        self.ax.set_title(f"Neural Activity at time {self.current_frame * 0.1:.1f} ms")
        self.figure_widget.draw()
        self.current_frame += 1

