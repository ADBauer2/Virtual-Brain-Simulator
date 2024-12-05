import sys
from PyQt5.QtWidgets import QApplication
from Sim_Setup import generate_time_series  # Import the simulation logic
from skeleton_gui import App  # Import the GUI class

def main():
    app = QApplication(sys.argv)

    # Assuming `run_simulation` generates or loads your time_series and conn (connectivity data)
    time_series, conn = generate_time_series()  # Replace with actual function to get simulation data

    window = App(time_series, conn)
    window.show()

    # Start the simulation
    window.start_simulation()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()