import sys
from PyQt5.QtWidgets import QApplication
from Sim_Setup import generate_time_series  # Import the simulation logic
from skeleton_gui import App  # Import the GUI class

def main():
    #Generate time series and connectivity data from the simulator
    time_series, conn = generate_time_series()

    #Initialize the Qt application
    app = QApplication(sys.argv)
    
    #Pass the simulation data to the GUI
    window = App(time_series, conn)  # Pass simulation data to the GUI
    
    #Show the window and start the simulation
    window.show()
    window.start_simulation()
    
    #Run the Qt application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()