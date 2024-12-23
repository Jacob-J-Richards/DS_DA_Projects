#!/usr/bin/env python
# First make sure you have the required libraries installed.
# Open your terminal/command prompt and run:
#   pip install matplotlib
#   pip install numpy

import matplotlib.pyplot as plt
import numpy as np

def create_sine_wave_plot():
    # Generate sample data points
    x = np.linspace(0, 10, 100)  # Creates 100 evenly spaced points from 0 to 10
    y = np.sin(x)                # Calculate sine for each x value
    
    # Create and configure the plot
    plt.figure(figsize=(8, 6))   # Set figure size in inches
    plt.plot(x, y, 'b-', label='Sine Wave')  # Create line plot (blue line)
    plt.title('Simple Sine Wave Plot')
    plt.xlabel('X axis')
    plt.ylabel('Y axis') 
    plt.grid(True)               # Add grid lines
    plt.legend()                 # Show legend
    
    # Save the plot as an image file
    plt.savefig('sine_wave_plot.png')
    print("Plot has been saved as 'sine_wave_plot.png' in the current directory")
    
    # Close the plot to free memory
    plt.close()

# Run the function to create and save the plot
if __name__ == "__main__":
    create_sine_wave_plot()
