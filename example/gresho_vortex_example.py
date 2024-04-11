import os
import sys
import shutil

import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.insert(0, "main")
import plot_utils as pu
import utils as ut
from main import Fluid2D

cwd = os.getcwd()


"""
SIMULATIION
"""

N = 250  # Dimensions of the domain
lim = 0.5  # Domain limits

# Get the initial conditions for the Gresho vortex
Ics = ut.VortexIcs(N, lim)

CFL_factor = 1
T = 3  # Final time

# Data saved as data.hdf5 in
output_folder = f"results/vortex/"

# Initialize the simulation class
Fluid = Fluid2D(Ics.x, Ics.y, Ics.D, Ics.U, Ics.V, Ics.P, CFL_factor=CFL_factor)
Fluid.set_output_folder(output_folder)

# Add tracers
fact = 0.9
M = 50
x_tracer = np.linspace(0, lim * fact, M)
y_tracer = np.linspace(0, -lim * fact, M)
Fluid.add_tracers(x_tracer, y_tracer)

# Run the simulation, and get 100 snapshots
Fluid.evaluate(T, snapshot_times=100)  # Around 10 seconds on my machine


"""
POSTPROCESSING AND PLOTTING
"""

# Postprocess the data, to compute Angular Momentum, etc.
ut.postprocess_data(Fluid.output_data_path)

# Plot the conserved quantities
save_path = output_folder + "conserved_quantites.jpg"
pu.plot_conserved_quantities(Fluid.output_data_path, save_path)

# Make a video of the snapshots
pu.animate_field(
    Fluid.output_data_path, output_folder, field_name="Velocity Magnitude", fps=20
)
