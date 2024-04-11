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

# Get the initial conditions for the wave
AD = 1e-1  # Amplitude of density perturbation
k_x, k_y = 1 * 2 * np.pi, 0 * 2 * np.pi  # wave propagating on x direction
D0, P0 = 1, 1  # Background density and pressure

Ics = ut.WaveIcs(N, lim)
Ics.set_perturbation(AD, k_x, k_y, D0=D0, P0=P0)


CFL_factor = 1
T = 5  # Final time

# Data saved as data.hdf5 in
output_folder = f"results/waves/"

# Initialize the simulation class
Fluid = Fluid2D(Ics.x, Ics.y, Ics.D, Ics.U, Ics.V, Ics.P, CFL_factor=CFL_factor)
Fluid.set_output_folder(output_folder)

# Add tracers :)
fact = 0.9
M = 50
phis = np.linspace(0, 2 * np.pi, M)
R = 0.2
x_tracer = (
    list(R * np.cos(phis))
    + list(R * 0.7 * np.cos(phis[M // 2 :]))
    + list(R * 0.1 * np.cos(phis[::5]) + 0.075)
    + list(R * 0.1 * np.cos(phis[::5]) - 0.075)
)
y_tracer = (
    list(R * np.sin(phis))
    + list(R * 0.7 * np.sin(phis[M // 2 :]))
    + list(R * 0.1 * np.sin(phis[::5]) + 0.1)
    + list(R * 0.1 * np.sin(phis[::5]) + 0.1)
)

Fluid.add_tracers(np.array(x_tracer), np.array(y_tracer))

# Run the simulation, and get 100 snapshots
Fluid.evaluate(T, snapshot_times=100)  # Around 22 seconds on my machine


"""
POSTPROCESSING AND PLOTTING
"""

# Postprocess the data, to compute Angular Momentum, etc.
ut.postprocess_data(Fluid.output_data_path)

# Plot the conserved quantities
save_path = output_folder + "conserved_quantites.jpg"
pu.plot_conserved_quantities(Fluid.output_data_path, save_path)

# Make a video of the snapshots
pu.animate_field(Fluid.output_data_path, output_folder, field_name="Pressure", fps=20)

"""
Possible field names for the plotting above (directly read from the HDF5 file):
Pressure, Density, Velocity Magnitude, Velocity X, Velocity Y, Angular Momentum Density, Momentum Density X, Momentum Density Y
"""
