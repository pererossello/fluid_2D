import os
import time

import numpy as np
from numpy.typing import NDArray
from numpy import float32
import h5py

"""
D: Density
U: Velocity in x-direction
V: Velocity in y-direction
P: Pressure
E: Energy
CFL_factor: Courant-Friedrichs-Lewy proportion factor
gamma: Adiabatic index
"""


class Fluid2D:
    """
    Class for solving the 2D Euler equations using the Lax-Friedrichs scheme
    """

    def __init__(
        self,
        X: NDArray[float32],
        Y: NDArray[float32],
        D: NDArray[float32],
        U: NDArray[float32],
        V: NDArray[float32],
        P: NDArray[float32],
        CFL_factor: float32 = 1,
        gamma: float32 = 1.4,
    ):

        self.X = X  # 1D array
        self.Y = Y  # 1D array
        self.D = D  # 2D array
        self.U = U  # 2D array
        self.V = V  # 2D array
        self.P = P  # 2D array
        self.CFL_factor = CFL_factor
        self.gamma = gamma

        # Initialization of arrays for the Lax-Friedrichs scheme
        self.DU = self.D * self.U
        self.DV = self.D * self.V
        self.DE = self.P / (self.gamma - 1) + 0.5 * self.D * (self.U**2 + self.V**2)

        self.dx = X[1] - X[0]
        self.dy = Y[1] - Y[0]

        self.L_x = X[-1] - X[0]
        self.L_y = Y[-1] - Y[0]

        self.one_over_dx = 1 / self.dx
        self.one_over_dy = 1 / self.dy

        # Output folder where to save data. Sepcify with set_output_folder()
        self.output_folder = None

        # Default with no tracers
        self.are_there_tracers = False

    def evaluate(self, T: float32, snapshot_times=None):

        if self.output_folder is None:
            print("No output_folder path provided. Exiting...")
            return

        # Handle snapshot times
        if snapshot_times is None:
            snapshot_times = np.linspace(0, T, 50)
        elif isinstance(snapshot_times, int):
            snapshot_times = np.linspace(0, T, snapshot_times)
        self.snapshot_times = snapshot_times

        self.time = 0.0
        self.step = 0
        self.snapshot_step = 0

        # Main loop for time integration
        clock_start = time.time()
        while self.time < T:

            # Save snapshots at predefined times
            if self.time >= snapshot_times[self.snapshot_step]:
                self.save_snapshot()
                self.snapshot_step += 1
                print(f"t: {self.time:0.2f}", end="\r")

            # Precompute for further use
            self.UU_VV = self.U**2 + self.V**2
            self.one_over_D = 1 / self.D

            # Compute time step using CFL condition
            self.compute_dt()

            if self.are_there_tracers:
                self.update_tracers()

            self.compute_step()

            self.step += 1
            self.time += self.dt

        clock_end = time.time()

        print(f"Compute time: {clock_end - clock_start:0.2e} s")

        # Save last frame
        self.save_snapshot()

    def update_tracers(self):

        # Get tracers velocity
        self.tracer_U = self.U[self.idx_y, self.idx_x]
        self.tracer_V = self.V[self.idx_y, self.idx_x]

        # Update tracers position using Euler step
        self.x_tracer = (
            self.x_tracer + self.tracer_U * self.dt - self.X[0]
        ) % self.L_x + self.X[0]
        self.y_tracer = (
            self.y_tracer + self.tracer_V * self.dt - self.Y[0]
        ) % self.L_y + self.Y[0]

        # Update tracers indices in grid
        self.idx_x = np.searchsorted(self.X, self.x_tracer)
        self.idx_y = np.searchsorted(self.Y, self.y_tracer)
        self.idx_x = np.clip(self.idx_x, 0, len(self.X) - 1)
        self.idx_y = np.clip(self.idx_y, 0, len(self.Y) - 1)

    def add_tracers(self, x, y):

        self.are_there_tracers = True

        self.x_tracer = x
        self.y_tracer = y

        # Set tracers indices in grid
        self.idx_x = np.searchsorted(self.X, x)
        self.idx_y = np.searchsorted(self.Y, y)
        self.idx_y = np.clip(self.idx_y, 0, len(self.Y) - 1)
        self.idx_x = np.clip(self.idx_x, 0, len(self.X) - 1)

        # Get tracers velocity
        self.tracer_U = self.U[self.idx_y, self.idx_x]
        self.tracer_V = self.V[self.idx_y, self.idx_x]

    def compute_step(self):
        """
        Compute the next step using the Lax-Friedrichs scheme
        """

        # Precopmute for ruther use
        DUV = self.DU * self.V

        # Continuity Equation update
        self.D = update_field(
            self.D, self.DU, self.DV, self.dt, self.one_over_dx, self.one_over_dy
        )

        # Momentum Equation update (X and Y coords)
        self.DU = update_field(
            self.DU,
            self.DU * self.U + self.P,
            DUV,
            self.dt,
            self.one_over_dx,
            self.one_over_dy,
        )
        self.DV = update_field(
            self.DV,
            DUV,
            self.DV * self.V + self.P,
            self.dt,
            self.one_over_dx,
            self.one_over_dy,
        )

        # Energy Equation update
        self.DE = update_field(
            self.DE,
            self.DE * self.U + self.P * self.U,
            self.DE * self.V + self.P * self.V,
            self.dt,
            self.one_over_dx,
            self.one_over_dy,
        )

        # Update fields
        self.U = self.DU * self.one_over_D
        self.V = self.DV * self.one_over_D
        self.P = (self.gamma - 1) * (self.DE - 0.5 * self.D * self.UU_VV)

    def compute_dt(self):
        """
        Compute the adaptative time step using the Courant-Friedrichs-Lewy condition
        """

        sound_speed = np.sqrt(self.gamma * self.P * self.one_over_D)

        dt_x = self.CFL_factor * self.dx / np.max(np.abs(self.U) + sound_speed)
        dt_y = self.CFL_factor * self.dy / np.max(np.abs(self.V) + sound_speed)

        self.dt = np.min([dt_x, dt_y])

    def set_output_folder(self, output_folder):
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.output_data_path = os.path.join(self.output_folder, "data.hdf5")

    def save_snapshot(self):
        mode = "w" if self.snapshot_step == 0 else "a"
        with h5py.File(self.output_data_path, mode) as f:
            if self.snapshot_step == 0:
                header_grp = f.create_group("Header")
                header_grp.attrs["X"] = self.X
                header_grp.attrs["Y"] = self.Y
                header_grp.attrs["Adiabatic Index"] = self.gamma

                if self.are_there_tracers:
                    header_grp.attrs["Tracers"] = 1
                else:
                    header_grp.attrs["Tracers"] = 0

                # Compute total angular momentum at first time-step
                area = self.dx * self.dy
                X, Y = np.meshgrid(self.X, self.Y)
                L = self.D * (X * self.V - Y * self.U)
                L_f = np.sum(L) * area
                header_grp.attrs["L_i"] = L_f

            if self.snapshot_step == 0:
                self.one_over_D = 1 / self.D

            E = self.DE * self.one_over_D

            step_group = f.create_group(f"{self.snapshot_step:03d}")
            step_group.create_dataset("Density", data=self.D, dtype=np.float32)
            step_group.create_dataset("Velocity X", data=self.U, dtype=np.float32)
            step_group.create_dataset("Velocity Y", data=self.V, dtype=np.float32)
            step_group.create_dataset("Pressure", data=self.P, dtype=np.float32)
            step_group.create_dataset("Specific Energy", data=E, dtype=np.float32)
            step_group.attrs["Time"] = self.time

            step_group.create_dataset("Tracer X", data=self.x_tracer, dtype=np.float32)
            step_group.create_dataset("Tracer Y", data=self.y_tracer, dtype=np.float32)

            if self.snapshot_step == len(self.snapshot_times) - 1:
                header_grp = f["Header"]
                header_grp.attrs["N"] = len(f.keys()) - 1

                f.attrs["Time"] = [
                    f[f"{i:03d}"].attrs["Time"] for i in range(header_grp.attrs["N"])
                ]

                # Compute total angular momentum at last time-step
                area = self.dx * self.dy
                X, Y = np.meshgrid(self.X, self.Y)
                L = self.D * (X * self.V - Y * self.U)
                L_f = np.sum(L) * area
                header_grp.attrs["L_f"] = L_f


"""
Auxiliary function
"""


def update_field(
    PHI: NDArray[float32],
    F_x: NDArray[float32],
    F_y: NDArray[float32],
    dt: float32,
    one_over_dx: float32,
    one_over_dy: float32,
) -> NDArray[float32]:
    """
    Update the field using the 2D generalization of the Lax-Friedrichs scheme
    """

    PHI_i_plus_1 = np.roll(PHI, shift=-1, axis=1)
    PHI_i_minus_1 = np.roll(PHI, shift=1, axis=1)
    PHI_j_plus_1 = np.roll(PHI, shift=-1, axis=0)
    PHI_j_minus_1 = np.roll(PHI, shift=1, axis=0)

    F_x_i_plus_1 = np.roll(F_x, shift=-1, axis=1)
    F_x_i_minus_1 = np.roll(F_x, shift=1, axis=1)

    F_y_j_plus_1 = np.roll(F_y, shift=-1, axis=0)
    F_y_j_minus_1 = np.roll(F_y, shift=1, axis=0)

    PHI_x = 0.5 * (PHI_i_plus_1 + PHI_i_minus_1) - 0.5 * dt * one_over_dx * (
        F_x_i_plus_1 - F_x_i_minus_1
    )
    PHI_y = 0.5 * (PHI_j_plus_1 + PHI_j_minus_1) - 0.5 * dt * one_over_dy * (
        F_y_j_plus_1 - F_y_j_minus_1
    )

    return 0.5 * (PHI_x + PHI_y)
