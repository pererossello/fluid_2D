import numpy as np
import h5py


class VortexIcs:

    def __init__(self, N, lim, gamma=1.4, direction="up", V_X=0, V_Y=0):

        # N should be integer or size 2 tuple of integers
        if isinstance(N, int):
            N = (N, N)
        elif isinstance(N, tuple) and len(N) == 2:
            N = tuple(int(n) for n in N)
        else:
            raise ValueError("N should be integer or size 2 tuple of integers")

        # if lim is a number, it is used as the domain limits in both directions, -lim and lim,
        # otherwise, it should be a tuple of two tuples of two numbers
        if isinstance(lim, (int, float)):
            lim = ((-lim, lim), (-lim, lim))
        elif isinstance(lim, tuple) and len(lim) == 2:
            lim = ((-lim[0], lim[0]), (-lim[1], lim[1]))
        else:
            raise ValueError(
                "lim should be a number or a tuple of two tuples of two numbers"
            )

        self.x = np.linspace(lim[0][0], lim[0][1], N[0])
        self.y = np.linspace(lim[1][0], lim[1][1], N[1])
        self.N = N
        self.gamma = gamma

        X, Y = np.meshgrid(self.x, self.y)
        R = np.sqrt(X**2 + Y**2)

        v_phi = np.zeros_like(R)
        pressure = np.zeros_like(R)

        # logic for setting the initial conditons of the vortex
        bool_1 = R <= 0.2
        bool_2 = (R > 0.2) & (R < 0.4)
        bool_3 = R >= 0.4

        v_phi[bool_1] = 5 * R[bool_1]
        v_phi[bool_2] = 2 - 5 * R[bool_2]
        v_phi[bool_3] = 0

        pressure[bool_1] = 5 + 12.5 * R[bool_1] ** 2
        pressure[bool_2] = (
            9 + 12.5 * R[bool_2] ** 2 - 20 * R[bool_2] + 4 * np.log(5 * R[bool_2])
        )
        pressure[bool_3] = 3 + 4 * np.log(2)

        self.U = -v_phi * Y / R + V_X
        self.V = v_phi * X / R + V_Y

        self.D = np.full_like(R, 1, dtype=np.float32)
        self.P = np.array(pressure, dtype=np.float32)
        self.U = np.array(self.U, dtype=np.float32)
        self.V = np.array(self.V, dtype=np.float32)

        if direction == "down":
            self.U = -self.U
            self.V = -self.V


class WaveIcs:

    def __init__(self, N, lim, gamma=1.4):

        # N should be integer or size 2 tuple of integers
        if isinstance(N, int):
            N = (N, N)
        elif isinstance(N, tuple) and len(N) == 2:
            N = tuple(int(n) for n in N)
        else:
            raise ValueError("N should be integer or size 2 tuple of integers")

        # if lim is a number, it is used as the domain limits in both directions, -lim and lim,
        # otherwise, it should be a tuple of two typles of two numbers
        if isinstance(lim, (int, float)):
            lim = ((-lim, lim), (-lim, lim))
        elif isinstance(lim, tuple) and len(lim) == 2:
            lim = tuple(tuple(float(l) for l in lim) for lim in lim)
        else:
            raise ValueError(
                "lim should be a number or a tuple of two tuples of two numbers"
            )

        self.x = np.linspace(lim[0][0], lim[0][1], N[0])
        self.y = np.linspace(lim[1][0], lim[1][1], N[1])

        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.N = N
        self.gamma = gamma

    def set_perturbation(
        self,
        AD,
        k_x,
        k_y,
        D0=1,
        P0=1,
        phi=0,
    ):

        sound_speed = np.sqrt(
            self.gamma * P0 / D0
        )  # Speed of sound of isothermal unperturbed state

        AP = sound_speed**2 * AD

        k = np.sqrt(k_x**2 + k_y**2)

        AU = k_x / k * sound_speed * AD / D0
        AV = k_y / k * sound_speed * AD / D0

        self.D = D0 + AD * np.cos(k_x * self.X + k_y * self.Y + phi)

        self.U = AU * np.cos(k_x * self.X + k_y * self.Y + phi)

        self.V = AV * np.cos(k_x * self.X + k_y * self.Y + phi)

        self.P = P0 + AP * np.cos(k_x * self.X + k_y * self.Y + phi)


def postprocess_data(data_path):

    with h5py.File(data_path, "a") as file:
        M = file["Header"].attrs["N"]
        time = file.attrs["Time"]
        x = file["Header"].attrs["X"]
        y = file["Header"].attrs["Y"]
        gamma = file["Header"].attrs["Adiabatic Index"]

        area_element = (x[1] - x[0]) * (y[1] - y[0])

        mass = np.zeros(M)
        energy = np.zeros(M)
        momentum = np.zeros(M)
        angular_momentum = np.zeros(M)
        mean_pressure = np.zeros(M)

        X, Y = np.meshgrid(x, y)

        for i in range(M):
            file_step = file[f"{i:03d}"]

            v_x = file_step["Velocity X"][()]
            v_y = file_step["Velocity Y"][()]
            specific_energy = file_step["Specific Energy"][()]
            density = file_step["Density"][()]
            pressure = file_step["Pressure"][()]

            v_magnitude = np.sqrt(v_x**2 + v_y**2)
            file_step.create_dataset(
                "Velocity Magnitude", data=v_magnitude, dtype=np.float32
            )

            angular_momentum_density = X * v_y - Y * v_x
            file_step.create_dataset(
                "Angular Momentum Density",
                data=angular_momentum_density,
                dtype=np.float32,
            )

            momentum_density_x = v_x * density
            momentum_density_y = v_y * density
            file_step.create_dataset(
                "Momentum Density X", data=momentum_density_x, dtype=np.float32
            )
            file_step.create_dataset(
                "Momentum Density Y", data=momentum_density_y, dtype=np.float32
            )

            mass[i] = np.sum(density) * area_element
            energy[i] = np.sum(density * specific_energy) * area_element
            angular_momentum[i] = np.sum(angular_momentum_density) * area_element
            momentum_x = np.sum(momentum_density_x) * area_element
            momentum_y = np.sum(momentum_density_y) * area_element
            momentum[i] = np.sqrt(momentum_x**2 + momentum_y**2)
            mean_pressure[i] = np.mean(pressure)

        file.attrs["Mass"] = mass
        file.attrs["Energy"] = energy
        file.attrs["Momentum"] = momentum
        file.attrs["Angular Momentum"] = angular_momentum
        file.attrs["Mean Pressure"] = mean_pressure
