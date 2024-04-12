import os
import shutil
import subprocess

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image
import PIL
import h5py


def plot_conserved_quantities(data_path, save_path):

    lw = 0.4
    ts = 2
    Fig = Figure(4, 1, ratio=0.75, fig_size=1080, hspace=0.0)
    axs = Fig.get_axes(flat=True)
    fs = Fig.fs

    with h5py.File(data_path, "r") as file:

        time = file.attrs["Time"][()]

        mass = file.attrs["Mass"][()]
        energy = file.attrs["Energy"][()]
        momentum = file.attrs["Momentum"][()]
        angular_momentum = file.attrs["Angular Momentum"][()]

        fields = [angular_momentum, mass, energy, momentum]

    ts_fact = 0.75
    txt_mean = [
        r"\langle L \rangle",
        r"\langle M \rangle",
        r"\langle E \rangle",
        r"\langle p \rangle",
    ]
    txt_std = [r"\sigma_L", r"\sigma_M", r"\sigma_E", r"\sigma_p"]
    label = ["$L$", "$M$", "$E$", "$p$"]
    for i, field in enumerate(fields):
        axs[i].plot(time, field, lw=lw * fs, c="lightcoral")
        axs[i].set_xlim([time[0], time[-1]])

        axs[i].text(
            0.975,
            0.975,
            f"${txt_mean[i]} = {np.mean(field):0.2e}$\n${txt_std[i]} = {np.std(field):0.2e}$",
            transform=axs[i].transAxes,
            fontsize=fs * ts * ts_fact,
            ha="right",
            va="top",
        )

        axs[i].set_ylabel(label[i], fontsize=fs * ts)

    Fig.save(save_path)
    plt.close()


def animate_field(
    data_path,
    output_folder,
    field_name,
    cmap="coolwarm",
    n_plots=None,
    fps=20,
):
    n_plots = None

    save_folder = output_folder + "frames/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)

    with h5py.File(data_path, "r") as file:

        M = file["Header"].attrs["N"]
        x = file["Header"].attrs["X"]
        y = file["Header"].attrs["Y"]
        are_there_tracers = file["Header"].attrs["Tracers"]

        X, Y = np.meshgrid(x, y)

        if n_plots is None:
            n_plots = M
        idxs = np.linspace(0, M - 1, n_plots, dtype=int)

        cmap = "coolwarm"
        Fig = Figure(1, 1, fig_size=1080, ratio=1)
        Fig.grid = False
        ax = Fig.get_axes()
        fs = Fig.fs
        ts = Fig.ts

        idx = 0
        i = 0

        for k in range(M):
            step_group = file[f"{k:03d}"]
            # field = (
            #     step_group["Velocity X"][()] ** 2 + step_group["Velocity Y"][()] ** 2
            # )
            # field = np.sqrt(field)
            field = step_group[field_name][()]
            if k == 0:
                max_field = np.max(field)
                min_field = np.min(field)
            else:
                max_field = np.max([max_field, np.max(field)])
                min_field = np.min([min_field, np.min(field)])

            min_max = (min_field, max_field)

        ax.set_aspect("equal", "box")
        ax.text(
            0.5,
            1.05,
            field_name,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=fs * ts * 1.2,
            color="k",
        )
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])

        for i, idx in enumerate(idxs):
            ims = []

            step_group = file[f"{i:03d}"]

            time = step_group.attrs["Time"]

            time_text = ax.text(
                0.02,
                0.98,
                f"t={time:0.2f}",
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=fs * ts * 0.8,
                color="k",
            )

            x_tracer = np.array(step_group["Tracer X"])
            y_tracer = np.array(step_group["Tracer Y"])
            field = (
                step_group["Velocity X"][()] ** 2 + step_group["Velocity Y"][()] ** 2
            )
            field = np.sqrt(field)

            im = ax.pcolormesh(X, Y, field, vmin=min_max[0], vmax=min_max[1], cmap=cmap)
            j = 0
            if i == 0:
                cbar = Fig.fig.colorbar(
                    im,
                    ax=ax,
                    orientation="vertical",
                    pad=0.02 * fs,
                    aspect=20,
                    shrink=0.79,
                )
                Fig.customize_axes(cbar.ax, ylabel_pos="right")
                # cbar.ax.tick_params(labelsize=fs*ts)
                # cbar.set_label(field_name, fontsize=fs*ts)

            ims.append(im)

            if are_there_tracers:
                s = ax.scatter(x_tracer, y_tracer, c="k", s=fs * 0.35, lw=0)

            image_path = save_folder + f"frame_{i:04d}.jpg"
            Fig.save(image_path, bbox_inches="tight")

            plt.close()

            time_text.remove()
            for im in ims:
                im.remove()

            if are_there_tracers:
                s.remove()

            print(f"Saving frames {i+1}/{n_plots}", end="\r")

    print("\n")

    save_folder = output_folder + "frames/"
    video_name = field_name.replace(" ", "_") + "_animation"
    frames_to_mp4(save_folder, fps=fps, title=video_name, extension=".jpg")

    # Delete frames
    shutil.rmtree(save_folder)


class Figure:

    def __init__(
        self,
        subplot_1=1,
        subplot_2=1,
        fig_size=540,
        ratio=2,
        dpi=300,
        width_ratios=None,
        height_ratios=None,
        hspace=None,
        wspace=None,
    ):

        self.fig_size = fig_size
        self.ratio = ratio
        self.dpi = dpi
        self.subplot_1 = subplot_1
        self.subplot_2 = subplot_2
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.hspace = hspace
        self.wspace = wspace

        fig_width, fig_height = fig_size * ratio / dpi, fig_size / dpi
        fs = np.sqrt(fig_width * fig_height)
        self.fs = fs

        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        self.ts = 2
        self.sw = 0.2
        self.pad = 0.21
        self.minor_ticks = True
        self.grid = True
        self.ax_color = "k"
        self.facecolor = "w"
        self.text_color = "k"

    def get_axes(self, flat=False):

        plt.rcParams.update({"text.color": self.text_color})
        self.fig.patch.set_facecolor(self.facecolor)

        # GridSpec setup
        subplots = (self.subplot_1, self.subplot_2)
        self.subplots = subplots
        self.gs = mpl.gridspec.GridSpec(
            nrows=subplots[0],
            ncols=subplots[1],
            figure=self.fig,
            width_ratios=self.width_ratios or [1] * subplots[1],
            height_ratios=self.height_ratios or [1] * subplots[0],
            hspace=self.hspace,
            wspace=self.wspace,
        )

        self.axes = []
        for i in range(self.subplots[0]):
            row_axes = []
            for j in range(self.subplots[1]):
                ax = self.fig.add_subplot(self.gs[i, j])
                row_axes.append(ax)
                self.customize_axes(ax)

                if self.hspace == 0 and i != self.subplots[0] - 1:
                    ax.set_xticklabels([])

            self.axes.append(row_axes)

        if self.subplot_1 == 1 and self.subplot_2 == 1:
            return self.axes[0][0]

        self.axes_flat = [ax for row in self.axes for ax in row]

        if flat:
            return self.axes_flat
        else:
            return self.axes

    def customize_axes(
        self,
        ax,
        ylabel_pos="left",
        xlabel_pos="bottom",
    ):

        # if ylabel_pos == "left":
        #     labelright_bool = False
        #     labelleft_bool = True
        # elif ylabel_pos == "right":
        #     labelright_bool = True
        #     labelleft_bool = False

        # if xlabel_pos == "bottom":
        #     labeltop_bool = False
        #     labelbottom_bool = True
        # elif xlabel_pos == "top":
        #     labeltop_bool = True
        #     labelbottom_bool = False

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.ts * self.fs,
            size=self.fs * self.sw * 5,
            width=self.fs * self.sw * 0.9,
            pad=self.pad * self.fs,
            top=True,
            right=True,
            # labelbottom=labelbottom_bool,
            # labeltop=labeltop_bool,
            # labelright=labelright_bool,
            # labelleft=labelleft_bool,
            direction="inout",
            color=self.ax_color,
            labelcolor=self.ax_color,
        )

        if self.minor_ticks == True:
            ax.minorticks_on()

            ax.tick_params(
                axis="both",
                which="minor",
                direction="inout",
                top=True,
                right=True,
                size=self.fs * self.sw * 2.5,
                width=self.fs * self.sw * 0.8,
                color=self.ax_color,
            )

        ax.set_facecolor(self.facecolor)

        for spine in ax.spines.values():
            spine.set_linewidth(self.fs * self.sw)
            spine.set_color(self.ax_color)

        if self.grid:

            ax.grid(
                which="major",
                linewidth=self.fs * self.sw * 0.5,
                color=self.ax_color,
                alpha=0.25,
            )

    def save(self, path, bbox_inches=None, pad_inches=None):

        self.fig.savefig(
            path, dpi=self.dpi, bbox_inches=bbox_inches, pad_inches=pad_inches
        )

        self.path = path

    def check_saved_image(self):

        if not hasattr(self, "path"):
            raise ValueError("Figure has not been saved yet.")

        with Image.open(self.path) as img:
            print(img.size)
            return

    def show_image(self):

        if not hasattr(self, "path"):
            raise ValueError("Figure has not been saved yet.")

        with Image.open(self.path) as img:
            img.show()
            return


def frames_to_mp4(
    fold,
    title="video",
    fps=36,
    digit_format="04d",
    res=None,
    resize_factor=1,
    custom_bitrate=None,
    extension=".jpg",
):

    # Get a list of all .png files in the directory
    files = [f for f in os.listdir(fold) if f.endswith(extension)]
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if not files:
        raise ValueError("No PNG files found in the specified folder.")

    im = PIL.Image.open(os.path.join(fold, files[0]))
    resx, resy = im.size

    if res is not None:
        resx, resy = res
    else:
        resx = int(resize_factor * resx)
        resy = int(resize_factor * resy)
        resx += resx % 2  # Ensuring even dimensions
        resy += resy % 2

    basename = os.path.splitext(files[0])[0].split("_")[0]

    ffmpeg_path = "ffmpeg"
    abs_path = os.path.abspath(fold)
    parent_folder = os.path.dirname(abs_path) + os.sep
    output_file = os.path.join(parent_folder, f"{title}.mp4")

    crf = 2  # Lower for higher quality, higher for lower quality
    bitrate = custom_bitrate if custom_bitrate else "5000k"
    preset = "slow"
    tune = "film"

    command = f'{ffmpeg_path} -y -r {fps} -i {os.path.join(fold, f"{basename}_%{digit_format}{extension}")} -c:v libx264 -profile:v high -crf {crf} -preset {preset} -tune {tune} -b:v {bitrate} -pix_fmt yuv420p -vf scale={resx}:{resy} {output_file}'

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error during video conversion:", e)
