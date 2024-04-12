# 2D Implementation of the Lax-Friederich Scheme

Python program for 2D implementation of Lax-Friederichs scheme for simulation of inviscid flow of a perfect gas. 

## Structure

 - `main/`
    - `main.py`: Main module for performing the simulation. 
    - `utils.py`: Utility functions for postprocessing simulation output and generating initial conditions. 
    - `plot_utils.py`: Utility functions for plotting and creating animations.
- `example/`
    - `gresho_vortex_example.py`: Example simulation of the Gresho Vortex. 
    - `waves_example.py`: Example simulation of propagating waves.
    - `analysis_vortex.ipynb`: Notebook that outputs figures related to the Gresho vortex. 
    - `analysis_wave.ipynb`: Notebook that outputs figures related to the propagating waves. 
- `example_videos/`: Two example videos for the Gresho vortex and propagating wave simulations

## Usage

The core functionality relies in the `Fluid2D()` class inside `main/main.py`, which implements the Lax-Friederichs scheme in 2D for the fluid flow equations.  The simulation output is saved in a `.hdf5` file format. There is also the possibility to add lagrangian tracers to the simulation. See files in `example/gresho_vortex_example.py` for example usage.  

## Dependencies
- [numpy](https://github.com/numpy/numpy): For math operations.
- [matplotlib](https://github.com/matplotlib/matplotlib): For plotting.
- [h5py](https://github.com/h5py/h5py): For saving simulation output data.
- [PIL](https://github.com/python-pillow/Pillow): For saving frames. 
- [ffmpeg](https://ffmpeg.org/): For rendering animations.


## License
This project is licensed under the [MIT License](LICENSE.md).