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
    - `analysis.ipynb`: Notebook that outputs the figuees used in the assingment. 
- `example_videos/`: Two example videos for the Gresho vortex and propagating wave simulations

## Dependencies
This project requires the following Python packages:
- [numpy](https://github.com/numpy/numpy): For math operations.
- [matplotlib](https://github.com/matplotlib/matplotlib): For plotting.
- [h5py](https://github.com/h5py/h5py): For saving simulation output data.
- [PIL](https://github.com/python-pillow/Pillow): For saving frames. 
- [ffmpeg](https://ffmpeg.org/): For rendering animations.


## License
This project is licensed under the [MIT License](LICENSE.md).