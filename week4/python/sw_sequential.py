#!/usr/bin/env python3
"""
Applied High Performance Computing

Shallow Waters on GPUs

Assignment: Make a GPU parallelised shallow water code using Cupy

Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
Date:   October 2025
License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
"""
import numpy as np
import sys
import time

# Grid size can be set at module level
NX = 512
NY = 512

# Precision control
PREC = 8  # 4 for float32, 8 for float64

if PREC == 4:
    real_t = np.float32
elif PREC == 8:
    real_t = np.float64
else:
    real_t = np.float32

class Sim_Configuration:
    """ Configuration class for simulation parameters """
    def __init__(self, arguments):
        self.iter = 1000           # Number of iterations
        self.dt = 0.05             # Size of the integration time step
        self.g = 9.80665           # Gravitational acceleration
        self.dx = 1.0              # Integration step size in the horizontal direction
        self.dy = 1.0              # Integration step size in the vertical direction
        self.data_period = 100     # how often to save coordinate to file
        self.filename = "sw_output.data"  # name of the output file with history
        
        i = 1
        while i < len(arguments):
            arg = arguments[i]
            if arg == "-h":  # Write help
                print("./sw --iter <number of iterations> --dt <time step>",
                      "--g <gravitational const> --dx <x grid size> --dy <y grid size>",
                      "--fperiod <iterations between each save> --out <name of output file>")
                sys.exit(0)
            elif i == len(arguments) - 1:
                raise ValueError(f"The last argument ({arg}) must have a value")
            else:
                if arg == "--iter":
                    self.iter = int(arguments[i + 1])
                    if self.iter < 0:
                        raise ValueError("iter must be a positive integer (e.g. --iter 1000)")
                elif arg == "--dt":
                    self.dt = float(arguments[i + 1])
                    if self.dt < 0:
                        raise ValueError("dt must be a positive real number (e.g. --dt 0.05)")
                elif arg == "--g":
                    self.g = float(arguments[i + 1])
                elif arg == "--dx":
                    self.dx = float(arguments[i + 1])
                    if self.dx < 0:
                        raise ValueError("dx must be a positive real number (e.g. --dx 1)")
                elif arg == "--dy":
                    self.dy = float(arguments[i + 1])
                    if self.dy < 0:
                        raise ValueError("dy must be a positive real number (e.g. --dy 1)")
                elif arg == "--fperiod":
                    self.data_period = int(arguments[i + 1])
                    if self.data_period < 0:
                        raise ValueError("fperiod must be a positive integer (e.g. --fperiod 100)")
                elif arg == "--out":
                    self.filename = arguments[i + 1]
                else:
                    print("---> error: the argument type is not recognized")                
                i += 2

class Water:
    """ Representation of a water world including ghost lines, which is a "1-cell padding" of rows and columns
    around the world. These ghost lines is a technique to implement periodic boundary conditions. """
    def __init__(self):
        self.u = np.zeros((NY, NX), dtype=real_t) # The speed in the horizontal direction.
        self.v = np.zeros((NY, NX), dtype=real_t) # The speed in the vertical direction.
        self.e = np.zeros((NY, NX), dtype=real_t) # The water elevation.

        ii = 100.0 * (np.arange(1, NY - 1, dtype=real_t) - (NY - 2.0) / 2.0) / NY # The vertical coordinate
        jj = 100.0 * (np.arange(1, NX - 1, dtype=real_t) - (NX - 2.0) / 2.0) / NX # The horizontal coordinate
        II, JJ = np.meshgrid(ii, jj, indexing="ij")                               # Meshgrid for 2D coordinates to enable vectorized computation
        self.e[1:NY - 1, 1:NX - 1] = np.exp(-0.02 * (II * II + JJ * JJ))

def to_file(water_history, filename):
    """ Write a history of the water heights to a binary file. Arguments:
        water_history: List of water elevation grids to write
        filename: The output filename of the binary file """
    # Convert list to numpy array and write as binary
    history_array = np.array(water_history, dtype=real_t)
    with open(filename, 'wb') as f:
        history_array.tofile(f)

def exchange_horizontal_ghost_lines(data):
    """ Exchange the horizontal ghost lines i.e. copy the second data row to the very last data row and vice versa.
        data: The data update, which could be the water elevation `e` or the speed in the horizontal direction `u`. """
    data[0, :] = data[NY - 2, :]
    data[NY - 1, :] = data[1, :]

def exchange_vertical_ghost_lines(data):
    """Exchange the vertical ghost lines i.e. copy the second data column to the rightmost data column and vice versa.
       data: The data update, which could be the water elevation `e` or the speed in the vertical direction `v`.
    """
    data[:, 0] = data[:, NX - 2]
    data[:, NX - 1] = data[:, 1]

def integrate(w, dt, dx, dy, g):
    """ One integration step
        w: The water world to update.
        dt: Time step
        dx, dy: Grid spacing in x and y direction
        g: Gravitational acceleration """
    exchange_horizontal_ghost_lines(w.e)
    exchange_horizontal_ghost_lines(w.v)
    exchange_vertical_ghost_lines(w.e)
    exchange_vertical_ghost_lines(w.u)
    
    # Update velocities (vectorized)
    w.u[0:NY-1, 0:NX-1] -= dt / dx * g * (w.e[0:NY-1, 1:NX] - w.e[0:NY-1, 0:NX-1])
    w.v[0:NY-1, 0:NX-1] -= dt / dy * g * (w.e[1:NY, 0:NX-1] - w.e[0:NY-1, 0:NX-1])
    
    # Update elevations (vectorized)
    w.e[1:NY-1, 1:NX-1] -= dt / dx * (w.u[1:NY-1, 1:NX-1] - w.u[1:NY-1, 0:NX-2]) + \
                           dt / dy * (w.v[1:NY-1, 1:NX-1] - w.v[0:NY-2, 1:NX-1])

def simulate(config):
    """ Simulation of shallow water
        config: Sim_Configuration object with simulation parameters """
    water_world = Water()    
    # Save initial conditions before any integration, and store history in a list
    water_history = [water_world.e.copy()]
    
    begin = time.perf_counter()
    
    for t in range(config.iter):
        integrate(water_world, config.dt, config.dx, config.dy, config.g)
        if t % config.data_period == 0:
            water_history.append(water_world.e.copy())
    
    end = time.perf_counter()
    
    to_file(water_history, config.filename)
    
    checksum = np.sum(water_world.e)
    print(f"checksum: {checksum}")
    print(f"elapsed time: {end - begin} sec")

if __name__ == "__main__":
    """ parse the command line and start the simulation """
    config = Sim_Configuration(sys.argv)
    simulate(config)