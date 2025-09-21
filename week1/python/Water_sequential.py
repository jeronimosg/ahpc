# Applied High Performance Computing 
# 
# Molecular Dynamics Simulation of Water Molecules 
# 
# Description: This program simulates flexible water molecules using a simple
#              classical model. Each water has two covalent bonds and one angle.
#              All non-bonded atoms interact through LJ potential. 
#              Verlet integrator is used. 
#
# Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen

import sys
import time
import numpy as np
from typing import List

deg2rad = np.pi / 180.0 # pi/180 for changing degs to radians

accumulated_forces_bond: float = 0.0 # Checksum: accumulated size of forces
accumulated_forces_angle: float = 0.0 # Checksum: accumulated size of forces
accumulated_forces_non_bond: float = 0.0 # Checksum: accumulated size of forces

# Number of closest neighbors to consider in neighbor list. 
N_CLOSEST: int = 8

# Create a vector of 3 doubles
def Vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    return np.array([x,y,z])

def mag(v: np.ndarray) -> float: # magnitude of vector
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

def mag2(v: np.ndarray) -> float: # squared magnitude of vector
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]

def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray: # cross product of two vectors
    return Vec3(a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

def dot(a: np.ndarray, b: np.ndarray) -> float: # dot product of two vectors
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

# atom class
class Atom:
    def __init__(self, mass: float, ep: float, sigma: float, charge: float, name: str):
        self.mass = mass  # The mass of the atom in (U)
        self.ep = ep  # epsilon for LJ potential
        self.sigma = sigma  # Sigma, somehow the size of the atom
        self.charge = charge  # charge of the atom (partial charge)
        self.name = name  # Name of the atom
        # the position in (nm), velocity (nm/ps) and forces (k_BT/nm) of the atom
        self.p = Vec3()
        self.v = Vec3()
        self.f = Vec3()

# class for the covalent bond between two atoms U=0.5k(r12-L0)^2
class Bond:
    def __init__(self, K: float, L0: float, a1: int, a2: int):
        self.K = K  # force constant
        self.L0 = L0  # relaxed length
        self.a1 = a1  # the indexes of the atoms at either end
        self.a2 = a2

# class for the angle between three atoms U=0.5K(phi123-phi0)^2
class Angle:
    def __init__(self, K: float, Phi0: float, a1: int, a2: int, a3: int):
        self.K = K
        self.Phi0 = Phi0
        self.a1 = a1  # the indexes of the three atoms, with a2 being the centre atom
        self.a2 = a2
        self.a3 = a3

# molecule class
class Molecule:
    def __init__(self, atoms: List[Atom], bonds: List[Bond], angles: List[Angle]):
        self.atoms = atoms  # list of atoms in the molecule
        self.bonds = bonds  # the bond potentials, eg for water the left and right bonds
        self.angles = angles  # the angle potentials, for water just the single one, but keep it a list for generality
        self.neighbours: List[int] = []  # indices of the neighbours

# system class
class System:
    def __init__(self):
        self.molecules: List[Molecule] = []  # all the molecules in the system
        self.time: float = 0.0  # current simulation time

class Sim_Configuration:
    def __init__(self, argument: List[str]):
        self.steps: int = 20     # number of steps
        self.no_mol: int = 100   # number of molecules
        self.dt: float = 0.0005  # integrator time step
        self.data_period: int = 100  # how often to save coordinate to trajectory
        self.filename: str = "trajectory.txt"  # name of the output file with trajectory

        # simulation configurations: number of step, number of the molecules in the system, 
        # IO frequency, time step and file name
        i = 1
        n = len(argument)
        while i < n:
            arg = argument[i]
            if arg == "-h":  # Write help
                print(
                    "MD -steps <number of steps> -no_mol <number of molecules> -fwrite <io frequency> -dt <size of timestep> -ofile <filename>"
                )
                sys.exit(0)
            if i + 1 >= n:
                print("---> error: missing value for argument", arg)
                break
            if arg == "-steps":
                self.steps = int(argument[i + 1])
            elif arg == "-no_mol":
                self.no_mol = int(argument[i + 1])
            elif arg == "-fwrite":
                self.data_period = int(argument[i + 1])
            elif arg == "-dt":
                self.dt = float(argument[i + 1])
            elif arg == "-ofile":
                self.filename = argument[i + 1]
            else:
                print("---> error: the argument type is not recognized")
            i += 2

        self.dt /= 1.57350  # convert to ps based on having energy in k_BT, and length in nm

# Update neighbour list for each atom, allowing us to quickly loop through all relevant non-bonded forces. Given the
# short timesteps, it takes many steps to go from being e.g. 20th closest to 2nd closest; only needs infrequent updating
def BuildNeighborList(sysobj: System):

    # We want at most N_CLOSEST neighbors, but no more than number of molecules.
    target_num = min(N_CLOSEST, max(0, len(sysobj.molecules) - 1))

    distances2 = np.empty(len(sysobj.molecules)) # array of distances to other molecules

    for i in range(len(sysobj.molecules)):     # For each molecule, build the neighbour list
        sysobj.molecules[i].neighbours.clear() # empty neighbour list of molecule i
        for j in range(len(sysobj.molecules)):
            dp = sysobj.molecules[i].atoms[0].p - sysobj.molecules[j].atoms[0].p
            distances2[j] = mag2(dp)
        distances2[i] = 1e99 # exclude own molecule from neighbour list

        # argpartition partition so that target_num indices gives the smallest distances. Ignore the rest
        index = np.argpartition(distances2, target_num)[:target_num]

        # Test if index already exists in the neighbour list of other molecule and if not insert it in neighbour list of molecule i
        for k in index:  # k: molecule nr of the jth close molecule to molecule i
            if k < i:  # neighbour list of molecule k has already been created
                if i not in sysobj.molecules[k].neighbours:  # molecule i is not in neighbour list of molecule k
                    sysobj.molecules[i].neighbours.append(k)  # add molecule k to the neighbour list of molecule i
            else:
                sysobj.molecules[i].neighbours.append(k)  # add molecule k to the neighbour list of molecule i

# Given a bond, updates the force on all atoms correspondingly
def UpdateBondForces(sysobj: System):
    global accumulated_forces_bond
    for molecule in sysobj.molecules:
        # Loops over the (2 for water) bond constraints
        for bond in molecule.bonds:
            atom1 = molecule.atoms[bond.a1]
            atom2 = molecule.atoms[bond.a2]
            dp = atom1.p - atom2.p
            r = mag(dp)
            # f   = -bond.K*(1-bond.L0/mag(dp))*dp;
            f = (-bond.K * (1.0 - bond.L0 / r)) * dp
            atom1.f += f
            atom2.f -= f
            accumulated_forces_bond += mag(f)

# Iterates over all angles in molecules and updates forces on atoms correpondingly
def UpdateAngleForces(sysobj: System):
    global accumulated_forces_angle
    for molecule in sysobj.molecules:
        for angle in molecule.angles:
            #====  angle forces  (H--O---H bonds) U_angle = 0.5*k_a(phi-phi_0)^2
            # f_H1 = K(phi-ph0)/|H1O|*Ta
            # f_H2 = K(phi-ph0)/|H2O|*Tc
            # f_O  = - (f_H1 + f_H2)
            # Ta = norm(H1O x (H1O x H2O))
            # Tc = norm(H2O x (H2O x H1O))
            #=============================================================
            atom1 = molecule.atoms[angle.a1]
            atom2 = molecule.atoms[angle.a2]
            atom3 = molecule.atoms[angle.a3]

            d21 = atom2.p - atom1.p
            d23 = atom2.p - atom3.p

            # phi = d21 dot d23 / |d21| |d23|
            norm_d21 = mag(d21)
            norm_d23 = mag(d23)
            phi = np.arccos(dot(d21, d23) / (norm_d21 * norm_d23))

            # d21 cross (d21 cross d23)
            c21_23 = cross(d21, d23)
            Ta = cross(d21, c21_23)
            Ta_mag = mag(Ta)
            Ta = Ta / Ta_mag

            # d23 cross (d23 cross d21) = - d23 cross (d21 cross d23) = c21_23 cross d23
            Tc = cross(c21_23, d23)
            Tc_mag = mag(Tc)
            Tc = Tc / Tc_mag

            f1 = Ta * (angle.K * (phi - angle.Phi0) / norm_d21)
            f3 = Tc * (angle.K * (phi - angle.Phi0) / norm_d23)

            atom1.f += f1
            atom2.f -= (f1 + f3)
            atom3.f += f3

            accumulated_forces_angle += mag(f1) + mag(f3)

# Iterates over atoms in different molecules and calculate non-bonded forces
def UpdateNonBondedForces(sysobj: System):
    """
    nonbonded forces: only a force between atoms in different molecules
    The total non-bonded forces come from Lennard Jones (LJ) and coulomb interactions
    U = ep[(sigma/r)^12-(sigma/r)^6] + C*q1*q2/r
    """
    global accumulated_forces_non_bond
    acc = 0.0
    for i in range(len(sysobj.molecules)):
        for j in sysobj.molecules[i].neighbours:  # iterate over all neighbours of molecule i
            for atom1 in sysobj.molecules[i].atoms:
                for atom2 in sysobj.molecules[j].atoms:  # iterate over all pairs of atoms, similar as well as dissimilar
                    ep = np.sqrt(atom1.ep * atom2.ep)  # ep = sqrt(ep1*ep2)
                    sigma2 = (0.5 * (atom1.sigma + atom2.sigma)) ** 2  # sigma = (sigma1+sigma2)/2
                    KC = 80 * 0.7  # Coulomb prefactor
                    q = KC * atom1.charge * atom2.charge

                    dp = atom1.p - atom2.p
                    r2 = mag2(dp)
                    r = np.sqrt(r2)

                    sir = sigma2 / r2  # crossection**2 times inverse squared distance
                    sir3 = sir * sir * sir
                    # LJ + Coulomb forces
                    f = (ep * (12 * sir3 * sir3 - 6 * sir3) * sir + q / (r * r2)) * dp

                    atom1.f += f
                    # update both pairs, since the force is equal and opposite and pairs only exist in one neigbor list
                    atom2.f -= f
                    acc += mag(f)

    accumulated_forces_non_bond += acc


# integrating the system for one time step using Leapfrog symplectic integration
def UpdateKDK(sysobj: System, sc: Sim_Configuration):
    for molecule in sysobj.molecules:
        for atom in molecule.atoms:
            atom.v += (sc.dt / atom.mass) * atom.f  # Update the velocities
            atom.f = Vec3()                         # set the forces zero to prepare for next potential calculation
            atom.p += sc.dt * atom.v                # update position

    sysobj.time += sc.dt  # update time


# Setup one water molecule
def MakeWater(N_molecules: int) -> System:
    #===========================================================
    # creating water molecules at position X0,Y0,Z0. 3 atoms
    #                        H---O---H
    # The angle is 104.45 degrees and bond length is 0.09584 nm
    #===========================================================
    # mass units of dalton
    # initial velocity and force is set to zero for all the atoms by the constructor
    L0 = 0.09584
    angle = 104.45 * deg2rad

    # bonds beetween first H-O and second H-O respectively
    waterbonds = [
        Bond(K=20000, L0=L0, a1=0, a2=1),
        Bond(K=20000, L0=L0, a1=0, a2=2),
    ]

    # angle between H-O-H
    waterangle = [
        Angle(K=1000, Phi0=angle, a1=1, a2=0, a3=2),
    ]

    sysobj = System()
    # initialize all water molecules on a sphere.
    phi = np.arccos(-1) * (np.sqrt(5.0) - 1.0)
    radius = np.sqrt(N_molecules) * 0.15

    for i in range(N_molecules):
        y = 1.0 - (i / (N_molecules - 1.0)) if N_molecules > 1 else 0.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        #          mass  ep   sigma  charge name
        Oatom = Atom(16, 0.65, 0.31, -0.82, "O")       # Oxygen atom
        Hatom1 = Atom(1, 0.18828, 0.238, 0.41, "H")    # Hydrogen atom
        Hatom2 = Atom(1, 0.18828, 0.238, 0.41, "H")    # Hydrogen atom

        P0 = Vec3(x * radius, y * radius, z * radius)
        Oatom.p = Vec3(P0[0], P0[1], P0[2])
        Hatom1.p = Vec3(P0[0] + L0 * np.sin(angle / 2), P0[1] + L0 * np.cos(angle / 2), P0[2])
        Hatom2.p = Vec3(P0[0] - L0 * np.sin(angle / 2), P0[1] + L0 * np.cos(angle / 2), P0[2])
        atoms = [Oatom, Hatom1, Hatom2]
        sysobj.molecules.append(Molecule(atoms, waterbonds, waterangle))

    return sysobj

# Write the system configurations in the trajectory file.
def WriteOutput(sysobj: System, file):
    # Loop over all atoms in model one molecule at a time and write out position
    for molecule in sysobj.molecules:
        for atom in molecule.atoms:
            # Match C++ iostream default formatting (~6 significant digits, general format)
            file.write(
                f"{sysobj.time:.6g} {atom.name} {atom.p[0]:.6g} {atom.p[1]:.6g} {atom.p[2]:.6g}\n"
            )

#======================================================================================================
#======================== Main function ===============================================================
#======================================================================================================
sc = Sim_Configuration(sys.argv) # Load the system configuration from command line data

sysobj = MakeWater(sc.no_mol) # this will create a system containing sc.no_mol water molecules

file = open(sc.filename, "w") # open file

WriteOutput(sysobj, file) # writing the initial configuration in the trajectory file

tstart = time.perf_counter() # start time (seconds)

for step in range(sc.steps):
    # BuildNeighborList every 100th step
    if step % 100 == 0:
        BuildNeighborList(sysobj)

    # Always evolve the system
    UpdateBondForces(sysobj)
    UpdateAngleForces(sysobj)
    UpdateNonBondedForces(sysobj)
    UpdateKDK(sysobj, sc)

    # Write output every data_period steps
    if step % sc.data_period == 0:
        WriteOutput(sysobj, file)
    
file.close()

tend = time.perf_counter() # end time (seconds)

# Print summaries
print(f"Accumulated forces Bonds   : {accumulated_forces_bond:9.5f}")
print(f"Accumulated forces Angles  : {accumulated_forces_angle:9.5f}")
print(f"Accumulated forces Non-bond: {accumulated_forces_non_bond:9.5f}")
print(f"Elapsed total time:       {tend - tstart:9.4f}")