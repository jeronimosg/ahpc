#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>

// Applied High Performance Computing
//
// Molecular Dynamics Simulation of Water Molecules
//
// Description: This program simulates flexible water molecules using a simple
//              classical model. Each water has two covalent bonds and one
//              angle. All non-bonded atoms interact through LJ potential.
//              Verlet integrator is used.
//
// Authors: Troels Haugbølle, Rasmus Damgaard Nielsen, Weria Pezeshkian
//          Niels Bohr Institute, University of Copenhagen

const double deg2rad = acos(-1) / 180.0; // pi/180 for changing degs to radians
double accumulated_forces_bond = 0.;     // Checksum: accumulated size of forces
double accumulated_forces_angle = 0.;    // Checksum: accumulated size of forces
double accumulated_forces_non_bond = 0.; // Checksum: accumulated size of forces
constexpr size_t nClosest =
    8; // Number of closest neighbors to consider in neighbor list.
// constexpr just helps the compiler to optimize the code better

class Vec3 {
public:
  double x, y, z;
  Vec3(double x, double y, double z)
      : x(x), y(y), z(z) {} // initialization of vector
  double mag2() const {     // squared size of vector (slightly faster)
    return x * x + y * y + z * z;
  }
  double mag() const { // size of vector
    return sqrt(mag2());
  }
  Vec3 operator-(const Vec3 &other) const { // subtraction of two vectors
    return {x - other.x, y - other.y, z - other.z};
  }
  Vec3 operator+(const Vec3 &other) const { // addition of two vectors
    return {x + other.x, y + other.y, z + other.z};
  }
  Vec3 operator*(double scalar)
      const { // multiplication of vector by scalar (vec x scalar)
    return {scalar * x, scalar * y, scalar * z};
  }
  Vec3 operator/(double scalar) const { // division of vector by scalar
    return {x / scalar, y / scalar, z / scalar};
  }
  Vec3 &operator+=(const Vec3 &other) { // add and assign to vector
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
  Vec3 &operator-=(const Vec3 &other) { // subtract and assign to vector
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }
  Vec3 &operator*=(double scalar) { // multiply and assign to vector
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
  }
  Vec3 &operator/=(double scalar) { // divide and assign to vector
    x /= scalar;
    y /= scalar;
    z /= scalar;
    return *this;
  }
};
Vec3 operator*(
    double scalar,
    const Vec3 &y) { // multiplication of scalar by vector (scalar x vec)
  return y * scalar;
}
Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* atom class */
class Atom {
public:
  double mass;      // The mass of the atom in (U)
  double ep;        // epsilon for LJ potential
  double sigma;     // Sigma, somehow the size of the atom
  double charge;    // charge of the atom (partial charge)
  std::string name; // Name of the atom
  // the position in (nm), velocity (nm/ps) and forces (k_BT/nm) of the atom
  Vec3 p, v, f;
  // constructor, takes parameters and allocates p, v and f properly
  Atom(double mass, double ep, double sigma, double charge, std::string name)
      : mass{mass}, ep{ep}, sigma{sigma}, charge{charge}, name{name},
        p{0, 0, 0}, v{0, 0, 0}, f{0, 0, 0} {}
};

/* class for the covalent bond between two atoms U=0.5k(r12-L0)^2 */
class Bond {
public:
  double K;      // force constant
  double L0;     // relaxed length
  size_t a1, a2; // the indexes of the atoms at either end
};

/* class for the angle between three atoms U=0.5K(phi123-phi0)^2 */
class Angle {
public:
  double K;
  double Phi0;
  size_t a1, a2,
      a3; // the indexes of the three atoms, with a2 being the centre atom
};

/* molecule class */
class Molecule {
public:
  std::vector<Atom> atoms; // list of atoms in the molecule
  std::vector<Bond>
      bonds; // the bond potentials, eg for water the left and right bonds
  std::vector<Angle> angles; // the angle potentials, for water just the single
                             // one, but keep it a list for generality
  std::vector<size_t> neighbours; // indices of the neighbours
};

// ===============================================================================
// Two new classes arranging Atoms in a Structure-of-Array data structure
// ===============================================================================

/* atoms class, representing N instances of identical atoms */
class Atoms {
public:
  double mass;      // The mass of the atom in (U)
  double ep;        // epsilon for LJ potential
  double sigma;     // Sigma, somehow the size of the atom
  double charge;    // charge of the atom (partial charge)
  std::string name; // Name of the atom
  // the position in (nm), velocity (nm/ps) and forces (k_BT/nm) of the atom
  std::vector<Vec3> p, v, f;
  // constructor, takes parameters and allocates p, v and f properly
  Atoms(double mass, double ep, double sigma, double charge, std::string name,
        size_t N_identical)
      : mass{mass}, ep{ep}, sigma{sigma}, charge{charge}, name{name},
        p{N_identical, {0, 0, 0}}, v{N_identical, {0, 0, 0}},
        f{N_identical, {0, 0, 0}} {}
};

/* molecule class */
class Molecules {
public:
  std::vector<Atoms> atoms; // list of atoms in the molecule
  std::vector<Bond>
      bonds; // the bond potentials, eg for water the left and right bonds
  std::vector<Angle> angles; // the angle potentials, for water just the single
                             // one, but keep it a list for generality
  std::vector<std::vector<size_t>> neighbours; // indices of the neighbours
  size_t no_mol;

  // constructor, takes parameters and allocates neigbour list vector properly
  Molecules(std::vector<Atoms> atoms, std::vector<Bond> bonds,
            std::vector<Angle> angles, size_t no_mol)
      : atoms{atoms}, bonds{bonds}, angles{angles}, neighbours{no_mol},
        no_mol{no_mol} {}
};

// ===============================================================================

/* system class */
class System {
public:
  Molecules molecules; // all the molecules in the system
  double time = 0;     // current simulation time
};

class Sim_Configuration {
public:
  size_t steps = 10000;     // number of steps
  size_t no_mol = 100;      // number of molecules
  double dt = 0.0005;       // integrator time step
  size_t data_period = 100; // how often to save coordinate to trajectory
  std::string filename =
      "trajectory.txt"; // name of the output file with trajectory

  // simulation configurations: number of step, number of the molecules in the
  // system, IO frequency, time step and file name
  Sim_Configuration(std::vector<std::string> argument) {
    for (size_t i = 1; i < argument.size(); i += 2) {
      std::string arg = argument.at(i);
      if (arg == "-h") { // Write help
        std::cout << "MD -steps <number of steps> -no_mol <number of molecules>"
                  << " -fwrite <io frequency> -dt <size of timestep> -ofile "
                     "<filename> \n";
        exit(0);
        break;
      } else if (arg == "-steps") {
        steps = std::stoi(argument[i + 1]);
      } else if (arg == "-no_mol") {
        no_mol = std::stoi(argument[i + 1]);
      } else if (arg == "-fwrite") {
        data_period = std::stoi(argument[i + 1]);
      } else if (arg == "-dt") {
        dt = std::stof(argument[i + 1]);
      } else if (arg == "-ofile") {
        filename = argument[i + 1];
      } else {
        std::cout << "---> error: the argument type is not recognized \n";
      }
    }

    dt /= 1.57350; /// convert to ps based on having energy in k_BT, and length
                   /// in nm
  }
};

// Update neighbour list for each atom, allowing us to quickly loop through all
// relevant non-bonded forces. Given the short timesteps, it takes many steps to
// go from being e.g. 20th closest to 2nd closest; only needs infrequent
// updating
void BuildNeighborList(System &sys) {
  // array of distances to other molecules
  std::vector<double> distances2(sys.molecules.no_mol);
  // index array used for argsort
  std::vector<size_t> index(sys.molecules.no_mol);

  // For each molecule, build the neighbour list
  for (size_t mol_a_idx = 0; mol_a_idx < sys.molecules.no_mol; mol_a_idx++) {
    // empty neighbour list of molecule a
    sys.molecules.neighbours[mol_a_idx].clear();

    for (size_t mol_b_idx = 0; mol_b_idx < sys.molecules.no_mol; mol_b_idx++) {
      Vec3 dp = sys.molecules.atoms[0].p[mol_a_idx] -
                sys.molecules.atoms[0].p[mol_b_idx];
      distances2[mol_b_idx] = dp.mag2();
      index[mol_b_idx] = mol_b_idx;
    }

    distances2[mol_a_idx] = 1e99; // exclude own molecule from neighbour list

    // We want at most nClosest neighbors, but no more than number of molecules.
    size_t target_num = std::min(nClosest, sys.molecules.no_mol - 1);

    // Lambda function to compare distances with indices as the keys to sort
    auto lambda_compare = [&](size_t &a, size_t &b) {
      return distances2[a] < distances2[b];
    };

    // partial sort puts the lowest target_num elements at the start of the list
    // and ignore the rest
    std::partial_sort(index.begin(),              // Start of the list
                      index.begin() + target_num, // Point where to stop sorting
                      index.end(),                // End of the list
                      lambda_compare              // Compare function
    );

    // Test if index already exists in the neighbour list of other molecule and
    // if not insert it in neighbour list of molecule i
    for (size_t target_idx = 0; target_idx < target_num; target_idx++) {
      auto &k = index[target_idx]; // k: molecule nr of the jth closest molecule
                                   // to molecule i
      if (k <
          mol_a_idx) { // neighbour list of molecule k has already been created
        auto &neighbours = sys.molecules.neighbours[k];
        if (std::find(neighbours.begin(), neighbours.end(), mol_a_idx) ==
            neighbours
                .end()) // molecule i is not in neighbour list of molecule k
          sys.molecules.neighbours[mol_a_idx].push_back(
              k); // add molecule k to the neighbour list of molecule i
      } else {
        sys.molecules.neighbours[mol_a_idx].push_back(
            k); // add molecule k to the neighbour list of molecule i
      }
    }
  }
}

// Given a bond, updates the force on all atoms correspondingly
void UpdateBondForces(System &sys) {
  for (auto &bond : sys.molecules.bonds) {
    for (size_t mol_idx = 0; mol_idx < sys.molecules.no_mol; mol_idx++) {
      Vec3 dp = sys.molecules.atoms[bond.a1].p[mol_idx] -
                sys.molecules.atoms[bond.a2].p[mol_idx];
      double dp_mag = dp.mag();
      Vec3 f = -bond.K * (1 - bond.L0 / dp_mag) * dp;
      sys.molecules.atoms[bond.a1].f[mol_idx] += f;
      sys.molecules.atoms[bond.a2].f[mol_idx] -= f;
      accumulated_forces_bond += f.mag();
    }
  }
}

// Iterates over all angles in molecules and updates forces on atoms
// correpondingly
void UpdateAngleForces(System &sys) {

  for (auto &angle : sys.molecules.angles) {
    for (size_t mol_idx = 0; mol_idx < sys.molecules.no_mol; mol_idx++) {
      //====  angle forces  (H--O---H bonds) U_angle = 0.5*k_a(phi-phi_0)^2
      // f_H1 = K(phi-ph0)/|H1O|*Ta
      // f_H2 = K(phi-ph0)/|H2O|*Tc
      // f_O  = - (f_H1 + f_H2)
      // Ta = norm(H1O x (H1O x H2O))
      // Tc = norm(H2O x (H2O x H1O))
      //=============================================================
      Vec3 d21 = sys.molecules.atoms[angle.a2].p[mol_idx] -
                 sys.molecules.atoms[angle.a1].p[mol_idx];
      Vec3 d23 = sys.molecules.atoms[angle.a2].p[mol_idx] -
                 sys.molecules.atoms[angle.a3].p[mol_idx];

      // phi = d21 dot d23 / |d21| |d23|
      double norm_d21 = d21.mag();
      double norm_d23 = d23.mag();

      if (norm_d21 > 1e-10 && norm_d23 > 1e-10) { // Avoid division by zero
        double cos_phi = dot(d21, d23) / (norm_d21 * norm_d23);
        // Clamp cos_phi to avoid acos domain errors
        cos_phi = std::max(-1.0, std::min(1.0, cos_phi));
        double phi = acos(cos_phi);

        // d21 cross (d21 cross d23)
        Vec3 c21_23 = cross(d21, d23);
        Vec3 Ta = cross(d21, c21_23);
        double Ta_mag = Ta.mag();
        if (Ta_mag > 1e-10) {
          Ta /= Ta_mag;
        }

        // d23 cross (d23 cross d21) = - d23 cross (d21 cross d23) = c21_23
        // cross d23
        Vec3 Tc = cross(c21_23, d23);
        double Tc_mag = Tc.mag();
        if (Tc_mag > 1e-10) {
          Tc /= Tc_mag;
        }

        Vec3 f1 = Ta * (angle.K * (phi - angle.Phi0) / norm_d21);
        Vec3 f3 = Tc * (angle.K * (phi - angle.Phi0) / norm_d23);

        sys.molecules.atoms[angle.a1].f[mol_idx] += f1;
        sys.molecules.atoms[angle.a2].f[mol_idx] -= f1 + f3;
        sys.molecules.atoms[angle.a3].f[mol_idx] += f3;

        accumulated_forces_angle += f1.mag() + f3.mag();
      }
    }
  }
}

// Iterates over atoms in different molecules and calculate non-bonded forces
void UpdateNonBondedForces(System &sys) {
  /* nonbonded forces: only a force between atoms in different molecules
     The total non-bonded forces come from Lennard Jones (LJ) and coulomb
     interactions U = ep[(sigma/r)^12-(sigma/r)^6] + C*q1*q2/r */
  for (size_t mol_a_idx = 0; mol_a_idx < sys.molecules.no_mol; mol_a_idx++) {
    for (auto &mol_b_idx : sys.molecules.neighbours[mol_a_idx]) {
      for (auto &atom_type_a : sys.molecules.atoms) {
        for (auto &atom_type_b : sys.molecules.atoms) {
          double ep = sqrt(atom_type_a.ep * atom_type_b.ep);
          double sigma2 = pow(0.5 * (atom_type_a.sigma + atom_type_b.sigma),
                              2); // sigma = (sigma1+sigma2)/2
          double KC = 80 * 0.7;   // Coulomb prefactor
          double q = KC * atom_type_a.charge * atom_type_b.charge;

          Vec3 dp = atom_type_a.p[mol_a_idx] - atom_type_b.p[mol_b_idx];
          double r2 = dp.mag2();
          double r = sqrt(r2);

          double sir =
              sigma2 / r2; // crossection**2 times inverse squared distance
          double sir3 = sir * sir * sir;
          Vec3 f = (ep * (12 * sir3 * sir3 - 6 * sir3) * sir + q / (r * r2)) *
                   dp; // LJ + Coulomb forces

          atom_type_a.f[mol_a_idx] += f;
          atom_type_b.f[mol_b_idx] -=
              f; // update both pairs, since the force is equal and opposite and
          // pairs only exist in one neigbor list
          accumulated_forces_non_bond += f.mag();
        }
      }
    }
  }
}

// integrating the system for one time step using Leapfrog symplectic
// integration
void UpdateKDK(System &sys, Sim_Configuration &sc) {
  for (auto &atom : sys.molecules.atoms) {
    for (size_t mol_idx = 0; mol_idx < sys.molecules.no_mol; mol_idx++) {
      atom.v[mol_idx] +=
          sc.dt / atom.mass * atom.f[mol_idx]; // Update the velocities
      atom.f[mol_idx] = {0, 0, 0}; // set the forces zero to prepare for
                                   // next potential calculation
      atom.p[mol_idx] += sc.dt * atom.v[mol_idx]; // update position
    }
  }
  sys.time += sc.dt; // update time once per timestep
}

System MakeWater(size_t N_molecules) {
  //===========================================================
  // creating water molecules at position X0,Y0,Z0. 3 atoms
  //                        H---O---H
  // The angle is 104.45 degrees and bond length is 0.09584 nm
  //===========================================================
  // mass units of dalton
  // initial velocity and force is set to zero for all the atoms by the
  // constructor
  const double L0 = 0.09584;
  const double angle = 104.45 * deg2rad;

  //         mass    ep    sigma charge name
  Atoms Oatom(16, 0.65, 0.31, -0.82, "O", N_molecules);    // Oxygen atom
  Atoms Hatom1(1, 0.18828, 0.238, 0.41, "H", N_molecules); // Hydrogen atom
  Atoms Hatom2(1, 0.18828, 0.238, 0.41, "H", N_molecules); // Hydrogen atom

  // bonds beetween first H-O and second H-O respectively
  std::vector<Bond> waterbonds = {{.K = 20000, .L0 = L0, .a1 = 0, .a2 = 1},
                                  {.K = 20000, .L0 = L0, .a1 = 0, .a2 = 2}};

  // angle between H-O-H
  std::vector<Angle> waterangle = {
      {.K = 1000, .Phi0 = angle, .a1 = 1, .a2 = 0, .a3 = 2}};

  System sys{.molecules = Molecules(std::vector<Atoms>{Oatom, Hatom1, Hatom2},
                                    waterbonds, waterangle, N_molecules)};
  // initialize all water molecules on a sphere.
  double phi = acos(-1) * (sqrt(5.) - 1.);
  double radius = sqrt(N_molecules) * 0.15;
  for (size_t i = 0; i < N_molecules; i++) {
    double y = 1 - (i / (N_molecules - 1.));
    double r = sqrt(1 - y * y);
    double theta = phi * i;

    double x = cos(theta) * r;
    double z = sin(theta) * r;

    Vec3 P0{x * radius, y * radius, z * radius};
    sys.molecules.atoms[0].p[i] = {P0.x, P0.y, P0.z};
    sys.molecules.atoms[1].p[i] = {P0.x + L0 * sin(angle / 2),
                                   P0.y + L0 * cos(angle / 2), P0.z};
    sys.molecules.atoms[2].p[i] = {P0.x - L0 * sin(angle / 2),
                                   P0.y + L0 * cos(angle / 2), P0.z};
  }

  return sys;
}

// Write the system configurations in the trajectory file.
void WriteOutput(System &sys, std::ofstream &file) {
  // Loop over all atoms in model one molecule at a time and write out
  // position
  for (auto &atom : sys.molecules.atoms) {
    for (size_t mol_idx = 0; mol_idx < sys.molecules.no_mol; mol_idx++) {
      file << sys.time << " " << atom.name << " " << atom.p[mol_idx].x << " "
           << atom.p[mol_idx].y << " " << atom.p[mol_idx].z << '\n';
    }
  }
}

//======================================================================================================
//======================== Main function
//===============================================================
//======================================================================================================
int main(int argc, char *argv[]) {
  Sim_Configuration sc(
      {argv,
       argv + argc}); // Load the system configuration from command line data

  System sys = MakeWater(sc.no_mol); // this will create a system containing
                                     // sc.no_mol water molecules
  std::ofstream file(sc.filename);   // open file

  WriteOutput(sys,
              file); // writing the initial configuration in the trajectory file

  auto tstart = std::chrono::high_resolution_clock::now(); // start time
                                                           // (nano-seconds)

  for (size_t step = 0; step < sc.steps; step++) {
    // BuildNeighborList every 100th step
    if (step % 100 == 0)
      BuildNeighborList(sys);

    // Always evolve the system
    UpdateBondForces(sys);
    UpdateAngleForces(sys);
    UpdateNonBondedForces(sys);
    UpdateKDK(sys, sc);

    // Write output every data_period steps
    if (step % sc.data_period == 0) {
      WriteOutput(sys, file);
    }
  }

  auto tend =
      std::chrono::high_resolution_clock::now(); // end time (nano-seconds)

  std::cout << "Accumulated forces Bonds   : " << std::setw(9)
            << std::setprecision(5) << accumulated_forces_bond << "\n";
  std::cout << "Accumulated forces Angles  : " << std::setw(9)
            << std::setprecision(5) << accumulated_forces_angle << "\n";
  std::cout << "Accumulated forces Non-bond: " << std::setw(9)
            << std::setprecision(5) << accumulated_forces_non_bond << "\n";
  std::cout << "Elapsed total time:       " << std::fixed << std::setw(9)
            << std::setprecision(4) << (tend - tstart).count() * 1e-9 << "\n";
}
