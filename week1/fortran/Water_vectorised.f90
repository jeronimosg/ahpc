! Applied High Performance Computing 2025
! 
! Molecular Dynamics Simulation of Water Molecules 
!
! Description: This program simulates flexible water molecules using a simple
!              classical model. Each water has two covalent bonds and one angle.
!              All non-bonded atoms interact through LJ potential. 
!              Verlet integrator is used. 
!
! Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
!
module water_mod
    implicit none
    
    ! Constants
    real(kind=8), parameter :: pi = 3.141592653589793d0 ! pi
    real(kind=8), parameter :: deg2rad = pi/180.0d0     ! pi/180 for changing degs to radians 
    real(kind=8) :: accumulated_forces_bond = 0.0d0     ! Checksum: accumulated size of forces
    real(kind=8) :: accumulated_forces_angle = 0.0d0    ! Checksum: accumulated size of forces
    real(kind=8) :: accumulated_forces_non_bond = 0.0d0 ! Checksum: accumulated size of forces
    integer, parameter :: nClosest = 8                  ! number of closest neighbors to consider in neighbor list.
        
    type :: Vec3 ! 3D vector type
        real(kind=8) :: x, y, z
    end type Vec3
    ! Operator interfaces for Vec3 type
    interface operator(+)
        module procedure vec3_add
    end interface
    interface operator(-)
        module procedure vec3_subtract
    end interface
    interface operator(*)
        module procedure vec3_multiply
        module procedure scalar_multiply_vec3
    end interface
    interface operator(/)
        module procedure vec3_divide
    end interface
    
    ! atom type
    type :: Atom
        real(kind=8) :: mass      ! The mass of the atom in (U)
        real(kind=8) :: ep        ! epsilon for LJ potential
        real(kind=8) :: sigma     ! Sigma, somehow the size of the atom
        real(kind=8) :: charge    ! charge of the atom (partial charge)
        character(len=10) :: name ! Name of the atom
        ! the position in (nm), velocity (nm/ps) and forces (k_BT/nm) of the atom
        type(Vec3) :: p = Vec3(0.0d0, 0.0d0, 0.0d0), &
                      v = Vec3(0.0d0, 0.0d0, 0.0d0), &
                      f = Vec3(0.0d0, 0.0d0, 0.0d0)
    end type Atom
    
    ! type for the covalent bond between two atoms U=0.5k(r12-L0)^2
    type :: Bond
        real(kind=8) :: K      ! force constant
        real(kind=8) :: L0     ! relaxed length
        integer :: a1, a2      ! the indexes of the atoms at either end
    end type Bond
    
    ! type for the angle between three atoms U=0.5K(phi123-phi0)^2
    type :: Angle
        real(kind=8) :: K
        real(kind=8) :: Phi0
        integer :: a1, a2, a3  ! the indexes of the three atoms, with a2 being the centre atom
    end type Angle
    
    ! molecule type
    type :: Molecule
        type(Atom), dimension(:), allocatable :: atoms  ! list of atoms in the molecule (Water has 3 atoms)
        type(Bond), dimension(:), allocatable :: bonds  ! the bond potentials, eg for water the left and right bonds
        type(Angle), dimension(:), allocatable :: angles  ! the angle potentials, for water just the single one, but keep it a list for generality
        integer, dimension(nClosest) :: neighbours  ! indices of the neighbours
        integer :: num_neighbours
    end type Molecule
    
    ! ===============================================================================
    ! Two new types arranging Atoms in a Structure-of-Array data structure
    ! ===============================================================================

    ! atoms type, representing N instances of identical atoms
    type :: Atoms
        real(kind=8) :: mass      ! The mass of the atom in (U)
        real(kind=8) :: ep        ! epsilon for LJ potential
        real(kind=8) :: sigma     ! Sigma, somehow the size of the atom
        real(kind=8) :: charge    ! charge of the atom (partial charge)
        character(len=10) :: name ! Name of the atom
        ! the position in (nm), velocity (nm/ps) and forces (k_BT/nm) of the atom
        type(Vec3), dimension(:), allocatable :: p, v, f
    end type Atoms

    ! molecule type (collection of atom-types)
    type :: Molecules
        type(Atoms), allocatable :: atoms(:)             ! list of atoms in the molecule
        type(Bond),  allocatable :: bonds(:)             ! the bond potentials, e.g. for water the left and right bonds
        type(Angle), allocatable :: angles(:)            ! the angle potentials, for water just the single one
        integer, allocatable :: neighbours(:,:)          ! indices of the neighbours (typically sized as (nClosest, no_mol))
        integer :: no_mol                                ! number of molecules in the type
    end type Molecules

    ! ===============================================================================

    ! system type
    type :: System
        type(Molecule), dimension(:), allocatable :: molecules  ! all the molecules in the system
        integer :: num_molecules
        real(kind=8) :: time                                    ! current simulation time
    end type System
    
    type :: Sim_Configuration
        integer :: steps                   ! number of steps
        integer :: no_mol                  ! number of molecules
        real(kind=8) :: dt                 ! integrator time step
        integer :: data_period             ! how often to save coordinate to trajectory
        character(len=100) :: filename     ! name of the output file with trajectory
    end type Sim_Configuration
    
contains
    real(kind=8) function vec3_mag2(a) result(mag2)  ! squared size of vector
        type(Vec3), intent(in) :: a
        mag2 = a%x*a%x + a%y*a%y + a%z*a%z
    end function vec3_mag2

    real(kind=8) function vec3_mag(a) result(mag)  ! size of vector
        type(Vec3), intent(in) :: a
        mag = sqrt(vec3_mag2(a))
    end function vec3_mag

    type(Vec3) function vec3_subtract(a, b) result(c) ! subtraction of two vectors
        type(Vec3), intent(in) :: a, b
        c%x = a%x - b%x; c%y = a%y - b%y; c%z = a%z - b%z
    end function vec3_subtract

    type(Vec3) function vec3_add(a, b) result(c) ! addition of two vectors
        type(Vec3), intent(in) :: a, b
        c%x = a%x + b%x; c%y = a%y + b%y; c%z = a%z + b%z
    end function vec3_add

    type(Vec3) function vec3_multiply(a, b) result(c) ! multiplication of vector by scalar (vector on left)
        type(Vec3), intent(in) :: a
        real(kind=8), intent(in) :: b
        c%x = a%x * b; c%y = a%y * b; c%z = a%z * b
    end function vec3_multiply

    type(Vec3) function scalar_multiply_vec3(a, b) result(c) ! multiplication of scalar by vector (scalar on left)
        real(kind=8), intent(in) :: a
        type(Vec3), intent(in) :: b
        c%x = a * b%x; c%y = a * b%y; c%z = a * b%z
    end function scalar_multiply_vec3

    type(Vec3) function vec3_divide(a, b) result(c) ! division of vector by scalar
        type(Vec3), intent(in) :: a
        real(kind=8), intent(in) :: b
        c%x = a%x / b; c%y = a%y / b; c%z = a%z / b
    end function vec3_divide

    type(Vec3) function vec3_cross(a, b) result(c)
        type(Vec3), intent(in) :: a, b
        c%x = a%y*b%z - a%z*b%y
        c%y = a%z*b%x - a%x*b%z
        c%z = a%x*b%y - a%y*b%x
    end function vec3_cross

    real(kind=8) function vec3_dot(a, b) result(dot)
        type(Vec3), intent(in) :: a, b
        dot = a%x*b%x + a%y*b%y + a%z*b%z
    end function vec3_dot

    function time_now() result(t)
        real(kind=8)    :: t
        integer(kind=8) :: count, count_rate
        call system_clock(count, count_rate)
        t = real(count, kind=8) / real(count_rate, kind=8)
    end function time_now

    ! simulation configurations: number of step, number of the molecules in the system, 
    ! IO frequency, time step and file name
    subroutine initialize_configuration(sc)
        type(Sim_Configuration), intent(out) :: sc
        integer :: argc, i, ios
        character(len=256) :: arg, val
        
        ! Defaults
        sc%steps = 10000
        sc%no_mol = 100
        sc%dt = 0.0005d0
        sc%data_period = 100
        sc%filename = 'trajectory.txt'

        ! Parse command line options
        argc = command_argument_count()
        do i = 1, argc, 2
            call get_command_argument(i, arg)
            arg = adjustl(arg)

            if (trim(arg) == '-h') then
                write(*,*) 'MD -steps <number of steps> -no_mol <number of molecules> '// &
                           '-fwrite <io frequency> -dt <size of timestep> -ofile <filename>'
                stop
            end if

            if (i+1 > argc) then
                write(*,*) '---> error: missing value for argument ', trim(arg)
                exit
            end if

            call get_command_argument(i+1, val)
            val = adjustl(val)

            select case (trim(arg))
            case ('-steps')
                read(val, *, iostat=ios) sc%steps
            case ('-no_mol')
                read(val, *, iostat=ios) sc%no_mol
            case ('-fwrite')
                read(val, *, iostat=ios) sc%data_period
            case ('-dt')
                read(val, *, iostat=ios) sc%dt
            case ('-ofile')
                sc%filename = trim(val)
            case default
                ios = -1; val = ''
            end select
            if (ios /= 0) then
                write(*,*) '---> error: argument type is not recognized or invalid value for '//trim(arg)//': '//trim(val)
                write(*,*) 'MD -steps <number of steps> -no_mol <number of molecules> '// &
                           '-fwrite <io frequency> -dt <size of timestep> -ofile <filename>'
                stop
            endif
        end do
        
        ! convert to ps based on having energy in k_BT, and length in nm
        sc%dt = sc%dt / 1.57350d0
    end subroutine initialize_configuration

    ! Update neighbour list for each atom, allowing us to quickly loop through all relevant non-bonded forces. Given the
    ! short timesteps, it takes many steps to go from being e.g. 20th closest to 2nd closest; only needs infrequent updating
    subroutine build_neighbor_list(sys)
        type(System), intent(inout) :: sys        
        real(kind=8) :: distances2(sys%num_molecules)
        type(Vec3) :: dp
        integer :: i, j, k, pos, target_num
        integer, dimension(:), allocatable :: indices
        real(kind=8), dimension(:), allocatable :: heap_dist

        !  We want at most nClosest neighbors, but no more than number of molecules.
        target_num = min(nClosest, sys%num_molecules - 1)
        allocate(indices(target_num),heap_dist(target_num))

        do i = 1, sys%num_molecules
            do j = 1, sys%num_molecules ! Calculate distances to all other molecules
                dp = sys%molecules(i)%atoms(1)%p - sys%molecules(j)%atoms(1)%p
                distances2(j) = vec3_mag2(dp)
            end do
            distances2(i) = 1.0d99 ! exclude own molecule from neighbour list

            heap_dist = 1.0d99 ! initialize heap to large values
            ! Find closest neighbors by scanning through distances keeping track of the
            ! sorted order. Keep the smallest distances in heap_dist and indices
            do j = 1, sys%num_molecules
                if (distances2(j) < heap_dist(1)) then
                    ! scan for position to insert
                    do k = 2, target_num
                        if (heap_dist(k) < distances2(j)) exit
                    enddo
                    pos = k - 1 ! insert at position pos since heap_dist(pos-1) > distances2(j) > heap_dist(pos)
                    do k = 1, pos-1 ! move all elements below pos down by one
                        indices(k) = indices(k+1)
                        heap_dist(k) = heap_dist(k+1)
                    end do
                    indices(pos) = j
                    heap_dist(pos) = distances2(j)
                end if
            end do

            ! Build neighbor list
            sys%molecules(i)%num_neighbours = 0
            do j = 1, target_num
                k = indices(j)  ! k: molecule nr of the jth closest molecule to molecule i
                if (k < i) then ! neighbour list of molecule k has already been created
                    ! Check if i is already in k's neighbor list othwerwise add it to neighbor list of molecule i
                    if (.not. any(sys%molecules(k)%neighbours(1:sys%molecules(k)%num_neighbours) == i)) then
                        sys%molecules(i)%num_neighbours = sys%molecules(i)%num_neighbours + 1
                        sys%molecules(i)%neighbours(sys%molecules(i)%num_neighbours) = k
                    end if
                else ! add molecule k to the neighbour list of molecule i
                    sys%molecules(i)%num_neighbours = sys%molecules(i)%num_neighbours + 1
                    sys%molecules(i)%neighbours(sys%molecules(i)%num_neighbours) = k
                end if
            end do
        end do
    end subroutine build_neighbor_list

    ! Given a bond, updates the force on all atoms correspondingly
    subroutine update_bond_forces(sys)
        type(System), intent(inout) :: sys
        integer :: i, j, num_bonds
        type(Vec3) :: dp, f
        real(kind=8) :: dp_mag
        
        num_bonds = size(sys%molecules(1)%bonds) ! assuming all molecules have same number of bonds
        do i = 1, sys%num_molecules
            do j = 1, num_bonds ! Loops over the (2 for water) bond constraints
                dp = sys%molecules(i)%atoms(sys%molecules(i)%bonds(j)%a1)%p - &
                     sys%molecules(i)%atoms(sys%molecules(i)%bonds(j)%a2)%p
                dp_mag = vec3_mag(dp)

                f = (-sys%molecules(i)%bonds(j)%K * (1.0d0 - sys%molecules(i)%bonds(j)%L0/dp_mag)) * dp

                sys%molecules(i)%atoms(sys%molecules(i)%bonds(j)%a1)%f = &
                    sys%molecules(i)%atoms(sys%molecules(i)%bonds(j)%a1)%f + f

                sys%molecules(i)%atoms(sys%molecules(i)%bonds(j)%a2)%f = &
                    sys%molecules(i)%atoms(sys%molecules(i)%bonds(j)%a2)%f - f
                
                accumulated_forces_bond = accumulated_forces_bond + vec3_mag(f)
            end do
        end do
    end subroutine update_bond_forces

    ! Iterates over all angles in molecules and updates forces on atoms correpondingly
    subroutine update_angle_forces(sys)
        type(System), intent(inout) :: sys
        integer :: i, j, a1, a2, a3, num_angles
        type(Vec3) :: d21, d23, c21_23, Ta, Tc, f1, f3
        real(kind=8) :: norm_d21, norm_d23, phi

        num_angles = size(sys%molecules(1)%angles) ! assuming all molecules have same number of angles
        do i = 1, sys%num_molecules
            do j = 1, num_angles
                !====  angle forces  (H--O---H bonds) U_angle = 0.5*k_a(phi-phi_0)^2
                ! f_H1 = K(phi-ph0)/|H1O|*Ta
                ! f_H2 = K(phi-ph0)/|H2O|*Tc
                ! f_O  = - (f_H1 + f_H2)
                ! Ta = norm(H1O x (H1O x H2O))
                ! Tc = norm(H2O x (H2O x H1O))
                !=============================================================
                a1 = sys%molecules(i)%angles(j)%a1
                a2 = sys%molecules(i)%angles(j)%a2
                a3 = sys%molecules(i)%angles(j)%a3

                d21 = sys%molecules(i)%atoms(a2)%p - sys%molecules(i)%atoms(a1)%p
                d23 = sys%molecules(i)%atoms(a2)%p - sys%molecules(i)%atoms(a3)%p

                ! phi = d21 dot d23 / |d21| |d23|
                norm_d21 = vec3_mag(d21)
                norm_d23 = vec3_mag(d23)
                phi = acos(vec3_dot(d21, d23) / (norm_d21 * norm_d23))
                
                ! d21 cross (d21 cross d23)
                c21_23 = vec3_cross(d21, d23)
                Ta = vec3_cross(d21, c21_23)
                Ta = Ta / vec3_mag(Ta)
                
                ! d23 cross (d23 cross d21) = - d23 cross (d21 cross d23) = c21_23 cross d23
                Tc = vec3_cross(c21_23, d23)
                Tc = Tc / vec3_mag(Tc)

                f1 = Ta * (sys%molecules(i)%angles(j)%K * &
                                (phi - sys%molecules(i)%angles(j)%Phi0) / norm_d21)
                f3 = Tc * (sys%molecules(i)%angles(j)%K * &
                                (phi - sys%molecules(i)%angles(j)%Phi0) / norm_d23)

                sys%molecules(i)%atoms(a1)%f = sys%molecules(i)%atoms(a1)%f + f1
                sys%molecules(i)%atoms(a2)%f = sys%molecules(i)%atoms(a2)%f - (f1 + f3)
                sys%molecules(i)%atoms(a3)%f = sys%molecules(i)%atoms(a3)%f + f3

                accumulated_forces_angle = accumulated_forces_angle + (vec3_mag(f1) + vec3_mag(f3))
            end do
        end do
    end subroutine update_angle_forces

    ! Iterates over atoms in different molecules and calculate non-bonded forces
    subroutine update_non_bonded_forces(sys)
        type(System), intent(inout) :: sys
        integer :: i, j, atom1, atom2, neighbor_idx, natoms
        real(kind=8) :: ep, sigma2, KC, q, r2, r, sir, sir3
        type(Vec3) :: dp, f
        ! nonbonded forces: only a force between atoms in different molecules
        ! The total non-bonded forces come from Lennard Jones (LJ) and coulomb interactions
        ! U = ep[(sigma/r)^12-(sigma/r)^6] + C*q1*q2/r
        
        natoms = size(sys%molecules(1)%atoms)
        do i = 1, sys%num_molecules
            do neighbor_idx = 1, sys%molecules(i)%num_neighbours ! iterate over all neighbours of molecule i
                j = sys%molecules(i)%neighbours(neighbor_idx)
                do atom1 = 1, natoms
                    do atom2 = 1, natoms
                        ep = sqrt(sys%molecules(i)%atoms(atom1)%ep * sys%molecules(j)%atoms(atom2)%ep)
                        sigma2 = (0.5d0 * (sys%molecules(i)%atoms(atom1)%sigma + &
                                        sys%molecules(j)%atoms(atom2)%sigma))**2
                        KC = 80.0d0 * 0.7d0  ! Coulomb prefactor
                        q = KC * sys%molecules(i)%atoms(atom1)%charge * sys%molecules(j)%atoms(atom2)%charge

                        dp = sys%molecules(i)%atoms(atom1)%p - sys%molecules(j)%atoms(atom2)%p
                        r2 = vec3_mag2(dp)
                        r = sqrt(r2)
                        
                        sir = sigma2 / r2
                        sir3 = sir * sir * sir
                        
                        f = (ep * (12.0d0*sir3*sir3 - 6.0d0*sir3) * sir + q/(r*r2)) * dp

                        sys%molecules(i)%atoms(atom1)%f = sys%molecules(i)%atoms(atom1)%f + f
                        sys%molecules(j)%atoms(atom2)%f = sys%molecules(j)%atoms(atom2)%f - f

                        accumulated_forces_non_bond = accumulated_forces_non_bond + vec3_mag(f)
                    end do
                end do
            end do
        end do
    end subroutine update_non_bonded_forces

    ! integrating the system for one time step using Leapfrog symplectic integration
    subroutine update_kdk(sys, sc)
        type(System), intent(inout) :: sys
        type(Sim_Configuration), intent(in) :: sc
        integer :: i, j, natoms
        type(Vec3), parameter :: zero_vec = Vec3(0.0d0, 0.0d0, 0.0d0)
        
        natoms = size(sys%molecules(1)%atoms)
        do i = 1, sys%num_molecules
            do j = 1, natoms
                sys%molecules(i)%atoms(j)%v = sys%molecules(i)%atoms(j)%v + & ! Update the velocities
                                              (sc%dt / sys%molecules(i)%atoms(j)%mass) * sys%molecules(i)%atoms(j)%f
                sys%molecules(i)%atoms(j)%f = zero_vec ! set the forces zero to prepare for next potential calculation
                sys%molecules(i)%atoms(j)%p = sys%molecules(i)%atoms(j)%p + & ! update position
                                              sc%dt * sys%molecules(i)%atoms(j)%v
            end do
        end do
        
        sys%time = sys%time + sc%dt ! update time
    end subroutine update_kdk

    subroutine make_water(sys, N_molecules)
        type(System), intent(out) :: sys
        integer, intent(in) :: N_molecules
        
        real(kind=8), parameter :: L0 = 0.09584d0
        real(kind=8), parameter :: water_angle = 104.45d0 * deg2rad
        real(kind=8) :: phi, radius, y, r, theta, x, z
        type(Vec3) :: P0, P1, P2
        integer :: i

        !===========================================================
        ! creating water molecules at position X0,Y0,Z0. 3 atoms
        !                        H---O---H
        ! The angle is 104.45 degrees and bond length is 0.09584 nm
        !===========================================================
        ! mass units of dalton
        ! initial velocity and force is set to zero for all the atoms by the constructor

        ! Allocate molecules
        allocate(sys%molecules(N_molecules))
        sys%num_molecules = N_molecules
        sys%time = 0.0d0
        
        ! initialize all water molecules on a sphere.
        phi = pi * (sqrt(5.0d0) - 1.0d0)
        radius = sqrt(real(N_molecules, 8)) * 0.15d0
        do i = 1, N_molecules
            y = 1.0d0 - (real(i-1, 8) / real(N_molecules - 1, 8))
            r = sqrt(1.0d0 - y * y)
            theta = phi * real(i-1, 8)
            
            x = cos(theta) * r
            z = sin(theta) * r

            P0 = Vec3(x * radius, y * radius, z * radius) ! Oxygen position
            P1 = P0 + Vec3(L0*sin(water_angle/2.0d0), L0*cos(water_angle/2.0d0), 0.0d0)  ! H1 position
            P2 = P0 + Vec3(-L0*sin(water_angle/2.0d0), L0*cos(water_angle/2.0d0), 0.0d0) ! H2 position

            ! Allocate atoms, bonds, angles
            allocate(sys%molecules(i)%atoms(3))  ! 3 atoms in water
            allocate(sys%molecules(i)%bonds(2))  ! 2 bonds in water
            allocate(sys%molecules(i)%angles(1)) ! 1 angle in water
            
            !                                mass    ep         sigma   charge   name position
            sys%molecules(i)%atoms(1) = Atom(16.0d0, 0.65d0,    0.31d0, -0.82d0, 'O', P0)   ! Oxygen atom (index 1)
            sys%molecules(i)%atoms(2) = Atom(1.0d0,  0.18828d0, 0.238d0, 0.41d0, 'H', P1) ! Hydrogen atom 1 (index 2)
            sys%molecules(i)%atoms(3) = Atom(1.0d0,  0.18828d0, 0.238d0, 0.41d0, 'H', P2) ! Hydrogen atom 2 (index 3)

            ! bonds beetween first H-O and second H-O respectively
            sys%molecules(i)%bonds(1) = Bond(20000.0d0, L0, 1, 2)
            sys%molecules(i)%bonds(2) = Bond(20000.0d0, L0, 1, 3)
            
            ! Angle between (H-O-H)
            sys%molecules(i)%angles(1) = Angle(1000.0d0, water_angle, 2, 1, 3)
            
            ! Initialize neighbor list
            sys%molecules(i)%num_neighbours = 0
        end do
    end subroutine make_water

    ! Write the system configurations in the trajectory file.
    subroutine write_output(sys, file_unit)
        type(System), intent(in) :: sys
        integer, intent(in) :: file_unit
        integer :: i, j, natoms
        
        natoms = size(sys%molecules(1)%atoms)
        do i = 1, sys%num_molecules
            do j = 1, natoms
                write(file_unit, '(F12.6,1X,A,3(1X,F12.6))') sys%time, &
                    trim(sys%molecules(i)%atoms(j)%name), &
                    sys%molecules(i)%atoms(j)%p%x, &
                    sys%molecules(i)%atoms(j)%p%y, &
                    sys%molecules(i)%atoms(j)%p%z
            end do
        end do
    end subroutine write_output

end module water_mod

!================================================================================================
!======================== Program ===============================================================
!================================================================================================
program water_molecular_dynamics
    use water_mod
    implicit none    
    type(System) :: sys
    type(Sim_Configuration) :: sc
    integer :: file_unit = 10
    integer :: step
    real(kind=8) :: tstart,tend

    call initialize_configuration(sc) ! Load the system configuration from command line data
    
    call make_water(sys, sc%no_mol) ! this will create a system containing sc.no_mol water molecules
    open(unit=file_unit, file=trim(sc%filename)) ! open file
    
    call write_output(sys, file_unit) ! writing the initial configuration in the trajectory file

    tstart = time_now() ! Start timing
    do step = 0, sc%steps - 1
        ! Build neighbor list every 100th step
        if (mod(step, 100) == 0) then
            call build_neighbor_list(sys)
        endif

        ! Always evolve the system
        call update_bond_forces(sys)
        call update_angle_forces(sys)
        call update_non_bonded_forces(sys)
        call update_kdk(sys, sc)

        ! Write output every data_period steps
        if (mod(step, sc%data_period) == 0) then
            call write_output(sys, file_unit)
        end if
    end do
    
    tend = time_now()  ! End time
    
    close(file_unit)  ! Close file
    
    write(*,'(A,ES12.4)') 'Accumulated forces Bonds   : ', accumulated_forces_bond
    write(*,'(A,ES12.4)') 'Accumulated forces Angles  : ', accumulated_forces_angle
    write(*,'(A,ES12.4)') 'Accumulated forces Non-bond: ', accumulated_forces_non_bond
    write(*,'(A,F9.4)')   'Elapsed total time.        : ', tend - tstart

end program water_molecular_dynamics