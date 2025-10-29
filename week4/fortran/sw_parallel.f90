!
! Applied High Performance Computing
! 
! Shallow Waters on GPUs
! 
! Assignment: Make a GPU parallelised shallow water code using OpenACC
!
! Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
! Date:   February 2022
! License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
!

module shallow_water_module
    implicit none
    
    ! Grid size can be set at compile time via -DNX=... -DNY=...
#ifndef NX
#define NX 512
#endif
#ifndef NY
#define NY 512
#endif
#ifndef PREC
#define PREC 4
#endif
    
    integer, parameter :: NX_PARAM = NX, NY_PARAM = NY  ! World Size
#if PREC == 4
    integer, parameter :: rk = kind(1.0e0_4)   ! Single precision
#elif PREC == 8
    integer, parameter :: rk = kind(1.0e0_8)   ! Double precision
#endif
    real(rk), parameter :: pi = 4.0_rk * atan(1.0_rk)
    
    type :: Sim_Configuration
        integer :: iter = 1000                      ! Number of iterations
        real(rk) :: dt = 0.05_rk                    ! Size of the integration time step
        real(rk) :: g = 9.80665_rk                  ! Gravitational acceleration
        real(rk) :: dx = 1.0_rk                     ! Integration step size in the horizontal direction
        real(rk) :: dy = 1.0_rk                     ! Integration step size in the vertical direction
        integer :: data_period = 100                ! how often to save coordinate to file
        character(len=256) :: filename = "sw_output.data"  ! name of the output file with history
    end type Sim_Configuration
    
    ! Representation of a water world including ghost lines, which is a "1-cell padding" of rows and columns
    ! around the world. These ghost lines is a technique to implement periodic boundary conditions.
    type :: Water
        real(rk), dimension(NY_PARAM, NX_PARAM) :: u  ! The speed in the horizontal direction.
        real(rk), dimension(NY_PARAM, NX_PARAM) :: v  ! The speed in the vertical direction.
        real(rk), dimension(NY_PARAM, NX_PARAM) :: e  ! The water elevation.
    end type Water
    
contains

    ! Parse command line arguments
    subroutine parse_arguments(config)
        type(Sim_Configuration), intent(inout) :: config
        integer :: i, num_args, ios
        character(len=256) :: arg, val
        
        num_args = command_argument_count()
        
        i = 1
        do while (i <= num_args)
            call get_command_argument(i, arg)
            
            if (trim(arg) == '-h') then
                write(*,*) './sw --iter <number of iterations> --dt <time step>', &
                          ' --g <gravitational const> --dx <x grid size> --dy <y grid size>', &
                          ' --fperiod <iterations between each save> --out <name of output file>'
                stop
            else if (i == num_args) then
                write(*,*) 'Error: The last argument (', trim(arg), ') must have a value'
                stop
            else
                call get_command_argument(i + 1, val)
                
                if (trim(arg) == '--iter') then
                    read(val, *, iostat=ios) config%iter
                    if (ios /= 0 .or. config%iter < 0) then
                        write(*,*) 'Error: iter must be a positive integer (e.g. --iter 1000)'
                        stop
                    end if
                else if (trim(arg) == '--dt') then
                    read(val, *, iostat=ios) config%dt
                    if (ios /= 0 .or. config%dt < 0) then
                        write(*,*) 'Error: dt must be a positive real number (e.g. --dt 0.05)'
                        stop
                    end if
                else if (trim(arg) == '--g') then
                    read(val, *, iostat=ios) config%g
                else if (trim(arg) == '--dx') then
                    read(val, *, iostat=ios) config%dx
                    if (ios /= 0 .or. config%dx < 0) then
                        write(*,*) 'Error: dx must be a positive real number (e.g. --dx 1)'
                        stop
                    end if
                else if (trim(arg) == '--dy') then
                    read(val, *, iostat=ios) config%dy
                    if (ios /= 0 .or. config%dy < 0) then
                        write(*,*) 'Error: dy must be a positive real number (e.g. --dy 1)'
                        stop
                    end if
                else if (trim(arg) == '--fperiod') then
                    read(val, *, iostat=ios) config%data_period
                    if (ios /= 0 .or. config%data_period < 0) then
                        write(*,*) 'Error: fperiod must be a positive integer (e.g. --fperiod 100)'
                        stop
                    end if
                else if (trim(arg) == '--out') then
                    config%filename = trim(val)
                else
                    write(*,*) '---> error: the argument type is not recognized'
                end if
                i = i + 2
            end if
        end do
    end subroutine parse_arguments
    
    ! Initialize water world
    subroutine initialize_water(w)
        type(Water), intent(inout) :: w
        integer :: i, j
        real(rk) :: ii, jj
        
        w%u = 0.0_rk
        w%v = 0.0_rk
        w%e = 0.0_rk
        
        do i = 2, NY_PARAM - 1
            do j = 2, NX_PARAM - 1
                ii = 100.0_rk * ((i - 1) - (NY_PARAM - 2.0_rk) / 2.0_rk) / NY_PARAM
                jj = 100.0_rk * ((j - 1) - (NX_PARAM - 2.0_rk) / 2.0_rk) / NX_PARAM
                w%e(i, j) = exp(-0.02_rk * (ii * ii + jj * jj))
            end do
        end do
    end subroutine initialize_water
    
    ! Write a history of the water heights to an ASCII file
    !
    ! @param water_history  Array of the all water worlds to write
    ! @param filename       The output filename of the ASCII file
    ! @param nsteps         Number of time steps saved
    subroutine to_file(water_history, filename, nsteps)
        real(rk), dimension(:,:,:), intent(in) :: water_history
        character(len=*), intent(in) :: filename
        integer, intent(in) :: nsteps
        integer :: unit_num, ios
        
        open(newunit=unit_num, file=trim(filename), form='unformatted', &
             access='stream', status='replace', iostat=ios)
        if (ios /= 0) then
            write(*,*) 'Error: Could not open file ', trim(filename)
            return
        end if
        
        write(unit_num) water_history(:,:,1:nsteps)
        close(unit_num)
    end subroutine to_file
    
    ! Exchange the horizontal ghost lines i.e. copy the second data row to the very last data row and vice versa.
    !
    ! @param data   The data update, which could be the water elevation `e` or the speed in the horizontal direction `u`.
    ! @param shape  The shape of data including the ghost lines.
    subroutine exchange_horizontal_ghost_lines(data)
        real(rk), dimension(NY_PARAM, NX_PARAM), intent(inout) :: data
        integer :: j
        
        do j = 1, NX_PARAM
            data(1, j) = data(NY_PARAM - 1, j)
            data(NY_PARAM, j) = data(2, j)
        end do
    end subroutine exchange_horizontal_ghost_lines
    
    ! Exchange the vertical ghost lines i.e. copy the second data column to the rightmost data column and vice versa.
    !
    ! @param data   The data update, which could be the water elevation `e` or the speed in the vertical direction `v`.
    ! @param shape  The shape of data including the ghost lines.
    subroutine exchange_vertical_ghost_lines(data)
        real(rk), dimension(NY_PARAM, NX_PARAM), intent(inout) :: data
        integer :: i
        
        do i = 1, NY_PARAM
            data(i, 1) = data(i, NX_PARAM - 1)
            data(i, NX_PARAM) = data(i, 2)
        end do
    end subroutine exchange_vertical_ghost_lines
    
    ! One integration step
    !
    ! @param w The water world to update.
    subroutine integrate(w, dt, dx, dy, g)
        type(Water), intent(inout) :: w
        real(rk), intent(in) :: dt, dx, dy, g
        integer :: i, j
        
        call exchange_horizontal_ghost_lines(w%e)
        call exchange_horizontal_ghost_lines(w%v)
        call exchange_vertical_ghost_lines(w%e)
        call exchange_vertical_ghost_lines(w%u)
        
        do i = 1, NY_PARAM - 1
            do j = 1, NX_PARAM - 1
                w%u(i, j) = w%u(i, j) - dt / dx * g * (w%e(i, j + 1) - w%e(i, j))
                w%v(i, j) = w%v(i, j) - dt / dy * g * (w%e(i + 1, j) - w%e(i, j))
            end do
        end do
        
        do i = 2, NY_PARAM - 1
            do j = 2, NX_PARAM - 1
                w%e(i, j) = w%e(i, j) - dt / dx * (w%u(i, j) - w%u(i, j - 1)) &
                                       - dt / dy * (w%v(i, j) - w%v(i - 1, j))
            end do
        end do
    end subroutine integrate
    
    ! Simulation of shallow water
    !
    ! @param num_of_iterations  The number of time steps to simulate
    ! @param size               The size of the water world excluding ghost lines
    ! @param output_filename    The filename of the written water world history (HDF5 file)
    subroutine simulate(config)
        type(Sim_Configuration), intent(in) :: config
        type(Water) :: water_world
        real(rk), allocatable :: water_history(:,:,:)
        integer :: t, num_saves, save_idx
        real(rk) :: checksum
        integer :: start_time, end_time, count_rate
        
        ! Calculate number of saves needed (add 1 for initial conditions)
        num_saves = config%iter / config%data_period + 2
        allocate(water_history(NY_PARAM, NX_PARAM, num_saves))
        
        call initialize_water(water_world)
        
        ! Save initial conditions before any integration
        save_idx = 1
        water_history(:,:,save_idx) = water_world%e
        save_idx = save_idx + 1
        
        call system_clock(start_time, count_rate)
        
        do t = 0, config%iter - 1
            call integrate(water_world, config%dt, config%dx, config%dy, config%g)
            if (mod(t, config%data_period) == 0) then
                water_history(:,:,save_idx) = water_world%e
                save_idx = save_idx + 1
            end if
        end do
        
        call system_clock(end_time)
        
        call to_file(water_history, config%filename, save_idx - 1)
        
        checksum = sum(water_world%e)
        write(*,'(A,F20.10)') 'checksum: ', checksum
        write(*,'(A,F10.6,A)') 'elapsed time: ', real(end_time - start_time) / real(count_rate), ' sec'
        
        deallocate(water_history)
    end subroutine simulate

end module shallow_water_module

! Main function that parses the command line and start the simulation
program main
    use shallow_water_module
    implicit none
    
    type(Sim_Configuration) :: config
    
    call parse_arguments(config)
    call simulate(config)
    
end program main
