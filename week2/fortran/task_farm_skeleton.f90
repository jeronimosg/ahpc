!
! Applied High Performance Computing
! 
! Task Farming with MPI
! 
! Assignment: Make an MPI task farm. A "task" is a randomly generated integer.
! To "execute" a task, the worker sleeps for the given number of milliseconds.
! The result of a task should be send back from the worker to the master. It
! contains the rank of the worker
!
! Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
! Date: September 2025 [Derived from C++ version from 2022]
! License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
!
module task_farm_mod
    use mpi
    implicit none
    integer, parameter :: NTASKS = 5000
    integer, parameter :: GLOBAL_SEED = 1234
contains
    function time_now() result(t)
        real(kind=8)    :: t
        integer(kind=8) :: count, count_rate
        call system_clock(count, count_rate)
        t = real(count, kind=8) / real(count_rate, kind=8)
    end function time_now

    subroutine master(nworker)
        integer, intent(in) :: nworker
        
        integer :: task(NTASKS), result(NTASKS)
        integer :: i, worker, tasksdone, itask
        real(kind=8) :: tstart, tend, rand_val, std_dev
        integer :: seed_size
        integer, allocatable, dimension(:) :: seed, workdone

        ! set up a random number generator
        call random_seed(size=seed_size) ! get size of seed
        allocate(seed(seed_size))        ! allocate array for seed
        seed = GLOBAL_SEED               ! set the seed to a constant value
        call random_seed(put=seed)       ! set the seed for consistent random numbers
        deallocate(seed)                 ! deallocate the seed array
        
        ! make a distribution of random integers in the interval [0:30]
        do i = 1, NTASKS
            call random_number(rand_val)
            task(i) = floor(rand_val * 31.0_8)  ! set up some "tasks"
        end do

        ! Start time
        tstart = time_now()

        !
        !IMPLEMENT HERE THE CODE FOR THE MASTER
        !ARRAY task contains tasks to be done. Send one element at a time to workers
        !ARRAY result should at completion contain the ranks of the workers that did
        !the corresponding tasks
        !

        ! Print out a status on how many tasks were completed by each worker
        if (nworker == 0) then
            write(*,*) 'No workers available.'
            return
        end if

        tend = time_now()

        allocate(workdone(nworker))
        workdone = 0
        do worker = 1, nworker
            tasksdone = 0
            do itask = 1, NTASKS
                if (result(itask) == worker) then
                    tasksdone = tasksdone + 1
                    workdone(worker) = workdone(worker) + task(itask)
                end if
            end do
            !write(*, '(a,i3,a,i4,a,i4,a)') 'Master: Worker ', worker, ' solved ', &
            !                               tasksdone, ' tasks with work ', workdone(worker), ' units'
        end do

        ! Print statistics
        write(*,*) 'Minimum work done by a worker:', minval(workdone)
        write(*,*) 'Maximum work done by a worker:', maxval(workdone)
        write(*,'(A,F8.2)') ' Average work done by a worker: ', sum(workdone) / nworker
        std_dev = sqrt(sum((real(workdone, kind=8) - sum(real(workdone, kind=8)) / nworker)**2) / nworker)
        write(*,'(A,F8.2)') ' Std dev of work done by a worker: ', std_dev
        write(*,'(A,F8.3,A)') ' Expected runtime without overheads:          ', workdone / 1000.0, ' seconds'
        write(*,'(A,F8.3,A)') ' Minimum runtime with perfect load balancing: ', sum(workdone) / (nworker * 1000.0), ' seconds'
        write(*,'(A,F8.3,A)') ' Runtime for master process                   ', tend - tstart, ' seconds'

    end subroutine master

    ! call this function to complete the task. It sleeps for task milliseconds
    subroutine task_function(task)
        integer, intent(in) :: task
        real(kind=8) :: sleep_duration, start_time, current_time
        sleep_duration = real(task, kind=8) / 1000.0d0 ! Convert milliseconds to seconds

        ! Simple busy wait loop to simulate sleep using high-precision timer
        start_time = time_now(); current_time = start_time
        do while (current_time - start_time < sleep_duration)
            current_time = time_now()
        end do
    end subroutine task_function

    subroutine worker(rank)
        integer, intent(in) :: rank
        !
        ! IMPLEMENT HERE THE CODE FOR THE WORKER
        ! Use a call to "task_function" to complete a task
        !
    end subroutine worker

end module task_farm_mod
!
program task_farm
    use task_farm_mod
    implicit none
    integer :: nrank, rank, ierr

    call MPI_Init(ierr)                             ! set up MPI
    call MPI_Comm_size(MPI_COMM_WORLD, nrank, ierr) ! get the total number of ranks
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)  ! get the rank of this process

    if (rank == 0) then      ! rank 0 is the master
        call master(nrank-1) ! there is nrank-1 worker processes
    else                     ! ranks in [1:nrank] are workers
        call worker(rank)
    end if

    call MPI_Finalize(ierr)  ! shutdown MPI
end program task_farm