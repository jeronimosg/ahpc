!
! Applied High Performance Computing
! 
! Task Farming with MPI
! 
! Assignment: Make an MPI task farm for analysing HEP data. To "execute" a
! task, the worker computes the accuracy of a specific set of cuts.
! The resulting accuracy should be send back from the worker to the master.
!
! Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
! Date:   September 2025
! License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
!
module task_farm_HEP_mod
    use mpi
    implicit none
    
    ! Number of cuts to try out for each event channel.
    ! BEWARE! Generates n_cuts^8 permutations to analyse.
    ! If you run many workers, you may want to increase from 3.
    integer(kind=8), parameter :: n_cuts = 3
    integer(kind=8), parameter :: n_settings = n_cuts**8
    integer(kind=8), parameter :: NO_MORE_TASKS = n_settings + 1

    ! Class to hold the main data set together with a bit of statistics
    type :: Data
        integer(kind=8) :: nevents = 0
        character(len=30) :: name(8) = (/ "averageInteractionsPerCrossing", "p_Rhad                        ", &
                                          "p_Rhad1                       ", "p_TRTTrackOccupancy           ", &
                                          "p_topoetcone40                ", "p_eTileGap3Cluster            ", &
                                          "p_phiModCalo                  ", "p_etaModCalo                  " /)
        real(kind=8), allocatable :: data(:,:) ! event data (nevents, 8)
        integer(kind=8), allocatable :: NvtxReco(:)    ! counters; don't use them
        integer(kind=8), allocatable :: p_nTracks(:)
        integer(kind=8), allocatable :: p_truthType(:) ! authorative truth about a signal
        
        logical, allocatable :: signal(:)              ! True if p_truthType=2

        real(kind=8), dimension(8) :: means_sig = 0.0d0, means_bckg = 0.0d0 ! mean of signal and background for events
        real(kind=8), dimension(8) :: flip ! flip sign if background larger than signal for type of event
    end type Data

contains

    ! High precision timer
    function time_now() result(t)
        real(kind=8)    :: t
        integer(kind=8) :: count, count_rate
        call system_clock(count, count_rate)
        t = real(count, kind=8) / real(count_rate, kind=8)
    end function time_now

    ! Routine to read events data from csv file and calculate a bit of statistics
    type(Data) function read_data() result(ds)
        implicit none
        ! name of data file
        character(len=*), parameter :: filename = "../mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv"
        integer :: file_unit = 10, ios, i
        real(kind=8) :: values(12)
        integer(kind=8) :: ev, nsig, nbckg

        ! First pass to count the number of events
        open(unit=file_unit, file=filename, status='old', iostat=ios)
        if (ios /= 0) then
            write(*,*) 'Error opening file: '//trim(filename)
            stop
        end if
        ds%nevents = -1 ! first line is a header so start at -1
        do
            read(file_unit, *, iostat=ios)
            if (ios /= 0) exit
            ds%nevents = ds%nevents + 1
        end do
        
        ! Allocate arrays to hold data
        allocate(ds%data(8,ds%nevents), ds%NvtxReco(ds%nevents), ds%p_nTracks(ds%nevents), &
                 ds%p_truthType(ds%nevents), ds%signal(ds%nevents))

        ! Second pass to read data
        rewind(file_unit); read(file_unit, *) ! rewind to start of file and skip header
        do ev = 1, ds%nevents
            read(file_unit, *, iostat=ios) values ! read a line from the file as 12 floating point values
            ! 1    2                               3         4         5      6       7                   8              9                  10           11           12
            ! rec, averageInteractionsPerCrossing, NvtxReco, p_nTracks,p_Rhad,p_Rhad1,p_TRTTrackOccupancy,p_topoetcone40,p_eTileGap3Cluster,p_phiModCalo,p_etaModCalo,p_truthType
            ds%data(:, ev) = values((/ 2, 5, 6, 7, 8, 9, 10, 11 /))
            ds%NvtxReco(ev) = int(values(3))
            ds%p_nTracks(ev) = int(values(4))
            ds%p_truthType(ev) = int(values(12))
        end do
        close(file_unit)

        ! Calculate means. Signal has p_truthType = 2
        ds%signal = ds%p_truthType == 2 ! True if signal
        nsig = count(ds%signal); nbckg = ds%nevents - nsig    ! count signal and background events
        do i = 1, 8
            ds%means_sig(i) = sum(ds%data(i,:), mask=ds%signal) / nsig                   ! average over events of each type where signal is true
            ds%means_bckg(i) = sum(ds%data(i,:), mask=(ds%signal .eqv. .false.)) / nbckg ! average over events of each type where signal is false
        end do
        ! flip sign of data and means if mean of background is less than mean of signal
        ds%flip = merge(-1.0d0, 1.0d0, ds%means_bckg < ds%means_sig)
        ds%means_sig = ds%means_sig * ds%flip
        ds%means_bckg = ds%means_bckg * ds%flip
        do i = 1, 8
            ds%data(i,:) = ds%data(i,:) * ds%flip(i)
        end do
    end function read_data

    ! call this function to complete the task. It calculates the accuracy of a given set of settings
    real(kind=8) function task_function(setting, ds) result(accuracy)
        implicit none
        real(kind=8), intent(in) :: setting(8)
        type(Data), intent(in) :: ds
        !
        logical, dimension(ds%nevents) :: pred
        integer(kind=8) :: ev, nev

        ! count for how many events the cuts predict a signal if and only if it is a true signal
        nev = 0
        do ev = 1, ds%nevents
            !nev = nev + merge(1,0, all(ds%data(:, ev) < setting) .eqv. ds%signal(ev))
            pred(ev) = all(ds%data(:, ev) < setting)
            nev = nev + merge(1,0, pred(ev) .eqv. ds%signal(ev))
        end do

        ! accuracy is percentage of events that are predicted as true signal if and only if a true signal
        accuracy = real(nev, kind=8) / ds%nevents
    end function task_function

    subroutine master(nworker, ds)
        implicit none
        integer, intent(in) :: nworker
        type(Data), intent(in) :: ds
        
        real(kind=8) :: ranges(8, n_cuts) ! ranges for cuts to explore
        real(kind=8), allocatable, dimension(:,:) :: settings
        real(kind=8), allocatable, dimension(:)   :: accuracy
        real(kind=8) :: tstart, tend, best_accuracy_score = 0.0d0
        integer(kind=8) :: idx_best = 0, k, div, idx
        integer :: i

        ! loop over different event channels and set up cuts
        do i = 1, n_cuts
            ranges(:,i) = ds%means_sig + (i-1) * (ds%means_bckg - ds%means_sig) / n_cuts
        end do
        
        ! generate list of all permutations of the cuts for each channel
        allocate(settings(8, n_settings))
        do k = 1_8, n_settings
            div = 1
            do i = 1,8 ! get 8-dimensional coordinate in n_cut^8 space corresponding to k and store range value
                idx = mod((k-1) / div, n_cuts) + 1
                settings(i, k) = ranges(i,idx)
                div = div * n_cuts
            end do
        end do

        ! results vector with the accuracy of each set of settings
        allocate(accuracy(n_settings))

        tstart = time_now() ! start time

        ! ================================================================
        !
        !IMPLEMENT HERE THE CODE FOR THE MASTER
        !The master should pass a set of settings to a worker, and the worker should return the accuracy
        !

        ! THIS CODE SHOULD BE REPLACED BY TASK FARM
        ! loop over all possible cuts and evaluate accuracy
        do k = 1, n_settings
            accuracy(k) = task_function(settings(:,k), ds)
        end do
        ! THIS CODE SHOULD BE REPLACED BY TASK FARM
        ! ================================================================

        tend = time_now() ! end time
        
        ! diagnostics
        ! extract index and value for best accuracy
        do k = 1, n_settings
            if (accuracy(k) > best_accuracy_score) then
                best_accuracy_score = accuracy(k)
                idx_best = k
            end if
        end do
        
        write(*,'(A,F0.6)') 'Best accuracy obtained :', best_accuracy_score
        write(*,'(A)') 'Final cuts :'
        do i = 1, 8
            write(*,'(A30,A,F0.6)') trim(ds%name(i)), ' : ', settings(i, idx_best) * ds%flip(i)
        end do
        
        write(*,*)
        write(*,'(A,I9)') 'Number of settings:', n_settings
        write(*,'(A,F9.4)') 'Elapsed time      :', tend - tstart
        write(*,'(A,F9.4)') 'task time [mus]   :', (tend - tstart) * 1.0d6 / n_settings

        deallocate(settings, accuracy)
    end subroutine master

    subroutine worker(rank, ds)
        implicit none
        integer, intent(in) :: rank
        type(Data), intent(in) :: ds
        
        !
        !IMPLEMENT HERE THE CODE FOR THE WORKER
        !Use a call to "task_function" to complete a task and return accuracy to master.
        !
    end subroutine worker

end module task_farm_HEP_mod

program task_farm_HEP
    use task_farm_HEP_mod
    implicit none

    integer :: nrank, rank, ierr
    type(Data) :: ds

    call MPI_Init(ierr)                         ! set up MPI
    call MPI_Comm_size(MPI_COMM_WORLD, nrank, ierr) ! get the total number of ranks
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)  ! get the rank of this process

    ! All ranks need to read the data
    ds = read_data()

    if (rank == 0) then      ! rank 0 is the master
        call master(nrank-1, ds) ! there is nrank-1 worker processes
    else                     ! ranks in [1:nrank] are workers
        call worker(rank, ds)
    end if

    call MPI_Finalize(ierr)  ! shutdown MPI
end program task_farm_HEP
