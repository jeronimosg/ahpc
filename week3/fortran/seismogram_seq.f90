!
! Assignment: Make an OpenMP parallelised wave propagation
! model for computing the seismic repsonse for a wave
! propagating through a horizontally stratified medium
!
module seismogram_module
    implicit none
    
    ! ======================================================
    ! The number of frequencies sets the cost of the problem
#ifndef NFREQ
#define NFREQ 65536
#endif
    integer, parameter :: nfreq = NFREQ  ! frequencies in spectrum
    
    ! ======================================================
    ! Initialize Basic Constants
    real(kind=8), parameter :: dT = 0.001d0     ! sampling distance
    integer, parameter :: nsamp = 2*nfreq  ! samples in seismogram
    real(kind=8), parameter :: dF = 1.0d0 / (nsamp * dT)  ! Frequency resolution (frequency sampling distance)
    real(kind=8), parameter :: pi = 4.0d0*atan(1.0d0)
    
contains

    ! High precision timer
    function time_now() result(t)
        real(kind=8)    :: t
        integer(kind=8) :: count, count_rate
        call system_clock(count, count_rate)
        t = real(count, kind=8) / real(count_rate, kind=8)
    end function time_now

    ! read data file with one number per line
    subroutine read_txt_file(fname, data)
        character(len=*), intent(in) :: fname
        real(kind=8), allocatable, intent(out) :: data(:)
        real(kind=8) :: value
        integer :: n, ios, i
        
        ! First pass: count lines
        n = 0
        open(unit=20, file=fname, status="old", action="read")
        do
            read(20, *, iostat=ios) value
            if (ios /= 0) exit
            n = n + 1
        end do
        
        ! Allocate array (0-based)
        allocate(data(0:n-1))
        
        ! Second pass: read data
        rewind(20)
        do i = 0, n-1
            read(20, *) data(i)
        end do
        close(20)
    end subroutine read_txt_file

    ! Cooley–Tukey FFT (in-place computation)
    recursive subroutine fft(x)
        complex(kind=8), intent(inout) :: x(0:)
        integer :: N, i, k
        complex(kind=8), allocatable :: even(:), odd(:)
        complex(kind=8) :: t
        
        N = size(x)
        if (N <= 1) return
        
        ! divide
        allocate(even(0:N/2-1), odd(0:N/2-1))
        do i = 0, N/2-1
            even(i) = x(2*i)
            odd(i)  = x(2*i+1)
        end do
        
        ! conquer
        call fft(even)
        call fft(odd)
        
        ! combine
        do k = 0, N/2-1
            t = exp(cmplx(0.0d0, -2.0d0 * pi * k / N, kind=8)) * odd(k)
            x(k)     = even(k) + t
            x(k+N/2) = even(k) - t
        end do
        deallocate(even, odd)
    end subroutine fft

    ! inverse fft (in-place)
    subroutine ifft(x)
        complex(kind=8), intent(inout) :: x(0:)
        real(kind=8) :: inv_size
        integer :: i
        inv_size = 1.0d0 / size(x)
        do i = 0, size(x)-1     ! conjugate the input
            x(i) = conjg(x(i))
        end do
        call fft(x)             ! forward fft        
        do i = 0, size(x)-1     ! conjugate the output and scale the numbers
            x(i) = conjg(x(i)) * inv_size
        end do
    end subroutine ifft

    ! Main routine: propgate wave through layers and compute seismogram
    subroutine propagator(wave, density, velocity, seismogram)
        real(kind=8), intent(in) :: wave(0:)
        real(kind=8), intent(in) :: density(0:)
        real(kind=8), intent(in) :: velocity(0:)
        real(kind=8), allocatable, intent(out) :: seismogram(:)
        
        integer :: nlayers
        real(kind=8), allocatable :: imp(:)           ! impedance
        real(kind=8), allocatable :: ref(:)           ! reflection coefficient
        complex(kind=8), allocatable :: half_filter(:) ! half filter
        complex(kind=8), allocatable :: filter(:)     ! full filter
        real(kind=8), allocatable :: half_wave(:)     ! half wave
        complex(kind=8), allocatable :: wave_spectral(:) ! FFT(wave)
        complex(kind=8), allocatable :: U(:)          ! Upgoing waves
        complex(kind=8), allocatable :: Upad(:)       ! FFT(seismogram)
        integer :: n_wave                        ! size of wave array
        integer :: lc                            ! low-cut indices
        real(kind=8) :: mean_wave                     ! wave zero point
        integer :: i, n
        complex(kind=8) :: omega, exp_omega, Y
        real(kind=8) :: tstart, tstart1, tstart2, tend, tend1, tend2
        
        tstart = time_now() ! start time (seconds)
        
        nlayers = size(density)
        n_wave = size(wave)
        lc = floor(nfreq * 0.01d0)
        mean_wave = 0.0d0
        
        ! Allocate arrays (all 0-based to match C++ version)
        allocate(imp(0:nlayers-1), ref(0:nlayers-2), half_filter(0:nfreq/2), &
                 filter(0:nfreq), half_wave(0:nfreq), wave_spectral(0:nsamp-1), &
                 U(0:nfreq), Upad(0:nsamp-1), seismogram(0:nsamp-1))
        
        ! Initialize arrays
        half_filter = cmplx(1.0d0, 0.0d0, kind=8)
        half_wave = 0.0d0
        U = cmplx(0.0d0, 0.0d0, kind=8)
        Upad = cmplx(0.0d0, 0.0d0, kind=8)
        
        ! Compute seismic impedance
        do i = 0, nlayers-1
            imp(i) = density(i) * velocity(i)
        end do
        
        ! Reflection coefficients at the base of the layers :
        do i = 0, nlayers-2
            ref(i) = (imp(i+1) - imp(i)) / (imp(i+1) + imp(i))
        end do
        
        ! Spectral window (both low- and high cut)
        do i = 0, lc
            half_filter(i) = (sin(pi*(2*i-lc)/(2.0d0*lc)))/2.0d0 + 0.5d0
        end do
        
        do i = 0, nfreq/2
            filter(i) = half_filter(i)
        end do
        
        filter(nfreq/2+1) = cmplx(1.0d0, 0.0d0, kind=8)
        
        do i = nfreq/2+2, nfreq
            filter(i) = half_filter(nfreq+1-i)
        end do
        
        do i = 0, n_wave/2-1
            half_wave(i) = wave(n_wave/2-1+i)
        end do
        
        do i = 0, 2*nfreq-1
            if (i < nfreq) then
                wave_spectral(i) = cmplx(half_wave(i), 0.0d0, kind=8)
            else
                wave_spectral(i) = cmplx(half_wave(2*nfreq-i), 0.0d0, kind=8)
            end if
            mean_wave = mean_wave + real(wave_spectral(i))
        end do
        
        mean_wave = mean_wave / nsamp
        
        do i = 0, 2*nfreq-1
            wave_spectral(i) = wave_spectral(i) - mean_wave
        end do
        
        ! Fourier transform waveform to frequency domain
        tstart1 = time_now() ! start time (seconds)
        call fft(wave_spectral)
        tend1 = time_now() ! end time (seconds)
        
        ! spectrum U of upgoing waves just below the surface.
        ! See eq. (43) and (44) in Ganley (1981).
        
        do i = 0, nfreq
            omega = cmplx(0.0d0, 2.0d0*pi*i*dF, kind=8)
            exp_omega = exp(-dT * omega)
            Y = cmplx(0.0d0, 0.0d0, kind=8)
            do n = nlayers-2, 0, -1
                Y = exp_omega * (ref(n) + Y) / (1.0d0 + ref(n)*Y)
            end do
            U(i) = Y
        end do
        
        ! Compute seismogram
        do i = 0, nfreq
            U(i) = U(i) * filter(i)
            Upad(i) = U(i)
        end do
        
        do i = nfreq+1, nsamp-1
            Upad(i) = conjg(Upad(nsamp - i))
        end do
        
        do i = 0, nsamp-1
            Upad(i) = Upad(i) * wave_spectral(i)
        end do
        
        ! Fourier transform back again
        tstart2 = time_now() ! start time (seconds)
        call ifft(Upad)
        tend2 = time_now() ! end time (seconds)
        
        do i = 0, nsamp-1
            seismogram(i) = real(Upad(i))
        end do
        
        tend = time_now() ! end time (seconds)
        
        write(*, '(A,F9.5)') "Wave zero-point        : ", mean_wave
        write(*, '(A,4(F9.5,A))') "Seismogram first coeff : ", &
            seismogram(0), ", ", seismogram(1), ", ", seismogram(2), ", ", seismogram(3), ""
        write(*, '(A,F9.4)') "Elapsed time for FFTs  :", (tend1 - tstart1) + (tend2 - tstart2)
        write(*, '(A,F9.4)') "Elapsed time without FFTs:", tend - tstart - ((tend1 - tstart1) + (tend2 - tstart2))
        write(*, '(A,F9.4)') "Elapsed time:", tend - tstart
        
    end subroutine propagator

end module seismogram_module

!======================================================================================================
!======================== Main function ===============================================================
!======================================================================================================
program seismogram_seq
    use seismogram_module
    implicit none
    
    real(kind=8), allocatable, dimension(:) :: wave, & ! input impulse wave in medium
                                            density, & ! density as a function of depth
                                           velocity, & ! seismic wave velocity as a function of depth
                                         seismogram    ! final seismogram
    real(kind=8) :: checksum
    integer :: i
    
    ! Load the wave profile and the density and velocity structure of the rock from text files
    call read_txt_file("../wave_data.txt", wave)
    call read_txt_file("../density_data.txt", density)
    call read_txt_file("../velocity_data.txt", velocity)
    
    ! Propagate wave
    call propagator(wave, density, velocity, seismogram)
    
    ! write output and make checksum
    checksum = 0.0d0
    open(unit=10, file="seismogram.txt", status="replace", action="write")
    do i = 0, nsamp-1
        write(10, *) seismogram(i)
        checksum = checksum + abs(seismogram(i))
    end do
    close(10)    
    write(*, '(A,F20.15)') "Checksum    :", checksum

end program seismogram_seq
