FC := gfortran

#######################################################
# Problem size definition
NFREQ ?= 16384

#######################################################
# Optimization flags are chosen as the last definition.
# Comment out using "#" at the begining of the line or rearrange according to your needs.
#
# Fastest executable (-ffast-math removes checking for NaNs and other things)
OPT=-O3 -ffast-math

# Add thread checking to code
#OPT=-O1 -fsanitize=thread -fPIE -pie

# Faster compilation time
#OPT=-O1

FFLAGS := $(OPT) -Wall -march=native -g -cpp -DNFREQ=$(NFREQ)

default: seq mp

seq: seismogram_seq.f90
	$(FC) seismogram_seq.f90 $(FFLAGS) -o seq

mp: seismogram_omp.f90
	$(FC) seismogram_omp.f90 $(FFLAGS) -fopenmp -o mp

clean:
	rm -fr seq mp *.mod

