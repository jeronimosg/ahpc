F90 := gfortran

#######################################################
# Optimization flags are chosen as the last definition.
# Comment out using "#" at the begining of the line or rearrange according to your needs.
#
# Fastest executable (-ffast-math removes checking for NaNs and other things)
#OPT=-O3 -ffast-math

# Add profiling to code
#OPT=-O1 -pg

# Faster compilation time
OPT=-O1

# Debug flags with runtime checks (slow!)
DBG =-fbacktrace -ffpe-trap=invalid,zero,overflow -fcheck=all
DBG = 

FFLAGS := $(OPT) $(DBG) -Wall -march=native -g -std=f2003

default: seq vec

seq: Water_sequential.f90
	$(F90) Water_sequential.f90 $(FFLAGS) -o seq

vec: Water_vectorised.f90
	$(F90) Water_vectorised.f90 $(FFLAGS) -fopenmp-simd -o vec

clean:
	rm -fr seq vec water_mod.mod
