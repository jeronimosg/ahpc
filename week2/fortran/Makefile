F90 := mpifort

#######################################################
# Optimization flags are chosen as the last definition.
# Comment out using "#" at the begining of the line or rearrange according to your needs.
#
# Fastest executable (-ffast-math removes checking for NaNs and other things)
OPT=-O3

# Debug flags with runtime checks (slow!)
DBG = -fbacktrace -ffpe-trap=invalid,zero,overflow -fcheck=all
DBG =

FFLAGS := $(OPT) $(DBG) -Wall -std=f2003 -march=native

default: task_farm task_farm_HEP task_farm_HEP_seq

task_farm: task_farm.f90
	$(F90) task_farm.f90 $(FFLAGS) -o task_farm

task_farm_HEP: task_farm_HEP.f90
	$(F90) task_farm_HEP.f90 $(FFLAGS) -o task_farm_HEP

task_farm_HEP_seq: task_farm_HEP_seq.f90
	$(F90) task_farm_HEP_seq.f90 $(FFLAGS) -o task_farm_HEP_seq

clean:
	rm -f task_farm task_farm_HEP task_farm_HEP_seq *.mod
