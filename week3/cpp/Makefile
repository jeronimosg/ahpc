CXX := clang++-12
CXX := g++ 

#######################################################
# Problem size definition
NFREQ ?= 16384

#######################################################
# Use NUMA aware memory allocator 
NUMA_ALLOCATOR =

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

CXXFLAGS := $(OPT) -Wall -march=native -g -std=c++14 -DNFREQ=$(NFREQ) -DNUMA_ALLOCATOR=$(NUMA_ALLOCATOR)

default: seq mp

seq: seismogram_seq.cpp
	$(CXX) seismogram_seq.cpp $(CXXFLAGS) -o seq

mp: seismogram_omp.cpp
	$(CXX) seismogram_omp.cpp $(CXXFLAGS) -fopenmp -o mp

clean:
	rm -fr seq mp
