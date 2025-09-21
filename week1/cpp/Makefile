CXX := g++

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
DBG = -ggdb -Wextra -pedantic -fsanitize=address,undefined
DBG = 

CXXFLAGS := $(OPT) -Wall -march=native -g -std=c++17

default: seq vec

seq: Water_sequential.cpp
	$(CXX) Water_sequential.cpp $(CXXFLAGS) -o seq

vec: Water_vectorised.cpp
	$(CXX) Water_vectorised.cpp $(CXXFLAGS) -fopenmp-simd -o vec

clean:
	rm -fr seq vec *dSYM
