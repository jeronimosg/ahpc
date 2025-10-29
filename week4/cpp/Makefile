CXX := nvc++

# Grid size parameters (can be overridden: make NX=1024 NY=1024)
NX ?= 512
NY ?= 512

# Precision of floating point output
PREC ?= 4  # 4=single precision, 8=double precision

#######################################################
# Optimization flags are chosen as the last definition.
# Comment out using "#" at the begining of the line or rearrange according to your needs.
#
# Fastest executable (-ffast-math removes checking for NaNs and other things)
OPT=-O3 -fast 

# Add profiling to code
#OPT=-O1 -pg

# Faster compilation time
#OPT=-O1

CXXFLAGS := $(OPT) -Wall -march=native -g -std=c++17 -DNX=$(NX) -DNY=$(NY) -DPREC=$(PREC)

# use on CPU for testing and development
ACC := -acc=multicore -Minfo=acc

# use on GPU for benchmarking and real tests of code
#ACC := -acc -gpu=cuda12.6 -Minfo=acc

.PHONY: clean all

all: sw_sequential sw_parallel
# Add sw_parallel when sw_parallel.cpp is created

sw_sequential: sw_sequential.cpp
	$(CXX) $(CXXFLAGS) sw_sequential.cpp -o sw_sequential

sw_parallel: sw_parallel.cpp
	$(CXX) $(CXXFLAGS) $(ACC) sw_parallel.cpp -o sw_parallel

clean:
	rm -f sw_sequential sw_parallel
