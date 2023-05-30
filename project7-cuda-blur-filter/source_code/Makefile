# A simple CUDA makefile
#
# Author: Naga Kandasamy
# Date: May 3, 2019
#
# CUDA depends on two things:
#  1) The CUDA nvcc compiler, which needs to be on your path,
#	or called directly, which we do here
#  2) The CUDA shared library being available at runtime,
#	which we make available by setting the LD_LIBRARY_PATH
#	variable for the duration of the makefile.
#   
#   Note that you can set your PATH and LD_LIBRARY_PATH variables as part of your
# .bash_profile so that you can compile and run without using this makefile.


CUDA_PATH ?= /usr/local/cuda-10.1
HOST_COMPILER ?= g++
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) 
NVCCFLAGS	:= -m64 -O3 -gencode arch=compute_75,code=compute_75

all: blur_filter

blur_filter: blur_filter.cu blur_filter_kernel.cu blur_filter_gold.cpp 
	$(NVCC) -o blur_filter blur_filter.cu blur_filter_gold.cpp $(NVCCFLAGS)

clean:
	rm blur_filter 
