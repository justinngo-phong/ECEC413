ECEC413/622: Parallel Computer Architecture
Project 1: Ways to SAXPY
Professor: Naga Kandasamy
Group members: Harrison Muller, Justin Ngo
Date: 4/20/2023

--------------------------------DESCRIPTION------------------------------------ 
This project compares two different ways of parallelizing the SAXPY (Single-
Precision AX Plus Y" loop. The routine takes in two vectors of 32-bit floating-
point values x and y with n elements each, and a scalar value a as input. It
multiplies each element x[i] by a and adds the result to y[i].

The two different ways of parallelizing the routine are:
- Chunking method: With k threads, each thread calculate SAXPY in parallel on 
smaller chunks of these vectors.
- Striding method: With k threads, each thread strides over elements of the 
vectors with some stride length, calculating SAXPY along the way.
-------------------------------------------------------------------------------


--------------------------------COMPILE AND RUN-------------------------------- 
To compile the code:
	gcc -o saxpy saxpy.c -O3 -Wall -std=c99 -lpthread -lm
Or use the Makefile included:
	make

To run the code:
	$ ./saxpy {number of elements} {number of threads}
For example:
	$ ./saxpy 1000000 8

To clean up executable files:
	make clean
-------------------------------------------------------------------------------
