ECEC413/622: Parallel Computer Architecture
Project 2: Jacobi Solver
Professor: Naga Kandasamy
Group members: Harrison Muller, Justin Ngo
Date: May 9, 2023

--------------------------------DESCRIPTION------------------------------------ 
In a system of linear equations Ax = b, the unknown x vector can be solved by 
using the Jacobi iteration method. This method will converge if the diagonal
values of a have an absolute value greater than the sum of the absolute values
of the other a's on the row; that is, the array of a's is diagonally dominant.

This project thake the iterative method of the Jacobi and parallelize it using
pthreads with chunking and striding methods.
-------------------------------------------------------------------------------


--------------------------------COMPILE AND RUN-------------------------------- 
To compile the code:
	gcc -o jacobi_solver jacobi_solver.c compute_gold.c -O3 -Wall -std=c99 -lpthread -lm
Or use the Makefile included:
	make

To run the code:
	$ ./jacobi_solver {matrix_size} {num_threads}
For example:
	$ ./jacobi_solver 1024 8

To clean up executable files:
	make clean

Note: 
	To see the convergence printed out on the console, uncomment the "#define
	PRINT_ITERATION" line in the jacobi_solver.h file, do "make clean" and then
	"make" to compile again. After that, run as normal.
-------------------------------------------------------------------------------
