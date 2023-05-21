ECEC413/622: Parallel Computer Architecture
Project 5: Gaussian Elimination Using OpenMP
Professor: Naga Kandasamy
Group members: Harrison Muller, Justin Ngo
Date: May 13, 2023

--------------------------------DESCRIPTION------------------------------------ 
In matrix notation, a system Ax = b is a system where A is a dense n*n matrix 
of coefficients, b is an n*1 vector, and x is the desired solution vector. 
This system of equations is usually solved in two stages. The first step is to 
reduce the original system of equations to an upper triangular system of form 
Ux = y, where U is an upper-triangular matrix, that is one where the subdiagonal 
entries are zero and all principal diagonal entries are equal to one. By 
division and elimination steps, such U matrix can be calculated from the original 
A matrix.

This project takes the serialized division and elimination steps of A and 
parallelize it using OpenMP. 
-------------------------------------------------------------------------------


--------------------------------COMPILE AND RUN-------------------------------- 
To compile the code:
	gcc -o gauss_elimnation gauss_elimination.c compute_gold.c -O3 -Wall -std=c99 -fopenmp -lm
Or use the Makefile included:
	make

To run the code:
	$ ./gauss_elimination {matrix_size}
For example:
	$ ./gauss_elimination 1024

To clean up executable files:
	make clean
-------------------------------------------------------------------------------
