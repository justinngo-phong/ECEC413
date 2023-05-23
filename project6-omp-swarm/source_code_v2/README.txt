ECEC413/622: Parallel Computer Architecture
Project 6: Particle Swarm Optimization
Professor: Naga Kandasamy
Group members: Harrison Muller, Justin Ngo
Date: May 13, 2023

--------------------------------DESCRIPTION------------------------------------ 
This project optimizes the Particle Swarm Optimization implementation by
parallelizing it with OpenMP 
-------------------------------------------------------------------------------


--------------------------------COMPILE AND RUN-------------------------------- 
Compile using the Makefile:
	make

To run the code:
	./pso function-name dimension swarm-size xmin xmax max-iter num-threads
	function-name: name of function to optimize
	dimension: dimensionality of search space
	swarm-size: number of particles in swarm
	xmin, xmax: lower and upper bounds on search domain
	max-iter: number of iterations to run the optimizer
	num-threads: number of threads to create

For example:
	$ ./pso rastrigin 10 10000 -5.12 5.12 10000 4

To clean up executable files:
	make clean
-------------------------------------------------------------------------------
