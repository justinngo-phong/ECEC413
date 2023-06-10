ECEC413/622: Parallel Computer Architecture
Project 9: CUDA Seperable 2D Convolution
Professor: Naga Kandasamy
Group members: Harrison Muller, Justin Ngo
Date: June 10, 2023

--------------------------------DESCRIPTION------------------------------------ 
This project calculates a 2D seperable convolution on the GPU with CUDA.
-------------------------------------------------------------------------------


--------------------------------COMPILE AND RUN-------------------------------- 
Compile using the Makefile:
	make clean && make

To run the code:
	$ ./seperable_convolution {num_rows} {num_cols}

For example:
	$ ./seperable_convolution 4096 4096

To clean up executable files:
	make clean
-------------------------------------------------------------------------------
