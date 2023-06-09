/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Build as follows: make clean && make

 * Author: Naga Kandasamy
 * Date modified: May 31, 2023
 *
 * Student name(s): Justin Ngo, Harrison Muller
 * Date modified: June 2, 2023
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */ 
//#define DEBUG 

int main(int argc, char **argv) 
{
	if (argc > 1) {
		printf("This program accepts no arguments\n");
		exit(EXIT_FAILURE);
	}

	matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
	matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
	matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

	/* Initialize the random number generator */
	srand(time(NULL));

	/* Generate diagonally dominant matrix */ 
	printf("\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
	A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL) {
		printf("Error creating matrix\n");
		exit(EXIT_FAILURE);
	}

	/* Create the other vectors */
	B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);
	reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

	struct timeval start, stop;
	/* Compute Jacobi solution on CPU */
	printf("\nPerforming Jacobi iteration on the CPU\n");
	gettimeofday(&start, NULL);
	compute_gold(A, reference_x, B);
	gettimeofday(&stop, NULL);
	printf("Gold method execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
	display_jacobi_solution(A, reference_x, B); /* Display statistics */

	/* Compute Jacobi solution on device. Solutions are returned 
	   in gpu_naive_solution_x and gpu_opt_solution_x. */
	printf("\nPerforming Jacobi iteration on device using naive method with %d 1D block size\n", THREAD_BLOCK_SIZE);
	gettimeofday(&start, NULL);
	compute_on_device_naive(A, gpu_naive_solution_x, B);
	gettimeofday(&stop, NULL);
	printf("Total naive method execution time including communication time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
	display_jacobi_solution(A, gpu_naive_solution_x, B); /* Display statistics */

	printf("\nPerforming Jacobi iteration on device using optimized method %d 1D block size\n", THREAD_BLOCK_SIZE);
	gettimeofday(&start, NULL);
	compute_on_device_optimized(A, gpu_opt_solution_x, B);
	gettimeofday(&stop, NULL);
	printf("Total optimized method execution time including communication time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
	display_jacobi_solution(A, gpu_opt_solution_x, B); 

	free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(gpu_naive_solution_x.elements);
	free(gpu_opt_solution_x.elements);

	exit(EXIT_SUCCESS);
}


/* Perform Jacobi calculation on device using the naive method */
void compute_on_device_naive(const matrix_t A, matrix_t gpu_naive_sol_x, const matrix_t B)
{
	/* Initialize vector x to b */
	for (int i = 0; i < MATRIX_SIZE; i++)
		gpu_naive_sol_x.elements[i] = B.elements[i];

	/* Allocate device memory for A, B, x, SSD and copy over data */
	matrix_t d_A = allocate_matrix_on_device(A);
	copy_matrix_to_device(d_A, A);

	matrix_t d_B = allocate_matrix_on_device(B);
	copy_matrix_to_device(d_B, B);

	matrix_t d_x = allocate_matrix_on_device(gpu_naive_sol_x);
	copy_matrix_to_device(d_x, gpu_naive_sol_x);

	matrix_t d_x_new = allocate_matrix_on_device(gpu_naive_sol_x);

	double *d_ssd;
	cudaMalloc((void **)&d_ssd, sizeof(double));
	check_CUDA_error("Allocate memory and copy data to kernel failed");

	/* Set up execution grid on device */
	dim3 blockDim(1, THREAD_BLOCK_SIZE, 1);
	dim3 gridDim(1, (MATRIX_SIZE + THREAD_BLOCK_SIZE- 1) / THREAD_BLOCK_SIZE);
	check_CUDA_error("Set up execution grid failed");

	/* Perform Jacobi with naive method */
	unsigned int done = 0;
	unsigned int num_iter = 0;
	double mse, ssd;
	struct timeval start, stop;

	gettimeofday(&start, NULL);
	while(!done) {
		num_iter++;

		/* cudaMemset ssd to 0 */
		cudaMemset(d_ssd, 0, sizeof(double));

		jacobi_iteration_kernel_naive<<<gridDim, blockDim>>>(d_A.elements, \
				d_B.elements, d_x.elements, d_x_new.elements, d_ssd);
		cudaDeviceSynchronize();
		check_CUDA_error("Kernel failed");

		cudaMemcpy(&ssd, d_ssd, sizeof(double), cudaMemcpyDeviceToHost);

		mse = sqrt(ssd);
#ifdef PRINT_ITERATIONS
		printf("Iteration: %d. MSE = %f\n", num_iter, mse);
#endif

		if (mse <= THRESHOLD)
			done = 1;

		/* Flip pointers for ping-pong buffers */
		//cudaMemcpy(d_x.elements, d_x_new.elements, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
		float* temp_elements = d_x.elements;
		d_x.elements = d_x_new.elements;
		d_x_new.elements = temp_elements;

	}
	gettimeofday(&stop, NULL);
	printf("Convergence achieved after %d iterations using naive method\n", num_iter);
	printf("Naive method execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));

	copy_matrix_from_device(gpu_naive_sol_x, d_x);

	cudaFree(d_x.elements);
	cudaFree(d_x_new.elements);
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_ssd);
}

/* Perform Jacobi calculation on device using the optimized method */
void compute_on_device_optimized(const matrix_t A, matrix_t gpu_opt_sol_x, const matrix_t B)
{
	/* Initialize vector x to b */
	for (int i = 0; i < MATRIX_SIZE; i++)
		gpu_opt_sol_x.elements[i] = B.elements[i];

	/* Allocate device memory for A_T, B, x, SSD and copy over data */
	matrix_t A_T = allocate_matrix_on_host(MATRIX_SIZE, MATRIX_SIZE, 0);
	matrix_t d_A_T = allocate_matrix_on_device(A_T);

	/* Transpose matrix A on the host */
	for (int i = 0; i < MATRIX_SIZE; i++) {
		for (int j = 0; j < MATRIX_SIZE; j++) {
			A_T.elements[i * MATRIX_SIZE + j] = A.elements[j * MATRIX_SIZE + i];
		}
	}
	copy_matrix_to_device(d_A_T, A_T);

#ifdef DEBUG
	print_matrix(A_T);
#endif

	matrix_t d_B = allocate_matrix_on_device(B);
	copy_matrix_to_device(d_B, B);

	matrix_t d_x = allocate_matrix_on_device(gpu_opt_sol_x);
	copy_matrix_to_device(d_x, gpu_opt_sol_x);

	matrix_t d_x_new = allocate_matrix_on_device(gpu_opt_sol_x);

	/* Set up execution grid on device */
	dim3 blockDim(1, THREAD_BLOCK_SIZE, 1);
	dim3 gridDim(1, (MATRIX_SIZE + THREAD_BLOCK_SIZE- 1) / THREAD_BLOCK_SIZE);
	check_CUDA_error("Set up execution grid failed");

	double *d_ssd;
	cudaMalloc((void **)&d_ssd, sizeof(double));
	check_CUDA_error("Allocate memory and copy data to kernel failed");

	/* Perform Jacobi with optimized method */
	unsigned int done = 0;
	unsigned int num_iter = 0;
	double mse, ssd;
	struct timeval start, stop;

	gettimeofday(&start, NULL);
	while(!done) {
		num_iter++;

		/* cudaMemset ssd to 0 */
		cudaMemset(d_ssd, 0, sizeof(double));

		jacobi_iteration_kernel_optimized<<<gridDim, blockDim>>>(d_A_T.elements, \
				d_B.elements, d_x.elements, d_x_new.elements, d_ssd);
		cudaDeviceSynchronize();
		check_CUDA_error("Kernel failed");

		cudaMemcpy(&ssd, d_ssd, sizeof(double), cudaMemcpyDeviceToHost);

		mse = sqrt(ssd);
#ifdef PRINT_ITERATIONS
		printf("Iteration: %d. MSE = %f\n", num_iter, mse);
#endif

		if (mse <= THRESHOLD)
			done = 1;

		/* Flip pointers for ping-pong buffers */
		//cudaMemcpy(d_x.elements, d_x_new.elements, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
		float* temp_elements = d_x.elements;
		d_x.elements = d_x_new.elements;
		d_x_new.elements = temp_elements;
	}
	gettimeofday(&stop, NULL);
	printf("Convergence achieved after %d iterations using optimized method\n", num_iter);
	printf("Optimized method execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));

	copy_matrix_from_device(gpu_opt_sol_x, d_x);

	cudaFree(d_x.elements);
	cudaFree(d_x_new.elements);
	cudaFree(d_A_T.elements);
	cudaFree(d_B.elements);
	cudaFree(d_ssd);
	free((void *) A_T.elements);
}

/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
	matrix_t Mdevice = M;
	int size = M.num_rows * M.num_columns * sizeof(float);
	cudaMalloc((void **)&Mdevice.elements, size);
	return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
 */
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{	
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows;
	int size = M.num_rows * M.num_columns;

	M.elements = (float *)malloc(size * sizeof(float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0) 
			M.elements[i] = 0; 
		else
			M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}

	return M;
}	

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
	int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
	Mdevice.num_rows = Mhost.num_rows;
	Mdevice.num_columns = Mhost.num_columns;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
	return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
	int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
	return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
		for (unsigned int j = 0; j < M.num_columns; j++) {
			printf("%f ", M.elements[i * M.num_columns + j]);
		}

		printf("\n");
	} 

	printf("\n");
	return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
	float r = rand()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	

	return;    
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));
	if (M.elements == NULL)
		return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
	unsigned int i, j;
	for (i = 0; i < size; i++)
		M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);

	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}

		M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

	return M;
}

