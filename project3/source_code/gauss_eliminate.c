/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date modified: April 26, 2023
 *
 * Student names(s): Justin Ngo, Harrison Muller
 * Date: May 5, 2023
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -lpthread -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);
void *gaussian(void* args);

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
		fprintf(stderr, "matrix-size: width and height of the square matrix\n");
		exit(EXIT_FAILURE);
	}

	int matrix_size = atoi(argv[1]);

	Matrix A;			                                            /* Input matrix */
	Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
	Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

	fprintf(stderr, "Generating input matrices\n");
	srand(time (NULL));                                             /* Seed random number generator */
	A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
	U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
	U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

	/* Copy contents A matrix into U matrices */
	int i, j;
	for (i = 0; i < A.num_rows; i++) {
		for (j = 0; j < A.num_rows; j++) {
			U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
			U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
		}
	}

	fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
	struct timeval start, stop;
	gettimeofday(&start, NULL);

	int status = compute_gold(U_reference.elements, A.num_rows);

	gettimeofday(&stop, NULL);
	fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec) / (float)1000000));

	if (status < 0) {
		fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
		exit(EXIT_FAILURE);
	}

	status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
	if (status < 0) {
		fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
		exit(EXIT_FAILURE);
	}
	fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");

	/* FIXME: Perform Gaussian elimination using pthreads. 
	 * The resulting upper triangular matrix should be returned in U_mt */
	fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");

	gettimeofday(&stop, NULL);
	gauss_eliminate_using_pthreads(U_mt);

	gettimeofday(&stop, NULL);

	/* Check if pthread result matches reference solution within specified tolerance */
	fprintf(stderr, "\nChecking results\n");
	int size = matrix_size * matrix_size;
	int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
	fprintf(stderr, "Parallel run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec) / (float)1000000));
	fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

	/* Free memory allocated for matrices */
	free(A.elements);
	free(U_reference.elements);
	free(U_mt.elements);

	exit(EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U)
{

	int tid, i;
	int num_threads = 16;
	int chunk_size = (int)floor(U.num_rows / num_threads);
	int remainder = U.num_rows % num_threads;

	pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	pthread_attr_t attributes;
	pthread_attr_init(&attributes);

	pthread_barrierattr_t barrier_attributes;
	pthread_barrier_t barrier;
	pthread_barrierattr_init(&barrier_attributes);
	pthread_barrier_init(&barrier, &barrier_attributes, num_threads);

	/* Initialize mutex lock */
	pthread_mutex_t lock;
	pthread_mutex_init(&lock, NULL);

	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);

	for(tid = 0; tid < num_threads; tid++){
		int start_index = tid * chunk_size;
		int end_index = (tid == num_threads - 1) ? (start_index + chunk_size + remainder - 1)
			: (start_index + chunk_size - 1);
		//printf("start: %d, end: %d\n", start_index, end_index);

		thread_data[tid].tid = tid;
		thread_data[tid].start_index = start_index;
		thread_data[tid].end_index = end_index;
		thread_data[tid].U = &U;
		thread_data[tid].barrier = &barrier;
		thread_data[tid].lock = &lock;
		thread_data[tid].matrix_size = U.num_rows;
		thread_data[tid].num_threads = num_threads;
	}

	/* Create threads */
	for (i = 0; i < num_threads; i++)
		pthread_create(&thread_id[i], &attributes, gaussian, (void *)&thread_data[i]);

	/* Join point */
	for (i = 0; i < num_threads; i++)
		pthread_join(thread_id[i], NULL);

	/*
	   for (i = 0; i < num_threads; i++)
	   pthread_create(&thread_id[i], &attributes, gauss_elem, (void *)&thread_data[i]);

	   for (i = 0; i < num_threads; i++)
	   pthread_join(thread_id[i], NULL);
	   */

	free((void *)thread_data);
	pthread_barrier_destroy(&barrier);
}

void *gaussian(void* args) { 

	thread_data_t *thread_data = (thread_data_t *)args;
	int tid = thread_data->tid; 
	Matrix *U = thread_data->U;
	//int start = thread_data->start_index;
	//int end = thread_data->end_index;
	pthread_barrier_t *barrier = thread_data->barrier;
	//pthread_mutex_t *lock = thread_data->lock;
	int matrix_size = thread_data->matrix_size;
	int num_threads = thread_data->num_threads;
	int i, j, k;

	for(k = 0; k < matrix_size; k++){
		for (j = k+1+tid; j < matrix_size; j+=num_threads) { 
			//for(j = ; j < matrix_size; j+=num_threads) { 
			//pthread_mutex_lock(lock);

			if (U->elements[matrix_size * k + k] == 0) {
				fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
				//pthread_mutex_unlock(lock);
				pthread_exit(NULL);
			}


			U->elements[matrix_size * k + j] /= U->elements[matrix_size * k + k];	/* Division step */

			//pthread_mutex_unlock(lock);
		}

		pthread_barrier_wait(barrier);

		//pthread_mutex_lock(lock);
		if (tid == 0)
			U->elements[matrix_size * k + k] = 1;
		//pthread_mutex_unlock(lock);
		//printf("%d, %d, %f\n", k, k, U->elements[matrix_size * k + k]);

		//pthread_barrier_wait(barrier);

		for (i = k+1+tid; i < matrix_size; i+=num_threads) {
			for (j = k+1; j < matrix_size; j++) {
				//for (j = k+1+tid; j < matrix_size; j+=num_threads) {
				//pthread_mutex_lock(lock);
				U->elements[matrix_size * i + j] -= (U->elements[matrix_size * i + k] * U->elements[matrix_size * k + j]);	
				//pthread_mutex_unlock(lock);
				//fprintf(stderr, "%d, %d, tid %d\n", i, j, tid);
			}

			//pthread_mutex_lock(lock);
			U->elements[matrix_size * i + k] = 0;
			//pthread_mutex_unlock(lock);
			//printf("%d, %d, %f\n", i, k, U->elements[matrix_size * i + k]);
		}

		pthread_barrier_wait(barrier);

	}

	pthread_exit(NULL);
}

/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
	int i;
	for (i = 0; i < size; i++)
		if(fabsf(A[i] - B[i]) > tolerance) 
			return -1;
	return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
 */
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
	int i;
	Matrix M;
	M.num_columns = num_columns;
	M.num_rows = num_rows;
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

	for (i = 0; i < size; i++) {
		if (init == 0)
			M.elements[i] = 0;
		else
			M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}

	return M;
}

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
	return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
	int i;
	for (i = 0; i < M.num_rows; i++)
		if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
			return -1;

	return 0;
}
