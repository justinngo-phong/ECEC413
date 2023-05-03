/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.
 * Author: Naga Kandasamy
 * Date modified: APril 26, 2023
 *
 * Student name(s): Justin Ngo, Harrison Muller 
 * Date modified: April 30, 2023 
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include "jacobi_solver.h"

/* Uncomment the line below to spit out debug information */ 
//#define DEBUG 
//#define PRINT_ITERATION

int main(int argc, char **argv) 
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s matrix-size num-threads\n", argv[0]);
		fprintf(stderr, "matrix-size: width of the square matrix\n");
		fprintf(stderr, "num-threads: number of worker threads to create\n");
		exit(EXIT_FAILURE);
	}

	int matrix_size = atoi(argv[1]);
	int num_threads = atoi(argv[2]);

	matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
	matrix_t mt_solution_x_v1;      /* Solution computed by pthread code using chunking */
	matrix_t mt_solution_x_v2;      /* Solution computed by pthread code using striding */

	/* Generate diagonally dominant matrix */
	fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
		fprintf(stderr, "Error creating matrix\n");
		exit(EXIT_FAILURE);
	}

	/* Create other matrices */
	B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x_v1 = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x_v2 = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

	struct timeval start, stop;

	/* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
	int max_iter = 100000; /* Maximum number of iterations to run */
	
	gettimeofday(&start, NULL);
	/* Compute using reference method */
	compute_gold(A, reference_x, B, max_iter);
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution  time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));

	display_jacobi_solution(A, reference_x, B); /* Display statistics */

	/* Compute the Jacobi solution using pthreads. 
	 * Solutions are returned in mt_solution_x_v1 and mt_solution_x_v2.
	 * */
	fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using chunking\n");
	gettimeofday(&start, NULL);
	/* Compute using chunking method */
	compute_using_pthreads_v1(A, mt_solution_x_v1, B, max_iter, num_threads);
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution  time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
	display_jacobi_solution(A, mt_solution_x_v1, B); /* Display statistics */

	fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using striding\n");
	gettimeofday(&start, NULL);
	/* Compute using striding method */
	compute_using_pthreads_v2(A, mt_solution_x_v2, B, max_iter, num_threads);
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution  time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
	display_jacobi_solution(A, mt_solution_x_v2, B); /* Display statistics */

	free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x_v1.elements);
	free(mt_solution_x_v2.elements);

	exit(EXIT_SUCCESS);
}

/* Function to perform the Jacobi calculation using pthreads using chunking. 
 * Result must be placed in mt_sol_x_v1. */
void compute_using_pthreads_v1(const matrix_t A, matrix_t mt_sol_x_v1, const matrix_t B, int max_iter, int num_threads)
{
	pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	pthread_attr_t attributes;
	pthread_attr_init(&attributes);

	/* Allocate new matrix to swap values during iterations with matrix x */
	matrix_t new_x = allocate_matrix(A.num_rows, 1, 0);

	int i;
	int chunk_size = (int)floor(mt_sol_x_v1.num_rows / num_threads);
	int remainder = mt_sol_x_v1.num_rows % num_threads;
	int converged = 0;
	float diff = 0.0;
	int num_iter = 0;

	/* Initialize barrier */
	pthread_barrierattr_t barrier_attributes;
	pthread_barrier_t barrier;
	pthread_barrierattr_init(&barrier_attributes);
	pthread_barrier_init(&barrier, &barrier_attributes, num_threads);

	/* Initialize mutex lock */
	pthread_mutex_t lock;
	pthread_mutex_init(&lock, NULL);

	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
	for (i = 0; i < num_threads; i++) {
		int start_index = i * chunk_size;
		// Decide whether this is the last thread. If it is, calculate the remainder chunk.
		int end_index = (i == num_threads - 1) ? start_index + chunk_size + remainder - 1
			: start_index + chunk_size - 1;

		thread_data[i].tid = i;
		thread_data[i].num_threads = num_threads;
		thread_data[i].A = A;
		thread_data[i].B = B;
		thread_data[i].x = &mt_sol_x_v1;
		thread_data[i].new_x = &new_x;
		thread_data[i].max_iter = max_iter;
		thread_data[i].start_index = start_index;
		thread_data[i].end_index = end_index;
		thread_data[i].barrier = &barrier;
		thread_data[i].lock = &lock;
		thread_data[i].diff = &diff;
		thread_data[i].converged = &converged;
		thread_data[i].num_iter = &num_iter;
	}

	/* Create threads */
	for (i = 0; i < num_threads; i++)
		pthread_create(&thread_id[i], &attributes, jacobi_v1, (void *)&thread_data[i]);

	/* Join point */
	for (i = 0; i < num_threads; i++)
		pthread_join(thread_id[i], NULL);

	if (num_iter < max_iter) 
		fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
	else
		fprintf(stderr, "\nMaximum allowed iterations reached\n");

	free(new_x.elements);
	free((void *)thread_data);
	pthread_barrier_destroy(&barrier);
}

void *jacobi_v1(void* args)
{
	thread_data_t *thread_data = (thread_data_t *)args;
	int tid = thread_data->tid;
	matrix_t A = thread_data->A;
	matrix_t *src = thread_data->x;
	matrix_t *dest = thread_data->new_x;
	matrix_t B = thread_data->B;
	int max_iter = thread_data->max_iter;
	int start_index = thread_data->start_index;
	int end_index = thread_data->end_index;
	float *diff = thread_data->diff;
	int *converged = thread_data->converged;
	pthread_barrier_t *barrier = thread_data->barrier;
	pthread_mutex_t *lock = thread_data->lock;
	int *num_iter = thread_data->num_iter;
	float mse;

	int i, j;
	float sum = 0.0;

	// Initialize matrix
	for (i = start_index; i <= end_index; i++)
		src->elements[i] = B.elements[i];

	/* Calculate partial diff and add to total diff then check */
	while (!*converged) {
		// only reset diff to 0 if it's the first thread
		if (tid == 0) {
			*diff = 0;
			(*num_iter)++;
		}

		//fprintf(stderr, "checkpoint 0.0	tid %d diff %f\n", tid, *diff);
		// wait until every threads has checked its tid
		pthread_barrier_wait(barrier);

		// calculate new matrix for the chunk
		for (i = start_index; i <= end_index; i++) {
			sum = 0.0;
			for (j = 0; j < A.num_columns; j++) {
				if (i != j)
					// sum += a[i][j] * x[j]
					sum += A.elements[i * A.num_columns + j] * src->elements[j];
			}
			// new_x[i] = (b[i] - sum)/a[i][i]
			dest->elements[i] = (B.elements[i] - sum) / A.elements[i * (A.num_columns + 1)];
		}

		// update partial diff
		float pdiff = 0.0;
		for (i = start_index; i <= end_index; i++) {
			pdiff += pow(dest->elements[i] - src->elements[i], 2); 
		}

		// update diff
		pthread_mutex_lock(lock);
		*diff += pdiff;
		pthread_mutex_unlock(lock);

		pthread_barrier_wait(barrier);

		mse = sqrt(*diff);

#ifdef PRINT_ITERATION
		if (tid == 0)
			fprintf(stderr, "Iteration: %d. MSE = %f\n", *num_iter, mse);
#endif

		// check if values have converged or number of iterations reaches max
		//if (((mse <= THRESHOLD) || (*num_iter == max_iter)) && (tid == 0)) { 
		if ((mse <= THRESHOLD) || (*num_iter == max_iter)){ 
			*converged = 1;

			// copy destination result to mt_sol_x_v1
			for (i = start_index; i <= end_index; i++)
				//printf("%d\n", thread_data->x);
				thread_data->x->elements[i] = dest->elements[i];
		}
		pthread_barrier_wait(barrier);

		// swap src and dest to ensure data get updated and no conflict occurs
		matrix_t *tmp = src;
		src = dest;
		dest = tmp;
	}

	pthread_exit(NULL);

}
/* FIXME: Complete this function to perform the Jacobi calculation using pthreads using striding. 
 * Result must be placed in mt_sol_x_v2. */
void compute_using_pthreads_v2(const matrix_t A, matrix_t mt_sol_x_v2, const matrix_t B, int max_iter, int num_threads)
{

	int tid;

	pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	pthread_attr_t attributes;
	pthread_attr_init(&attributes);

	matrix_t new_x = allocate_matrix(A.num_rows, 1, 0);

	int converged = 0;

	float diff = 0.0;

	int num_iter = 0;

	/*Initialze Barrier*/
	pthread_barrierattr_t barrier_attributes;
	pthread_barrier_t barrier;
	pthread_barrierattr_init(&barrier_attributes);
	pthread_barrier_init(&barrier, &barrier_attributes, num_threads);

	/*Initialize Mutex Lock*/
	pthread_mutex_t lock;
	pthread_mutex_init(&lock, NULL);

	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);

	for(tid = 0; tid < num_threads; tid++){
		thread_data[tid].tid = tid;
		thread_data[tid].num_threads = num_threads;
		thread_data[tid].A = A;
		thread_data[tid].B = B;
		thread_data[tid].x = &mt_sol_x_v2;
		thread_data[tid].new_x = &new_x;
		thread_data[tid].max_iter = max_iter;
		thread_data[tid].barrier = &barrier;
		thread_data[tid].lock = &lock;
		thread_data[tid].diff = &diff;
		thread_data[tid].converged = &converged;
		thread_data[tid].num_iter = &num_iter;
		thread_data[tid].start_index = tid;
		thread_data[tid].end_index = A.num_rows;
	}

	int i;
	/* Create threads */
	for (i = 0; i < num_threads; i++){
		pthread_create(&thread_id[i], &attributes, jacobi_v2, (void *)&thread_data[i]);
	}
	/* Join point */
	for (i = 0; i < num_threads; i++){
		pthread_join(thread_id[i], NULL);
	}
	if (num_iter < max_iter) {
		fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
	}else{
		fprintf(stderr, "\nMaximum allowed iterations reached\n");
	}
	free(new_x.elements);
	free((void *)thread_data);
	pthread_barrier_destroy(&barrier);

}

void *jacobi_v2(void* args){

	thread_data_t *thread_data = (thread_data_t *)args;
	int tid = thread_data->tid;
	int stride = thread_data->num_threads;
	matrix_t A = thread_data->A;
	matrix_t *src = thread_data->x;
	matrix_t *dest = thread_data->new_x;
	matrix_t B = thread_data->B;
	int max_iter = thread_data->max_iter;
	int start_index = thread_data->start_index;
	int end_index = thread_data->end_index;
	float *diff = thread_data->diff;
	int *converged = thread_data->converged;
	pthread_barrier_t *barrier = thread_data->barrier;
	pthread_mutex_t *lock = thread_data->lock;
	int *num_iter = thread_data->num_iter;
	float mse;
	int i = 0, j;
	float sum = 0.0;

	while(i < end_index){
		src->elements[i] = B.elements[i];
		i = i + stride;
	}

	while(!*converged) {
		if(tid == 0) {
			*diff = 0;
			(*num_iter)++;
		}

		pthread_barrier_wait(barrier);

		i = start_index;

		while(i < end_index){
			sum = 0.0;
			for(j=0; j < A.num_columns; j++){
				if(i != j)
					sum += A.elements[i * A.num_columns + j] * src->elements[j];

			}
			dest->elements[i] = (B.elements[i] - sum) / A.elements[i *(A.num_columns + 1)];

			i = i + stride;

		}

		float pdiff = 0.0;

		i = start_index;
		while(i < end_index){
			pdiff += pow(dest->elements[i] - src->elements[i], 2);
			i = i + stride;
		}

		// update diff
		pthread_mutex_lock(lock);
		*diff += pdiff;
		pthread_mutex_unlock(lock);

		// wait for every thread to update diff
		pthread_barrier_wait(barrier);

		mse = sqrt(*diff);

#ifdef PRINT_ITERATION
		if (tid == 0)
			fprintf(stderr, "Iteration: %d. MSE = %f\n", *num_iter, mse);
#endif

		// check if values have converged or number of iterations reaches max
		if ((mse <= THRESHOLD) || (*num_iter == max_iter)) { 
			*converged = 1;

			// copy destination result to mt_sol_x_v1
			for (i = start_index; i <= end_index; i++)
				thread_data->x->elements[i] = dest->elements[i];
		}

		pthread_barrier_wait(barrier);

		// swap src and dest to ensure data get updated and no conflict occurs
		matrix_t *tmp = src;
		src = dest;
		dest = tmp;

	}

	pthread_exit(NULL);
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
   */
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
	int i;    
	matrix_t M;
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

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
	int i, j;
	for (i = 0; i < M.num_rows; i++) {
		for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
		}

		fprintf(stderr, "\n");
	} 

	fprintf(stderr, "\n");
	return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
	float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
	int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}

		if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

	int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
		M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);

	/* Make diagonal entries large with respect to the entries on each row. */
	float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}

		M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

	/* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}

	return M;
}
