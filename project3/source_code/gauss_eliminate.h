#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

/* Matrix Structure declaration */
typedef struct {
    unsigned int num_columns;   /* Width of the matrix */ 
    unsigned int num_rows;      /* Height of the matrix */
    float* elements;            /* Pointer to the first element of the matrix */
} Matrix;

typedef struct thread_data_s {

int tid;
int start_index;
int end_index;
Matrix U; 
pthread_barrier_t *barrier;
pthread_mutex_t *lock;

}

#endif /* _MATRIXMUL_H_ */

