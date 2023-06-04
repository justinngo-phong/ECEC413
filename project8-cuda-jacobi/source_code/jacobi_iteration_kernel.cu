#include "jacobi_iteration.h"

/* Device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(float *A, float *B, float *x, float *new_x, double *ssd)
{
	extern __shared__ double s_ssd[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int n = MATRIX_SIZE;

	double sum = 0.0;
	for (int j = 0; j < n; j++) {
		if (i != j)
			sum += A[i * n + j] * x[j];
	}

	new_x[i] = (B[i] - sum) / A[i * n + i];

	double diff = new_x[i] - x[i];
	s_ssd[threadIdx.x] = diff * diff;

	__syncthreads();

	/* Parallel reduction within a block */
	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride)
			s_ssd[threadIdx.x] += s_ssd[threadIdx.x + stride];
		__syncthreads();
	}

	/* Store ssd back to global memory */
	if (threadIdx.x == 0)
		atomicAdd(ssd, s_ssd[0]);
}

__global__ void transpose_matrix_kernel(float *A, float *A_col)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int n = MATRIX_SIZE;

	A_col[i * n + j] = A[j * n + i];
}

__global__ void jacobi_iteration_kernel_optimized(float *A_col, float *B, float *x, float *new_x, double *ssd)
{
	extern __shared__ double s_ssd[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int n = MATRIX_SIZE;

	double sum = 0.0;
	for (int j = 0; j < n; j++) {
		if (i != j)
			sum += A_col[j * n + i] * x[j];  // Access A_col in column-major order
	}
	__syncthreads();

	new_x[i] = (B[i] - sum) / A_col[i * n + i];  // Access A_col in column-major order

	double diff = new_x[i] - x[i];
	s_ssd[threadIdx.x] = diff * diff;

	__syncthreads();

	/* Parallel reduction within a block */
	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride)
			s_ssd[threadIdx.x] += s_ssd[threadIdx.x + stride];
		__syncthreads();
	}

	/* Store ssd back to global memory */
	if (threadIdx.x == 0)
		atomicAdd(ssd, s_ssd[0]);
}

