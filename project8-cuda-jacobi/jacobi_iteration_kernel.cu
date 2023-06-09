#include "jacobi_iteration.h"

/* Device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(float *A, float *B, float *x, float *new_x, double *ssd)
{
	__shared__ double s_ssd[THREAD_BLOCK_SIZE];

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int n = MATRIX_SIZE;

	if (i < n) {
		double sum = -A[i * n + i] * x[i];
		for (int j = 0; j < n; j++) {
			sum += A[i * n + j] * x[j];
		}

		new_x[i] = (B[i] - sum) / A[i * n + i];

		double diff = new_x[i] - x[i];
		s_ssd[threadIdx.y] = diff * diff;

		__syncthreads();

		/* Parallel reduction within a block */
		for (unsigned int stride = blockDim.y >> 1; stride > 0; stride >>= 1) {
			if (threadIdx.y < stride)
				s_ssd[threadIdx.y] += s_ssd[threadIdx.y + stride];
			__syncthreads();
		}
	}

	/* Store ssd back to global memory */
	if (threadIdx.y == 0)
		atomicAdd(ssd, s_ssd[0]);
}

__global__ void jacobi_iteration_kernel_optimized(float *A_T, float *B, float *x, float *new_x, double *ssd)
{
	__shared__ double s_ssd[THREAD_BLOCK_SIZE];

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int n = MATRIX_SIZE;

	if (i < n) {
		double sum = -A_T[i * n + i] * x[i];
		for (int j = 0; j < n; j++) {
			sum += A_T[j * n + i] * x[j];  // Access A_T in column-major order
		}

		new_x[i] = (B[i] - sum) / A_T[i * n + i];  // Access A_T in column-major order

		double diff = new_x[i] - x[i];
		s_ssd[threadIdx.y] = diff * diff;

		__syncthreads();

		/* Parallel reduction within a block */
		for (unsigned int stride = blockDim.y >> 1; stride > 0; stride >>= 1) {
			if (threadIdx.y < stride)
				s_ssd[threadIdx.y] += s_ssd[threadIdx.y + stride];
			__syncthreads();
		}

	}
	/* Store ssd back to global memory */
	if (threadIdx.y == 0)
		atomicAdd(ssd, s_ssd[0]);
}

