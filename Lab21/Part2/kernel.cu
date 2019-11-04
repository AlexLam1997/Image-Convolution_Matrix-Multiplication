
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm> 
#include <time.h>

#include "Inputs/A_10_2.h"
#include "Inputs/A_32.h"
#include "Inputs/A_512.h"
#include "Inputs/A_1024.h"

#include "Inputs/b_10.h"
#include "Inputs/b_32.h"
#include "Inputs/b_512.h"
#include "Inputs/b_1024.h"
#include "Inputs/X_32.h"
#include "Inputs/X_512.h"
#include "Inputs/X_1024.h"

// ------------------------------------------------
using namespace std;
#include <iostream>

__global__ void gpu_process(float* x_temp, float* d_a, float* d_b, int num_threads, int num_blocks);
__global__ void sum_temp(float* x_temp, float* result, int num_threads, int num_blocks);
void pre_process(float** x_temp, float** x, float** d_A, float** d_B, float* A, float* B);
float run_process(int num_threads, float* d_a, float* d_b, float* x_temp, float* x);

const int N = 10;

void displayVector(float A[N]);
void displayFlat(float A[N * N]);
bool inverse(float A[N][N], float* inverse);
void adjoint(float A[N][N], float adj[N][N]);
float determinant(float A[N][N], int n);
void getCofactor(float A[N][N], float temp[N][N], int p, int q, int n);
void subtract(float* output, float A[N], float B[N]);
void display(float A[N][N]);

int main()
{
	float* x_temp, * x, * d_A, * d_B;
	float* difference = (float*)malloc(N * sizeof(float));

	if (N == 10)
	{
		float* A_orig = (float*)malloc(N * N * sizeof(float));
		float* d_orig_A;
		float inv_A[N * N];
		int number_of_threads = 2048;
		float time_spent_inverse, time_spent_multiplication;

		clock_t begin = clock();
		inverse(A_10, inv_A);
		time_spent_inverse = (float)(clock() - begin) / CLOCKS_PER_SEC;

		cout << "The input matrix is: \n";
		display(A_10);
		cout << "\nThe inverse is: \n";
		displayFlat(inv_A);
		printf("\nTime Spent Inversing matrix: %d seconds \n", time_spent_inverse);

		size_t matrixAsize = N * N * sizeof(float);
		cudaMalloc((void**) &d_orig_A, matrixAsize);

		// Flatten A
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				// change argument here for different weight matrices
				A_orig[i * N + j] = A_10[i][j];
			}
		}

		cudaMemcpy(d_orig_A, A_orig, matrixAsize, cudaMemcpyHostToDevice);
		pre_process(&x_temp, &x, &d_A, &d_B, inv_A, b_10);

		begin = clock();
		// inv_A * b
		run_process(number_of_threads, d_A, d_B, x_temp, x);
		cout << "X is: \n";
		displayVector(x);
		// A * x, x now holds b 
		run_process(number_of_threads, d_orig_A, x, x_temp, x);
		time_spent_multiplication = (float)(clock() - begin) / CLOCKS_PER_SEC;

		subtract(difference, x, b_10);
		displayVector(difference);
	}
	else
	{
		//float* b_provided = b_512;
		//float* a_provided = A_512;
		//float* x_provided = X_512;

		//pre_process(&x_temp, &x, &d_A, &d_B, a_provided, x_provided);

		//if (N != 1024)
		//{
		//	// RUN FOR ALL OTHER SIZES WHERE TIMING INFO NOT NEEDED
		//	int number_of_threads = 2048;
		//	run_process(number_of_threads, d_A, d_B, x_temp, x);
		//	cout << "\nA*x is: \n";
		//	displayVector(x);
		//	subtract(difference, x, b_provided);
		//	cout << "\nA*x-B is: \n";
		//	displayVector(difference);
		//}
		//else
		//{
		//	// ONLY RUN FOR 1024X1024 matrix
		//	 //Run through matrix multiplication with numthreads 
		//	int max_thread_power = 11;
		//	printf("\nMatrix Dimension: %d \n", N);
		//	for (int i = 0; i <= max_thread_power; i++) {
		//		int number_of_threads = pow(2, i);
		//		float duration = run_process(number_of_threads, d_A, d_B, x_temp, x);
		//		cout << "\n Number of threads: " << number_of_threads << "\t Run time: " << scientific << duration;
		//		cout << "\nX is: \n";
		//		displayVector(x);
		//	}
		//}

	}
	free(difference);
	cudaFree(x_temp);
	cudaFree(x);
	cudaFree(d_A);
	cudaFree(d_B);

	return 0;
}

float run_process(int num_threads, float* d_a, float* d_b, float* x_temp, float* x)
{
	int block_number = pow(2, num_threads / 1024);
	int threads_per_block = num_threads / block_number;

	float time_spent = 0.0;
	clock_t begin = clock();

	gpu_process << <block_number, threads_per_block >> > (x_temp, d_a, d_b, threads_per_block, block_number);
	cudaDeviceSynchronize();
	sum_temp << <block_number, threads_per_block >> > (x_temp, x, threads_per_block, block_number);
	//serial_sum_temp(x_temp, x);
	cudaDeviceSynchronize();

	clock_t end = clock();
	time_spent += (float)(end - begin) / CLOCKS_PER_SEC;
	return time_spent;
}

// Intermediate step in matrix multiplication of d_a x d_b.
// Multiplications are done and stored into x_temp
__global__ void gpu_process(float* x_temp, float* d_a, float* d_b, int num_threads, int num_blocks)
{
	int start, end;

	int thread_size = (N * N) / (num_blocks * num_threads);
	if (thread_size == 0) thread_size = 1;

	start = thread_size * (blockIdx.x * blockDim.x + threadIdx.x);
	end = start + thread_size;

	if (end > (N * N))
	{
		end = (N * N);
	}

	for (int i = start; i < end; i++)
	{
		x_temp[i] = d_a[i] * d_b[i % N];
	}
}

__global__ void sum_temp(float* x_temp, float* result, int num_threads, int num_blocks)
{
	// reinit results
	for (int j = 0; j < N; j++)
	{
		result[j] = 0;
	}

	int start, end;
	int thread_size = N / (num_blocks * num_threads);
	if (thread_size == 0) thread_size = 1;

	start = thread_size * (blockIdx.x * blockDim.x + threadIdx.x);
	end = start + thread_size;

	if (end > N) {
		end = N;
	}

	for (int i = start; i < end; i++)
	{
		for (int j = 0; j < N; j++)
		{
			// sum up rows 
			result[i] += x_temp[i * N + j];
		}
	}
}

// Allocated gpu memory for A x B multiplication. A is matrix, B is vector
// d_A holds gpu version of A, d_B for B
// x_temp holds intermediate multiplications. Shared Mem. 
// x holds final vector result
void pre_process(float** x_temp, float** x, float** d_A, float** d_B, float* A, float* B)
{
	unsigned error;

	// allocate and copy into device
	size_t matrixAsize = (size_t)(N * N * sizeof(float));
	size_t matrixBsize = (size_t)(N * sizeof(float));

	cudaMalloc((void**) & *d_A, matrixAsize);
	cudaMalloc((void**) & *d_B, matrixBsize);
	cudaMallocManaged((void**) & *x_temp, matrixAsize);

	cudaMemcpy(*d_A, A, matrixAsize, cudaMemcpyHostToDevice);
	cudaMemcpy(*d_B, B, matrixBsize, cudaMemcpyHostToDevice);

	// allocate shared memory for x
	cudaMallocManaged(x, matrixBsize);
}

void subtract(float* output, float A[N], float B[N])
{
	for (int i = 0; i < N; i++) {
		output[i] = A[i] - B[i];
	}
}

bool inverse(float A[N][N], float* inverse)
{
	float det = determinant(A, N);
	if (det == 0)
	{
		cout << "No inverse";
		return false;
	}

	// Find adjoint 
	float adj[N][N];
	adjoint(A, adj);

	// Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			inverse[i * N + j] = adj[i][j] / float(det);

	return true;
}

void getCofactor(float A[N][N], float temp[N][N], int p, int q, int n)
{
	int i = 0, j = 0;

	// Looping for each element of the matrix 
	for (int row = 0; row < n; row++)
	{
		for (int col = 0; col < n; col++)
		{
			//  Copying into temporary matrix only those element 
			//  which are not in given row and column 
			if (row != p && col != q)
			{
				temp[i][j++] = A[row][col];

				// Row is filled, so increase row index and 
				// reset col index 
				if (j == n - 1)
				{
					j = 0;
					i++;
				}
			}
		}
	}
}

float determinant(float A[N][N], int n)
{
	float D = 0; // Initialize result 

	//  Base case : if matrix contains single element 
	if (n == 1)
		return A[0][0];

	float temp[N][N]; // To store cofactors 

	int sign = 1;  // To store sign multiplier 

	 // Iterate for each element of first row 
	for (int f = 0; f < n; f++)
	{
		// Getting Cofactor of A[0][f] 
		getCofactor(A, temp, 0, f, n);
		D += sign * A[0][f] * determinant(temp, n - 1);

		// terms are to be added with alternate sign 
		sign = -sign;
	}

	return D;
}

void adjoint(float A[N][N], float adj[N][N])
{
	if (N == 1)
	{
		adj[0][0] = 1;
		return;
	}

	// temp is used to store cofactors of A[][] 
	int sign = 1;
	float temp[N][N];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			// Get cofactor of A[i][j] 
			getCofactor(A, temp, i, j, N);

			// sign of adj[j][i] positive if sum of row 
			// and column indexes is even. 
			sign = ((i + j) % 2 == 0) ? 1 : -1;

			// Interchanging rows and columns to get the 
			// transpose of the cofactor matrix 
			adj[j][i] = (sign) * (determinant(temp, N - 1));
		}
	}
}

void displayFlat(float A[N * N])
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			cout << A[i * N + j] << " ";
		cout << endl;
	}
}

void displayVector(float A[N])
{
	cout.precision(17);
	for (int i = 0; i < N; i++)
	{
		cout << A[i] << fixed << endl;
	}
}

void serial_sum_temp(float* x_temp, float* result)
{
	// reinit results
	for (int j = 0; j < N; j++)
	{
		result[j] = 0;
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			result[i] += x_temp[i * N + j];
		}
	}
}


void display(float A[N][N])
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			cout << A[i][j] << " ";
		cout << endl;
	}
}
