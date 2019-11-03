
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm> 
#include <time.h>

#include "Inputs/A_10_2.h"
#include "Inputs/A_32_2.h"
#include "Inputs/A_512_2.h"
#include "Inputs/A_1024_2.h"

#include "Inputs/b_10.h"
#include "Inputs/b_32.h"
#include "Inputs/b_512.h"
#include "Inputs/b_1024.h"

// ------------------------------------------------
//#include <bits/stdc++.h> 

const int N = 10; 

using namespace std;
#include <iostream>

void getCofactor(double A[N][N], double temp[N][N], int p, int q, int n)
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

double determinant(double A[N][N], int n)
{
    double D = 0; // Initialize result 

    //  Base case : if matrix contains single element 
    if (n == 1)
        return A[0][0];

    double temp[N][N]; // To store cofactors 

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

void adjoint(double A[N][N], double adj[N][N])
{
    if (N == 1)
    {
        adj[0][0] = 1;
        return;
    }

    // temp is used to store cofactors of A[][] 
    int sign = 1;
    double temp[N][N];

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

// Function to calculate and store inverse, returns false if 
// matrix is singular 
bool inverse(double A[N][N], double* inverse)
{
    // Find determinant of A[][] 
    double det = determinant(A, N);
    if (det == 0)
    {
        cout << "Singular matrix, can't find its inverse";
        return false;
    }

    // Find adjoint 
    double adj[N][N];
    adjoint(A, adj);

    // Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            inverse[i*N +j] = adj[i][j] / double(det);

    return true;
}

void displayFlat(double A[N*N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << A[i * N + j] << " ";
        cout << endl;
    }
}

__global__ void cudadisplayFlat(double A[N * N])
{
	printf("\nThe inverse is: \n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("%f \n", A[i * N + j]);
	}
}

void displayVector(double A[N])
{
	cout.precision(17);
    for (int i = 0; i < N; i++)
    {
        cout<< A[i] << fixed << endl;
    }
}

void display(double A[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << A[i][j] << " ";
        cout << endl;
    }
}

// -------------------------------------------------------------------------------------

__global__ void gpu_process(double* x_temp, double* d_a, double* d_b, int num_threads, int num_blocks)
{
    int start, end;
    
    int thread_size=(N*N)/(num_blocks * num_threads);    
    if (thread_size == 0) thread_size=1;

    start = thread_size* (blockIdx.x * blockDim.x + threadIdx.x);
    end = start+ thread_size;

    for (int i = start; i < end; i++)
    {
		x_temp[i] = d_a[i] * d_b[i % N];
    }
}

__global__ void sum_temp(double* x_temp, double* result, int num_threads, int num_blocks)
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

    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < N; j++)
        {	
			// sum up rows 
            result[i] += x_temp[i*N + j];
        }
    }
}

 void serial_sum_temp(double* x_temp, double* result)
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

double run_process(int num_threads, double* d_a, double* d_b, double* x_temp, double* x) {
    int block_number = num_threads / 1024 + 1;
    int threads_per_block = num_threads / block_number;
    double time_spent = 0.0;

    clock_t begin = clock();
    gpu_process << <block_number, threads_per_block >> > (x_temp, d_a, d_b, threads_per_block, block_number);
    cudaDeviceSynchronize();

	//cout << "Temp: \n";
	//displayFlat(x_temp);

    sum_temp<<<block_number, threads_per_block >>>(x_temp, x, threads_per_block, block_number);
	//serial_sum_temp(x_temp, x);

    cudaDeviceSynchronize();
    clock_t end = clock();

    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    return time_spent;
}

void pre_process(double** x_temp, double** x, double** d_A, double** d_B, double* A, double* B) {
    unsigned error;
    
    // allocate and copy into device
    size_t matrixAsize = (size_t) (N * N * sizeof(double));
    size_t matrixBsize = (size_t) (N * sizeof(double));

    cudaMalloc((void**) & *d_A, matrixAsize);
    cudaMalloc((void**) & *d_B, matrixBsize);
    cudaMallocManaged((void**) & *x_temp, matrixAsize);

    cudaMemcpy(*d_A, A, matrixAsize, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_B, B, matrixBsize, cudaMemcpyHostToDevice);

    // allocate shared memory for x
    cudaMallocManaged(x, matrixBsize); 
}


int main()
{
    double inv_A[N*N]; 
    double* x_temp, * x, * d_A, * d_B;

    inverse(A_10, inv_A);
    pre_process(&x_temp, &x, &d_A, &d_B, inv_A, b_10);

	cout << "The input matrix is: \n";
	display(A_10);
	cout << "\nThe inverse is: \n";
	displayFlat(inv_A);
    
	int max_thread_power = 11;

	printf("\nMatrix Dimension: %d \n", N);
	for (int i = 0; i <= max_thread_power; i++) {
		int number_of_threads = pow(2, i);
		double duration = run_process(number_of_threads, d_A, d_B, x_temp, x);
		cout << "Number of threads: " << number_of_threads << "\t Run time: " << scientific << duration;
		cout << "\nX is: \n";
		displayVector(x);
	}
    
    return 0;
}

