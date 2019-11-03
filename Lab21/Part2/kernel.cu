
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
using namespace std;
#define N 4 
#include <iostream>

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

int determinant(float A[N][N], int n)
{
    int D = 0; // Initialize result 

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

// Function to calculate and store inverse, returns false if 
// matrix is singular 
bool inverse(float A[N][N], float inverse[N*N])
{
    // Find determinant of A[][] 
    int det = determinant(A, N);
    if (det == 0)
    {
        cout << "Singular matrix, can't find its inverse";
        return false;
    }

    // Find adjoint 
    float adj[N][N];
    adjoint(A, adj);

    // Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            inverse[i*N +j] = adj[i][j] / float(det);

    return true;
}

void displayFlat(float A[N*N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << A[i * N + j] << " ";
        cout << endl;
    }
}

void display(float A[N][N])
{
    for (int i = 0; i < N*N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << A[i][j] << " ";
        cout << endl;
    }
}

// -------------------------------------------------------------------------------------

__global__ void gpu_process(unsigned char* input_image, unsigned char* output_image, unsigned width, unsigned height, int num_threads, int num_blocks)
{

}

double run_process(int num_threads, int width, int height, unsigned char* new_image, char* output_filename, unsigned char* d_image) {
    int block_number = num_threads / 1024 + 1;
    int threads_per_block = num_threads / block_number;

    double time_spent = 0.0;
    clock_t begin = clock();
    gpu_process << <block_number, threads_per_block >> > (d_image, new_image, width, height, threads_per_block, block_number);
    cudaDeviceSynchronize();
    //lodepng_encode32_file(output_filename, new_image, width / 2, height / 2);
    clock_t end = clock();

    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    return time_spent;
}

void pre_process(float** x, float** d_A, float** d_B, float* A, float* B) {
    unsigned error;
    
    // allocate and copy into device
    size_t matrixAsize = (size_t)((N) * (N) * 4 * sizeof(float));
    size_t matrixBsize = (size_t) (N * sizeof(float));

    cudaMalloc((void**) & *d_A, matrixAsize);
    cudaMalloc((void**) & *d_B, matrixBsize);

    cudaMemcpy(*d_A, A, matrixAsize, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_B, B, matrixBsize, cudaMemcpyHostToDevice);

    // allocate shared memory for x
    cudaMallocManaged(x, matrixAsize);
}

int main()
{
    float A[N][N] = { {5, -2, 2, 7},
                        {1, 0, 0, 3},
                        {-3, 1, 5, 0},
                        {3, -1, -9, 4} };

    float adj[N][N];  // To store adjoint of A[][] 

    float inv_A[N*N]; // To store inverse of A[][] 

    cout << "Input matrix is :\n";
    display(A);

    cout << "\nThe Adjoint is :\n";
    adjoint(A, adj);
    display(adj);

    cout << "\nThe Inverse is :\n";
    if (inverse(A, inv_A))
        displayFlat(inv_A);

    
    float* x, *d_A, *d_B; 
    
    pre_process(&x, &d_A, &d_B, inv_A, b_10);
    


    return 0;
}

