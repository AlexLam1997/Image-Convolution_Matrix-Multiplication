
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>

#include "lodepng.c"
#include <time.h>
#include <math.h> 
#include "wm.h"


const int Wsize = 3;

unsigned char clampJ(int result) {
    if (result < 0) {
        result = 0;
    }
    if (result > 255) {
        result = 255;
    }

    unsigned char new_result = (unsigned char)result;
    return new_result;

}

unsigned char convolveJ(int current_pixel, int channel, int width, int height, unsigned char* input, float wm[][Wsize], int dim_of_weight_matrix) { // should wm be something else 2d array???
    int i = current_pixel / width;
    int j = current_pixel % width;
    float result = 0;
    for (int ii = 0; ii < dim_of_weight_matrix; ii++) {
        for (int jj = 0; jj < dim_of_weight_matrix; jj++) {
            result = result + (input[(i + ii - 1) * 4 * width + 4 * (j + jj - 1) + channel]) * (wm[ii][jj]);
        }

    }
    int result1 = (int)result;
    return clampJ(result1);
}

int get_old_pixel(int x, int y, int new_width, int pixels_lost) {
    return ((y + 1) * (new_width + pixels_lost) + x + pixels_lost / 2);
}

__device__ int get_original_pixel(int x, int y, int new_width, int pixels_lost) {
    printf("Get Original \n");

    printf("x %d\n",x);
    printf("y %d\n", y);
    printf("new_width %d\n", new_width);
    printf("pixels_lost %d \n", pixels_lost);
    
    int pixel = ((y + 1) * (new_width + pixels_lost) + x + pixels_lost / 2);
    printf("pixel %d \n", pixel);

    return pixel;
}

void sequential_convolve(char* input_filename, char* output_filename, float wm[][Wsize], int dim_of_weight_matrix) {
    unsigned error;
    unsigned char* image, * new_image;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    new_image = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));

    int pixels_lost = (dim_of_weight_matrix / 2) * 2;

    width = width - pixels_lost;
    height = height - pixels_lost;
    int start = 0;
    int end = width * height;

    for (int i = start; i < end; i++) {
        int y = i / (width);
        int x = i % (width);
        int current_pixel_in_old_image = get_old_pixel(x, y, width, pixels_lost);
        new_image[4 * i + 0] = convolveJ(current_pixel_in_old_image, 0, width + pixels_lost, height + pixels_lost, image, wm, dim_of_weight_matrix);
        //image[4 * ((y + 1) * (width + 2) + x + 1) + 0];
        new_image[4 * i + 1] = convolveJ(current_pixel_in_old_image, 1, width + pixels_lost, height + pixels_lost, image, wm, dim_of_weight_matrix);
        //image[4 * ((y + 1) * (width + 2) + x + 1) + 1];//image[4 * i + 1];
        new_image[4 * i + 2] = convolveJ(current_pixel_in_old_image, 2, width + pixels_lost, height + pixels_lost, image, wm, dim_of_weight_matrix);
        //image[4 * ((y + 1) * (width + 2) + x + 1) + 2]; //!!!

        new_image[4 * i + 3] = (unsigned char)255;// set to max opacity instead of convolving them seems closer
            //convolveJ(current_pixel_in_old_image, 3, width + pixels_lost, height + pixels_lost, image, wm, dim_of_weight_matrix);	
        //image[4 * ((y + 1) * (width + 2) + x + 1) + 3];
    }

    lodepng_encode32_file(output_filename, new_image, width, height);

    free(image);
    free(new_image);
}

__device__ unsigned char clamp(int result) {
    printf("Clamp \n");
    if (result < 0) {
        result = 0;
    }
    if (result > 255) {
        result = 255;
    }

    unsigned char new_result = (unsigned char)result;
    return new_result;

}
__device__ unsigned char qpu_convolve(int current_pixel, int channel, int width, int height, unsigned char* input, float wm[], int dim_of_weight_matrix, size_t *pitch) { // should wm be something else 2d array???
    printf("Width: %d \n", width);
    printf("current_pixel: %d \n", current_pixel);
    printf("dim_of_weight_matrix: %d \n", dim_of_weight_matrix);

    printf("Weight: %f \n", (float*)(char*)wm);
    printf("Weight: %d \n", (int*)(char*)wm);


    int i = current_pixel / width;
    int j = current_pixel % width;
    float result = 0;
    for (int ii = 0; ii < dim_of_weight_matrix; ii++) {
        for (int jj = 0; jj < dim_of_weight_matrix; jj++) {
            printf("input: %f \n", (input[(i + ii - 1) * 4 * width + 4 * (j + jj - 1) + channel]));
            printf("Weight: %f", wm[ii* dim_of_weight_matrix+jj]);
            result = result + (input[(i + ii - 1) * 4 * width + 4 * (j + jj - 1) + channel]) * (wm[ii*dim_of_weight_matrix+jj]);

        }
    }
    int result1 = (int)result;
    printf("Convolve \n");

    return clamp(result1);
}

__global__ void threadProcess(int height, int width, unsigned char* new_image, unsigned char* image, int dim_of_weight_matrix, float wm[], int num_blocks, int num_threads, size_t* pitch) {
    int start;
    int end;
    int total_size = width * height;
    int thread_size = total_size / (num_threads * num_blocks);
    int pixels_lost= (dim_of_weight_matrix/2)*2;
    start = thread_size * (blockIdx.x * blockDim.x + threadIdx.x);
    end = start + thread_size;
    
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    printf("Block Dim: %d \n", blockDim);
    printf("Thread size: %d \n", thread_size);

    printf("Start: %d \n", start);
    printf("End: %d\n", end);

    printf("Weight: %f \n", &wm);
    for (int i = start; i < end; i++) {
        int y = i / (width);
        int x = i % (width);
        
        printf("X: %d \n", x);
        printf("Y: %d \n", y);
        int current_pixel_in_old_image = get_original_pixel(x, y, width, pixels_lost);
        new_image[4 * i + 0] = qpu_convolve(current_pixel_in_old_image, 0 ,width+pixels_lost,height+pixels_lost, image, wm, dim_of_weight_matrix, pitch);
           
        new_image[4 * i + 1] = qpu_convolve(current_pixel_in_old_image, 1, width + pixels_lost, height + pixels_lost, image, wm, dim_of_weight_matrix, pitch);
        
        new_image[4 * i + 2] = qpu_convolve(current_pixel_in_old_image, 2, width + pixels_lost, height + pixels_lost, image, wm, dim_of_weight_matrix, pitch);
          
        new_image[4 * i + 3] = (unsigned char)255; // full opacity
    }
}

void pre_thread_process(char* input_filename, char* output_filename, int number_threads, int dim_of_weight_matrix, float wm[]) {
    unsigned error;
    unsigned char* image, * new_image, * cuda_image, * cuda_new_image;
    unsigned width, height;
    float device_weights[Wsize* Wsize];

    int lost_pixels = (dim_of_weight_matrix / 2) * 2;//ie 3/2=1   1*2 = 2   5/2= 2  2* 2 = 4     7/2 = 3   3*2 = 6 
    //printf("%d \n", lost_pixels);
    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    new_image = (unsigned char*)malloc((width - lost_pixels) * (height - lost_pixels) * 4 * sizeof(unsigned char));

    cudaMalloc((void**)&cuda_image, width * height * 4 * sizeof(unsigned char));
    size_t* pitch = NULL;
    cudaMalloc((void**)& device_weights, Wsize*Wsize*sizeof(float));
    
    cudaMemcpy(cuda_image, image, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weights, wm, Wsize * Wsize * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)& cuda_new_image, (width - lost_pixels) * (height - lost_pixels) * 4 * sizeof(unsigned char));

    int block_number = number_threads / 1024 + 1;
    int threads_per_block = number_threads / block_number;

    double time_spent = 0.0;
    clock_t begin = clock();
    printf("%d \n", block_number);
    printf("%d \n", threads_per_block);
    threadProcess << < block_number, threads_per_block >> > (height - lost_pixels, width - lost_pixels, cuda_new_image, cuda_image, dim_of_weight_matrix, device_weights, block_number, threads_per_block, pitch);

    cudaMemcpy(new_image, cuda_new_image, (width - lost_pixels) * (height - lost_pixels) * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    lodepng_encode32_file(output_filename, new_image, width - lost_pixels, height - lost_pixels); //make the new image from the data 

    free(image);
    free(new_image);
    cudaFree(cuda_image);
    cudaFree(cuda_new_image);
    cudaFree(device_weights);
}

int main(int argc, char* argv[])
{
    //char* input_filename = argv[1];
    //char* output_filename = argv[2];
    //int thread_nums = atoi(argv[3]);
    
    int weight_size = sizeof(w) / sizeof(w[0][0]);
    
    
    float wm3[Wsize* Wsize];
    
    for (int i = 0; i < Wsize; i++) {
        for (int j = 0; j < Wsize; j++) {
            wm3[i*j] = w[i][j];
        }
    }

    //sequential_convolve("test.png", "output.png", wm3, 3);
    pre_thread_process("test.png", "output.png", 1, Wsize, wm3);

    //int i;
    //for (i = 0; i<=11; i++) {
        //int number_of_threads = pow(2,i);


        //double time_spent = 0.0;
        //clock_t begin = clock();

        //pre_thread_process(input_filename, output_filename, number_of_threads,Wsize, wm3);
        //clock_t end = clock();
        //time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        //printf("Number of threads: %d    Run time %f   \n", number_of_threads, time_spent);
    //}
    return 0;
}


