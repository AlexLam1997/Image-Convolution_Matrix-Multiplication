
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include ".\lodepng.h"
#include "wm.h"

const int Wsize = 7;

__device__ unsigned char clamp(float result) {
    if (result < 0) {
        result = 0;
    }
    if (result > 255) {
        result = 255;
    }

    unsigned char new_result = (unsigned char)result;
    return new_result;
}

__global__ void threadProcess(int new_height, int new_width, unsigned char* new_image, unsigned char* old_image, float* weights, int numBlocks, int numThreads)
{
    int start, end;
    int total_size = new_width * new_height;
    int thread_size = total_size / (numThreads * numBlocks);
    int pixels_lost = (Wsize / 2) * 2;

    start = thread_size * (blockIdx.x * blockDim.x + threadIdx.x);
    end = start + thread_size;

    //printf("Start: %d \t End: %d \n", start, end);

    for (int i = start; i < end; i++)
    {
        float r, g, b; 
        r = 0; 
        g = 0;
        b = 0;

        int x_new = i % new_width;
        int y_new = i / new_width;

        //printf("X: %d \t Y: %d \n", x_new, y_new);
        
        int old_coord = y_new*(new_width + pixels_lost) + x_new;
        printf("old_coord: %d \n", old_coord);

        // height
        for (int wY = 0; wY < Wsize; wY++)
        {
            // width
            for (int wX = 0; wX < Wsize; wX++)
            {
                float weight = weights[wX + wY * Wsize];
                int pixel_index = old_coord + wY*(new_width + pixels_lost) + wX; 
                // printf("Pixel index: %d \n", pixel_index);
                r += old_image[4 * pixel_index] * weight;
                g += old_image[4 * pixel_index + 1] * weight;
                b += old_image[4 * pixel_index + 2] * weight;
            }
        }

        new_image[4*i] = clamp(r);
        new_image[4*i+1] = clamp(g);
        new_image[4*i+2] = clamp(b);
        new_image[4*i+3] = old_image[4*(old_coord + Wsize/2 + (Wsize/2)*(new_width + pixels_lost)) + 3];
    }
}

void pre_thread_process(char* input_filename, char* output_filename, int number_threads, float* wm) 
{
    unsigned error;
    unsigned char* image, * new_image, * cuda_image, * cuda_new_image;
    unsigned width, height;
    float* device_weights;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    //ie 3/2=1   1*2 = 2   5/2= 2  2* 2 = 4     7/2 = 3   3*2 = 6 
    int lost_pixels = (Wsize / 2) * 2;
    int new_width = width - lost_pixels;
    int new_height = height - lost_pixels;

    new_image = (unsigned char*)malloc(new_width* new_height * 4 * sizeof(unsigned char));

    cudaMalloc((void**)& cuda_image, width * height * 4 * sizeof(unsigned char));
    cudaMalloc((void**)& device_weights, Wsize * Wsize * sizeof(float));

    cudaMemcpy(cuda_image, image, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weights, wm, Wsize * Wsize * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)& cuda_new_image, new_width * new_height * 4 * sizeof(unsigned char));

    int block_number = number_threads / 1024 + 1;
    int threads_per_block = number_threads / block_number;

    threadProcess << < block_number, threads_per_block >> > (new_height, new_width, cuda_new_image, cuda_image, device_weights, block_number, threads_per_block);
    cudaDeviceSynchronize();
    cudaMemcpy(new_image, cuda_new_image, (new_width) * (new_height) * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    lodepng_encode32_file(output_filename, new_image, new_width, new_height); //make the new image from the data 

    free(image);
    free(new_image);
    cudaFree(cuda_image);
    cudaFree(cuda_new_image);
    cudaFree(device_weights);
}


int main(int argc, char* argv[])
{
    char* input_filename = argv[1];
    char* output_filename = argv[2];

    float* wm = (float*)malloc(Wsize * Wsize * sizeof(float));

    // Flattening
    for (int i = 0; i < Wsize; i++) {
        for (int j = 0; j < Wsize; j++) {
            // change argument here for different weight matrices
            wm[i * Wsize + j] = w7[i][j];
            printf("%f \n", wm[i * Wsize + j]);
        }
    }

    //sequential_convolve("test.png", "output.png", wm3, 3);
    pre_thread_process(input_filename, output_filename, 128, wm);

    return 0;
}
