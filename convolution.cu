#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>

#define BLOCK_SIZE 2
#define WA 8
#define HA 8
#define HC 3
#define WC 3
#define WB (WA - WC + 1)
#define HB (HA - HC + 1)


__global__ void Convolution_GPU(float* A, float* B, float* C, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// GPU Convolution code
	float sum = 0.0;
	if(row < numBRows && col < numBCols){
		for(int i=0; i < numCRows; i++){
			for (int j=0; j< numCCols; j++){
				sum += A[(row+i) * numARows + (col+j) ] * C[i * numCRows + j];
			}
		}
		B[row * numBRows + col] = sum;
	}
	//

}

void Convolution_CPU(float* A, float* B, float* C, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols)
{
	// CPU Convolution code
	for(int i=0; i<numBRows; i++){
		for(int j=0; j<numBCols; j++){
			//one cell
			//now one cell calculate
			float tmp = 0.0;
			for(int k_i = 0; k_i < numCRows; k_i++){
				for(int k_j = 0; k_j < numCCols; k_j++){
					// kernel conv
					tmp += A[(i+k_i) * numARows + (j + k_j)] * C[k_i * numCRows + k_j];
				}
			}
			B[i * numBRows + j] = tmp;
		}
	}
	printf("\n");printf("\n");printf("\n");
	//////////////////////////////
}


void randomInit(float* data, int size)
{
	// random initialization code
	for (int i=0; i<size; ++i){
		for (int j=0; j< size; ++j) {
			data[i * size + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}
	//////////////////////////////
}


int main(int argc, char** argv)
{
	srand(2006);
	cudaError_t error;
	cudaEvent_t start_G, stop_G, start_C, stop_C;

	cudaEventCreate(&start_G);
	cudaEventCreate(&stop_G);
	cudaEventCreate(&start_C);
	cudaEventCreate(&stop_C);

	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*)malloc(mem_size_A);

	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*)malloc(mem_size_B);

	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* h_C = (float*)malloc(mem_size_C);

	// random initialization
	randomInit(h_A, HA);
	randomInit(h_C, HC);
	//////////////////////////////


	// cudaMalloc
	float *d_A, *d_B, *d_C;
	cudaMalloc((void**) &d_A, mem_size_A);
	cudaMalloc((void**) &d_B, mem_size_B);
	cudaMalloc((void**) &d_C, mem_size_C);

	//////////////////////////////

	// cudaMemcpy
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);
	//////////////////////////////

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((WA) / (BLOCK_SIZE), (WA) / (BLOCK_SIZE));

	cudaEventRecord(start_G);

	// GPU Convolution function call
	Convolution_GPU<<<grid, threads>>>(d_A, d_B, d_C, HA, WA, HB, WB, HC, WC);

	//////////////////////////////
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in launching kernel\n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}

	error = cudaDeviceSynchronize();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaDeviceSynchronize \n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}

	cudaEventRecord(stop_G);
	cudaEventSynchronize(stop_G);

	// cudaMemcpy - results
	cudaMemcpy(h_B, d_B, mem_size_B, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//////////////////////////////

	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start_G, stop_G);

	printf("Time took to compute matrix A of dimensions %d x %d  on GPU is %f ms \n \n \n", WA, HA, miliseconds);

	printf("=======input=========\n");
	for (int i = 0;i < HA;i++)
	{
		for (int j = 0;j < WA;j++)
		{
			printf("%f ", h_A[i*HA + j]);
		}
		printf("\n");
	}
	printf("\n\n=======kernel=========\n");
	for (int i = 0;i < HC;i++)
	{
		for (int j = 0;j < WC;j++)
		{
			printf("%f ", h_C[i*HC + j]);
		}
		printf("\n");
	}
	printf("\n\n=======GPU results=========\n");
	for (int i = 0;i < HB;i++)
	{
		for (int j = 0;j < WB;j++)
		{
			printf("%f ", h_B[i*HB + j]);
		}
		printf("\n");
	}

	cudaEventRecord(start_C);
	// CPU Convolution function call
	Convolution_CPU(h_A, h_B, h_C, HA, WA, HB, WB, HC, WC);
	//////////////////////////////

	cudaEventRecord(stop_C);
	cudaEventSynchronize(stop_C);

	cudaEventElapsedTime(&miliseconds, start_C, stop_C);
	printf("Time took to compute matrix A of dimensions %d x %d  on CPU is %f ms \n \n \n", WA, HA, miliseconds);

	printf("\n\n=======CPU results=========\n");
	for (int i = 0;i < HB;i++)
	{
		for (int j = 0;j < WB;j++)
		{
			printf("%f ", h_B[i*HB + j]);
		}
		printf("\n");
	}

	// memory release
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	//////////////////////////////

	return EXIT_SUCCESS;
}
