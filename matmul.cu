#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

// выполняется на GPU
__global__
void matmulDevice(int* A, int* B, int* C, int N)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < N && col < N) {
		int sum = 0;
		for (int i = 0; i < N; i++)
			sum += A[row * N + i] * B[i * N + col];
		C[row * N + col] = sum;
	}
}

void matmulHost(int* A, int* B, int* C, int N)
{

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int sum = 0;
			for (int k = 0; k < N; k++)
				sum += A[i * N + k] * B[k * N + j];
			C[i * N + j] = sum;
		}
	}
}

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

using namespace std;

int main(void)
{
	int device_count = 0;
	    cudaGetDeviceCount(&device_count);
	
		if (device_count == 0)
			cout << "Sorry! You dont have CudaDevice" << endl;
		else
			cout << "CudaDevice found! Device count: " << device_count << endl;
	
		int N = 2048;
		int block_size = 16;
		// Кол-во итераций
		int nIter = 1;

		unsigned int count = N*N;
		unsigned int mem_size = sizeof(int) * count;
	
	
		int* A = (int*)malloc(mem_size);
		int* B = (int*)malloc(mem_size);
		int* h_C = (int*)malloc(mem_size);
		int* hCuda_C = (int*)malloc(mem_size);
	
		int* d_A, * d_B, * d_C;
	
		for (int i = 0; i < count; i++) {
			A[i] = rand() % 100 + 1;
			B[i] = rand() % 100 + 1;
		}
	
		unsigned int start_time = clock();

		for (int j = 0; j < nIter; j++) {
			matmulHost(A, B, h_C, N);
		}

		unsigned int elapsedTime = clock() - start_time;
		float msecPerMatrixMulCpu = elapsedTime / nIter;

		cout << "CPU time: " << msecPerMatrixMulCpu << endl;
	
		checkCudaErrors(cudaMalloc((void**)& d_A, mem_size));
		checkCudaErrors(cudaMalloc((void**)& d_B, mem_size));
		checkCudaErrors(cudaMalloc((void**)& d_C, mem_size));
	
		// копируем данные на девайс
		checkCudaErrors(cudaMemcpy(d_A, A, mem_size,
			cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_B, B, mem_size,
			cudaMemcpyHostToDevice));
	
		dim3 threadsPerBlock(block_size, block_size);
		dim3 blocksPerGrid(N / block_size, N / block_size);
		
		cudaEvent_t start;
		cudaEvent_t stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		// Записываем начало события
		checkCudaErrors(cudaEventRecord(start, 0));

		for (int j = 0; j < nIter; j++) {
			matmulDevice << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
		}

		// Записываем конец события
		checkCudaErrors(cudaEventRecord(stop, 0));

		// Ждем конца события
		checkCudaErrors(cudaEventSynchronize(stop));

		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		float msecPerMatrixMul = msecTotal / nIter;
			   
		cout << "GPU time: " << msecPerMatrixMul << endl;

		cudaDeviceSynchronize();
	
		// копируем результат с девайса
		checkCudaErrors(cudaMemcpy(hCuda_C, d_C, mem_size, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
	
		// free device memory
		cudaFree(d_A);
	    cudaFree(d_B);
	    cudaFree(d_C);
	
		bool test = true;
	
		for (int i = 0; i < count; i++) {
			if (h_C[i] != hCuda_C[i])
				test = false;
		}

		if (test)
			cout << "PASS!" << endl;
		else 
			cout << "WASTED!" << endl;
	
    return 0;
}

