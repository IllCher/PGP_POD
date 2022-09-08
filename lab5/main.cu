#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

#define HANDLE_ERROR(error) if (error != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(error)); exit(0);}

const int blocks = 1024;
const int threads = 256;

__device__ void get_swap(int indx1, int indx2, int* array) {
    int value1 = array[indx1];
    int value2 = array[indx2];
    if (value1 > value2) {
        array[indx2] = value1;
        array[indx1] = value2;
    }
}

__global__ void full_device_array(int n, int value, int* device_array) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;
    for(int i = idx; i < n; i += offset) {
        device_array[i] = value;    
    }
    
}

__global__ void get_sorted(int* device_array) {
    __shared__ int shared_mem_array[blocks + (blocks / threads) + 1];
    int tmp1 = ((threads + 1) * (threadIdx.x / threads) + (threadIdx.x % threads));
    shared_mem_array[((threads + 1) * (threadIdx.x / threads) + (threadIdx.x % threads))] = device_array[threadIdx.x + blockIdx.x * blocks];
    int tmp2 =  ((threads + 1) * ((threadIdx.x + blocks / 2) / threads) + ((threadIdx.x + blocks / 2) % threads));
    shared_mem_array[tmp2] = device_array[threadIdx.x + blockIdx.x * blocks + blocks / 2];

    if(threadIdx.x == 0)
        shared_mem_array[((threads + 1) * (blocks / threads) + (blocks % threads))] = INT_MAX;

    __syncthreads();
    
    for(int i = 0; i < blocks; i++) {
        get_swap(((threads + 1) * (2 * threadIdx.x / threads) + (2 * threadIdx.x % threads)), ((threads + 1) * ((2 * threadIdx.x + 1) / threads) + ((2 * threadIdx.x + 1) % threads)), shared_mem_array);  
        __syncthreads();
        get_swap(((threads + 1) * ((2 * threadIdx.x + 1) / threads) + ((2 * threadIdx.x + 1) % threads)), ((threads + 1) * ((2 * threadIdx.x + 2) / threads) + ((2 * threadIdx.x + 2) % threads)), shared_mem_array);
        __syncthreads();
    }
    __syncthreads();
    
    device_array[threadIdx.x + blockIdx.x * blocks] = shared_mem_array[tmp1];
    
    device_array[threadIdx.x + blockIdx.x * blocks + blocks / 2] = shared_mem_array[tmp2];
}

__global__ void get_merged(int* device_array) {
    __shared__ int shared_mem_array[blocks + (blocks / threads)];
    int tmp1 = ((threads + 1) * (threadIdx.x / threads) + (threadIdx.x % threads));
    shared_mem_array[tmp1] = device_array[threadIdx.x + blockIdx.x * blocks];
    int tmp2 = ((threads + 1) * ((blocks - threadIdx.x - 1) / threads) + ((blocks - threadIdx.x - 1) % threads));
    shared_mem_array[tmp2] = device_array[threadIdx.x + blockIdx.x * blocks + blocks / 2];
    __syncthreads();

    int cut = blocks / 2;
    do {
        int i = threadIdx.x / cut;
        int j = threadIdx.x % cut;
        __syncthreads();
        get_swap(((threads + 1) * ((2 * cut * i + j) / threads) + ((2 * cut * i + j) % threads)), ((threads + 1) * ((2 * cut * i + j + cut) / threads) + ((2 * cut * i + j + cut) % threads)), shared_mem_array);
        cut /= 2;
    } while (cut > 0);
    __syncthreads();
    
    device_array[threadIdx.x + blockIdx.x * blocks] = shared_mem_array[tmp1];
    int tmp3 = ((threads + 1) * ((threadIdx.x + blocks / 2) / threads) + ((threadIdx.x + blocks / 2) % threads));
    device_array[threadIdx.x + blockIdx.x * blocks + blocks / 2] = shared_mem_array[tmp3];
}


int main() {
    uint32_t n;
    fread(&n, sizeof(uint32_t) , 1, stdin);
    int* array = (int*)malloc(n * sizeof(int));
    fread(array, sizeof(int), n, stdin);

    if (n == 0) {
        return 0;
    }
    
    do {
        uint32_t devide;
        uint32_t tmp = n % blocks;
        if (tmp == 0) {
            devide = n;
        } else {
            devide = n + blocks - tmp;
        }
        uint32_t blocks_count = devide / blocks;
        int* device_array;
        
        int bias = devide - n;
        HANDLE_ERROR(cudaMalloc(&device_array, devide * sizeof(int) ));
        HANDLE_ERROR(cudaMemcpy(device_array, array, n * sizeof(int) , cudaMemcpyHostToDevice));

        full_device_array<<<32, blocks>>>(bias, INT_MIN, device_array + n);

        HANDLE_ERROR(cudaGetLastError());
        get_sorted<<<blocks_count, blocks / 2>>>(device_array);
        HANDLE_ERROR(cudaGetLastError());
        if (blocks_count == 1) {
            HANDLE_ERROR(cudaMemcpy(array, device_array + bias, n * sizeof(int) , cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaFree(device_array));
            break;
        }
        for (uint32_t i = 0; i < blocks_count; i++) {
            get_merged<<<blocks_count - 1, blocks / 2>>>(device_array + blocks / 2);
            HANDLE_ERROR(cudaGetLastError());
            get_merged<<<blocks_count, blocks / 2>>>(device_array);
            HANDLE_ERROR(cudaGetLastError());
        }
        HANDLE_ERROR(cudaMemcpy(array, device_array + bias, n * sizeof(int) , cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaFree(device_array));    
        
    } while(0);

    

    fwrite(array, sizeof(int) , n, stdout);

    free(array);
}