#include <iostream>
#include <iomanip>

#define HANDLE_ERROR(error) if (error != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(error)); exit(0);}

__global__ void sum_vectors(double* vector1, double* vector2, double* sum, int n) {
    int i, idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    for(i = idx; i < n; i += offset)
        sum[i] = vector1[i] + vector2[i];
}

int main() {
    int n = 0;
    std::cin >> n;

    double* vector1 = new double[n];
    double* vector2 = new double[n];
    double* result = new double[n];
    for (int i = 0; i < n; i++) {
        std::cin >> vector1[i];
    }
    for (int j = 0; j < n; j++) {
        std::cin >> vector2[j];
    }
    double* dest1;
    double* dest2;
    double* dest_result;

    HANDLE_ERROR(cudaMalloc(&dest1, sizeof(double) * n));
    HANDLE_ERROR(cudaMalloc(&dest2, sizeof(double) * n));
    HANDLE_ERROR(cudaMalloc(&dest_result, sizeof(double) * n));

    HANDLE_ERROR(cudaMemcpy(dest1, vector1, sizeof(double) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dest2, vector2, sizeof(double) * n, cudaMemcpyHostToDevice));

    sum_vectors <<< 256, 256 >>> (dest1, dest2, dest_result, n);

    HANDLE_ERROR(cudaGetLastError());

    HANDLE_ERROR(cudaMemcpy(result, dest_result, sizeof(double) * n, cudaMemcpyDeviceToHost));

    std::cout.precision(10);

    std::cout.setf(std::ios::scientific);
    for (int i = 0; i < n; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << "\n";

    HANDLE_ERROR(cudaFree(dest1));
    HANDLE_ERROR(cudaFree(dest2));
    HANDLE_ERROR(cudaFree(dest_result));

    delete[] vector1;
    delete[] vector2;
    delete[] result;
}