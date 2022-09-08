#include <iostream>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <vector>

#define HANDLE_ERROR(error) if (error != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(error)); exit(0);}

struct comparator {
        __host__ __device__ bool operator()(double a, double b) {
            return abs(a) < abs(b);
        }
};

double zero = 1e-7;

uint32_t n, m, k;

std::vector<double> host_matrix;

double* device_matrix;

std::vector<double> X;

std::vector<std::pair<uint32_t, uint32_t>> stairs;

__global__ void down(double* matrix, uint32_t n, uint32_t m, uint32_t row, uint32_t clm) {
    uint32_t offsetx = blockDim.x * gridDim.x;
    uint32_t offsety = blockDim.y * gridDim.y;
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;

    for (uint32_t j = clm + 1 + idy; j < m; j += offsety) {
        for (uint32_t i = row + 1 + idx; i < n; i += offsetx) {
            matrix[i + n * j] -= matrix[n * clm + i] / matrix[n * clm + row] * matrix[n * j + row];
        }
    }
}

__global__ void up(double* matrix, uint32_t n, uint32_t m, uint32_t k, uint32_t row, uint32_t clm) {
    uint32_t offsetx = blockDim.x * gridDim.x;
    uint32_t offsety = blockDim.y * gridDim.y;
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;

    for (uint32_t j = m + idy; j < m + k; j += offsety) {
        for (uint32_t i = idx; i < row; i += offsetx) {
            matrix[i + n * j] -= matrix[n * clm + i] / matrix[n * clm + row] * matrix[n * j + row];
        }
    }    
}

__global__ void swap(double* matrix, uint32_t n, uint32_t m, uint32_t clm, uint32_t left, uint32_t right) {
    uint32_t offsetx = blockDim.x * gridDim.x;
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (uint32_t i = idx + clm; i < m; i += offsetx) {
        double tmp = matrix[n * i + left];
        matrix[n * i + left] = matrix[n * i + right];
        matrix[n * i + right] = tmp;
    }
}

int main() {

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> n >> m >> k;

    host_matrix.resize(n * m + n * k);
    
    X.resize(m * k);

    for (uint32_t i = 0; i < X.size(); i++) {
        X[i] = 0;
    }

    double tmp;

    for (uint32_t i = 0; i < n; i++){
        for (uint32_t j = 0; j < m; j++) {
            std::cin >> tmp;
            host_matrix[i + n * j] = tmp;
        }
    }

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < k; j++) {
            std::cin >> tmp;
            host_matrix[i + n * (j + m)] = tmp;
        }
    }

    HANDLE_ERROR(cudaMalloc(&device_matrix, host_matrix.size() * sizeof(double) ));
    HANDLE_ERROR(cudaMemcpy(device_matrix, host_matrix.data(), host_matrix.size() * sizeof(double), cudaMemcpyHostToDevice));

    //forward

    thrust::device_ptr<double> p_matrix = thrust::device_pointer_cast(device_matrix);

    comparator cmp;
    
    uint32_t i = 0, j = 0;
    while (i < n && j < m) {
        thrust::device_ptr<double> p_max = thrust::max_element(p_matrix + i + n * j, p_matrix + n * j + n, cmp);
        
        double maximum = -11;

        HANDLE_ERROR(cudaMemcpy(&maximum, thrust::raw_pointer_cast(p_max), sizeof(double), cudaMemcpyDeviceToHost));


        if (abs(maximum) < zero) {
            j++;
            continue;
        }

        std::pair<uint32_t, uint32_t> tmp;
        tmp.first = i;
        tmp.second = j;
        stairs.push_back(tmp);

        if (i >= n - 1) {
            break;
        }

        uint32_t lead = thrust::distance(p_matrix + i + n * j, p_max) + i;
        
        if (lead != i) {
            swap<<<256, 1024>>>(device_matrix, n, m + k, j, i, lead);            
        }

        down<<<dim3(128, 128), dim3(32, 32)>>>(device_matrix, n, m + k, i, j);
        
        i += 1;
        j += 1;
    }
    
    //forward


    //back

    for (uint32_t i = stairs.size() - 1; i + 1 > 0; i--) {

        up<<<dim3(128, 128), dim3(32, 32)>>>(device_matrix, n, m, k, stairs[i].first, stairs[i].second);

    }

    HANDLE_ERROR(cudaMemcpy(host_matrix.data(), device_matrix, host_matrix.size() * sizeof(double), cudaMemcpyDeviceToHost));
    

    for (uint32_t t = 0; t < k; t++) {
        for (uint32_t i = stairs.size() - 1; i + 1 > 0; i--) {
            X[stairs[i].second + m * t] = host_matrix[stairs[i].first + n * (m + t)] / host_matrix[stairs[i].first + n * stairs[i].second];
        }
    }
        
    //back

    std::cout.precision(10);
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < k; j++) {
            std::cout << std::scientific;
            std::cout << X[i + m * j] << " ";
        }
        std::cout << std::endl;
    }
    
    HANDLE_ERROR(cudaFree(device_matrix));

    return 0;

}