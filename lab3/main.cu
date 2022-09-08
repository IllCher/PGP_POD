#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define ACCURACY 1e-2


#define HANDLE_ERROR(error) if (error != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(error)); exit(0);}

typedef struct {
    int x, y;
} middle;


__device__ __constant__ float4 device_mids[500];


__host__ float find_norm(float4 a, float4 b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
}


__device__ float find_dist(const uchar4& a, const float4& b) {
    float x = b.x - float(a.x);
    float y = b.y - float(a.y);
    float z = b.z - float(a.z);
    return x * x + y * y + z * z;
}



__device__ int find_best_distance(const uchar4& point, const int clasters_cnt) {
    int match = 0;
    float closest = 1e14;

    for (int i = 0; i < clasters_cnt; i++) {
        float curr_dist = find_dist(point, device_mids[i]);
        if (curr_dist < closest) {
            match = i;
            closest = curr_dist;
        }
    }

    return match;
}


__global__ void kernel(uchar4* data, size_t n, ulonglong4* cache, uint32_t clasters_cnt) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;

    for (size_t i = id; i < n; i += offset) {
        uchar4& element = data[i];
        element.w = find_best_distance(element, clasters_cnt);

        ulonglong4* tmp_element = &cache[element.w];

        atomicAdd(&tmp_element->x, element.x);
        atomicAdd(&tmp_element->y, element.y);
        atomicAdd(&tmp_element->z, element.z);
        atomicAdd(&tmp_element->w, 1);

    }
}



int main() {

    char input[256], output[256];

    scanf("%s", input);
    scanf("%s", output);


    uint32_t clasters_cnt;
    middle* mids;
    scanf("%d", &clasters_cnt);
    mids = (middle*)malloc(clasters_cnt * sizeof(middle));
    for (uint32_t i = 0; i < clasters_cnt; i++) {
        scanf("%d", &mids[i].x);
        scanf("%d", &mids[i].y);
    }



    FILE* in = fopen(input, "rb");

    uint32_t w, h;
    
    fread(&w, sizeof(uint32_t), 1, in);
    fread(&h, sizeof(uint32_t), 1, in);
    uchar4* data = (uchar4 *)malloc(w * h * sizeof(uchar4));
    fread(data, sizeof(uchar4), w * h, in);


    fclose(in);
    
    size_t n = h * w;

    uchar4* device_data;

    float4* device_next_mids;

    ulonglong4* device_cache;

    float4 host_mids[500];

    float4* host_next_mids = (float4*)malloc(sizeof(float4) * clasters_cnt);
        
    ulonglong4* host_cache = (ulonglong4*)malloc(sizeof(ulonglong4) * clasters_cnt);



    HANDLE_ERROR(cudaMalloc(&device_data, sizeof(uchar4) * n));
    HANDLE_ERROR(cudaMemcpy(device_data, data, sizeof(uchar4) * n, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc(&device_next_mids, sizeof(float4) * clasters_cnt));
    HANDLE_ERROR(cudaMalloc(&device_cache, sizeof(ulonglong4) * clasters_cnt));

    float4* tmp_mids;

    tmp_mids = (float4*)malloc(sizeof(float4) * clasters_cnt);
    for (uint32_t i = 0; i < clasters_cnt; i++) {
        uchar4 element = data[(size_t)mids[i].y * w + (size_t)mids[i].x];
        tmp_mids[i] = make_float4(element.x, element.y, element.z, 0.0f);
    }
    
    HANDLE_ERROR(cudaMemcpy(device_next_mids, tmp_mids, sizeof(float4) * clasters_cnt, cudaMemcpyHostToDevice));
    free(tmp_mids);
    

    unsigned long long same = 1;


    while (same != 0) {

        same = 0;
        HANDLE_ERROR(cudaMemcpyToSymbol(device_mids, device_next_mids, sizeof(float4) * clasters_cnt, 0, cudaMemcpyDeviceToDevice));

        HANDLE_ERROR(cudaMemset(device_cache, 0, sizeof(ulonglong4) * clasters_cnt));

        kernel<<<dim3(256, 1, 1), dim3(256, 1, 1)>>>(device_data, n, device_cache, clasters_cnt);
        
        HANDLE_ERROR(cudaGetLastError());
        
        
        HANDLE_ERROR(cudaMemcpy(host_cache, device_cache, sizeof(ulonglong4) * clasters_cnt, cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaMemcpyFromSymbol(host_mids, device_mids, sizeof(float4) * 500, 0, cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaMemcpy(host_next_mids, device_next_mids, sizeof(float4) * clasters_cnt, cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < clasters_cnt; i++) {

            ulonglong4 tmp_element = host_cache[i];

            float4 element = make_float4(float(tmp_element.x) / float(tmp_element.w), float(tmp_element.y) / float(tmp_element.w), float(tmp_element.z) / float(tmp_element.w), 0.0f);


            //device_next_mids[i] = element; // заменить
            host_next_mids[i] = element;

            //float4 old_mid = device_mids[i]; // заменить

            float4 old_mid = host_mids[i];

            float difference = find_norm(old_mid, element);
            
            if (difference > ACCURACY) {
                same += 1;
            }


        }

        HANDLE_ERROR(cudaMemcpy(device_next_mids, host_next_mids, sizeof(float4) * clasters_cnt, cudaMemcpyHostToDevice));


        HANDLE_ERROR(cudaGetLastError());

    }

    free(host_next_mids);
    free(host_cache);

    HANDLE_ERROR(cudaFree(device_cache));
    HANDLE_ERROR(cudaMemcpy(data, device_data, sizeof(uchar4) * n, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(device_next_mids));
    HANDLE_ERROR(cudaFree(device_data));



    FILE* out = fopen(output, "wb");
    
    fwrite(&w, sizeof(uint32_t), 1, out);
    fwrite(&h, sizeof(uint32_t), 1, out);
    fwrite(data, sizeof(uint32_t), w * h, out);


    fclose(out);


    free(data);
    free(mids);
	
	return 0;
}