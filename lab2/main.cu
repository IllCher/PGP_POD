#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define HANDLE_ERROR(error) if (error != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(error)); exit(0);}

#define bright(r) (0.299 * float(r.x) + 0.587 * float(r.y) + 0.114 * float(r.z))

texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ int4 sum(uchar4 a, uchar4 b, uchar4 c) {
    int4 result;
    result.x = a.x + b.x + c.x;
    result.y = a.y + b.y + c.y;
    result.z = a.z + b.z + c.z;
    result.w = 0;
    return result;
}

__device__ int4 dif(int4 a, int4 b) {
    int4 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    result.w = 0;
    return result;
}

__device__ float pw(uchar4* z) {
    int4 first_str = sum(z[0], z[1], z[2]);
    int4 last_str = sum(z[6], z[7], z[8]);

    int4 first_col = sum(z[0], z[3], z[6]);
    int4 last_col = sum(z[2], z[5], z[8]);

    int4 g_x = dif(last_str, first_str);
    int4 g_y = dif(last_col, first_col);

    return sqrtf(bright(g_x) * bright(g_x) + bright(g_y) * bright(g_y)); 

}

__global__ void kernel(uchar4* ans, uint32_t w, uint32_t h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    uchar4 z[9];
    int first_col, last_col, first_str, last_str;

    for (int x = idx; x < w; x += offsetx) {
        for (int y = idy; y < h; y += offsety) {
            
            first_str = y - 1;
            last_str = y + 1;
            first_col = x - 1;
            last_col = x + 1;

            z[0] = tex2D(tex, first_col, first_str);
            z[1] = tex2D(tex, x, first_str);
            z[2] = tex2D(tex, last_col, first_str);
            z[3] = tex2D(tex, first_col, y);
            z[4] = tex2D(tex, x, y);
            z[5] = tex2D(tex, last_col, y);
            z[6] = tex2D(tex, first_col, last_str);
            z[7] = tex2D(tex, x, last_str);
            z[8] = tex2D(tex, last_col, last_str);

            float tmp = pw(z);
            unsigned char result = tmp;
            if (tmp >= 255) {
                result = 255;
            }
            ans[x + y * w] = make_uchar4(result, result, result, 0);
        }
    }
}

int main() {
    char input[256], output[256];

    scanf("%s", input);
    scanf("%s", output);

    FILE* in = fopen(input, "rb");

    uint32_t w, h;
    
    fread(&w, sizeof(uint32_t), 1, in);
    fread(&h, sizeof(uint32_t), 1, in);
    uint32_t* data = (uint32_t *)malloc(w * h * sizeof(uint32_t));
    fread(data, sizeof(uint32_t), w * h, in);


    fclose(in);

    cudaArray* arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();

    HANDLE_ERROR(cudaMallocArray(&arr, &ch, w, h));
    HANDLE_ERROR(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice));

    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.channelDesc = ch;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;

    HANDLE_ERROR(cudaBindTextureToArray(tex, arr, ch));
    uchar4* device_data;
    HANDLE_ERROR(cudaMalloc(&device_data, sizeof(uchar4) * h * w));


    kernel<<<dim3(1, 1), dim3(8, 8)>>>(device_data, w, h);
    HANDLE_ERROR(cudaGetLastError());

    HANDLE_ERROR(cudaMemcpy(data, device_data, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost));

    FILE* out = fopen(output, "wb");
    
    fwrite(&w, sizeof(uint32_t), 1, out);
    fwrite(&h, sizeof(uint32_t), 1, out);
    fwrite(data, sizeof(uint32_t), w * h, out);

    fclose(out);

    HANDLE_ERROR(cudaUnbindTexture(tex));
    HANDLE_ERROR(cudaFreeArray(arr));
    HANDLE_ERROR(cudaFree(device_data));
    free(data);
	
	return 0;
}
