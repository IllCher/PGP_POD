#include <iostream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>


#define _i(i, j, k) i + 1 + (j + 1) * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2)

#define HANDLE_ERROR(error) if (error != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(error)); exit(0);}

__global__ void kernel(double* next_data, double* data, int n_x, int n_y, int n_z, double h_x, double h_y, double h_z) {
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int id_z = blockIdx.z * blockDim.z + threadIdx.z;
    int offset_y = blockDim.y * gridDim.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_z = blockDim.z * gridDim.z;
   
    for (int i = id_x; i < n_x; i += offset_x) {
        for (int j = id_y; j < n_y; j += offset_y) {
            for (int k = id_z; k < n_z; k += offset_z) {
                double h_x_squared = h_x * h_x;
                double h_y_squared = h_y * h_y;
                double h_z_squared = h_z * h_z;

                double add1 = (data[i + 2 + (j + 1) * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2)] + data[i + (j + 1) * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2)]) / h_x_squared;
                double add2 = (data[i + 1 + (j + 2) * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2)] + data[i + 1 + j * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2)]) / h_y_squared; 
                double add3 = (data[i + 1 + (j + 1) * (n_x + 2) + (k + 2) * (n_x + 2) * (n_y + 2)] + data[i + 1 + (j + 1) * (n_x + 2) + k * (n_x + 2) * (n_y + 2)]) / h_z_squared;
                double devider = 2 * (1.0 / h_x_squared + 1.0 / h_y_squared + 1.0 / h_z_squared);

                int tmp = i + 1 + (j + 1) * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2);
                next_data[tmp] = (add1 + add2 + add3) / devider;
            }
        }
            
    }
        
}

__global__ void get_error(double* next_data, double* data, int n_x, int n_y, int n_z) {
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int id_z = blockIdx.z * blockDim.z + threadIdx.z;
    int offset_y = blockDim.y * gridDim.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_z = blockDim.z * gridDim.z;

    for (int i = id_x - 1; i <= n_x; i += offset_x) {
        for (int j = id_y - 1; j <= n_y; j += offset_y) {
            for (int k = id_z - 1; k <= n_z; k += offset_z) {
				 if (i == -1 || j == -1 || k == -1 || i == n_x || j == n_y || k == n_z) {
                    data[_i(i, j, k)] = 0;
                } else {
					data[_i(i, j, k)] = fabs(next_data[_i(i, j, k)] - data[_i(i, j, k)]);
                }
            }
        }      
    }    
}

__global__ void get_copy(double* edge, double* data, int n_x, int n_y, int n_z, int index, int device_to_host, double u, int type) {
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_y = blockDim.y * gridDim.y;
    int offset_x = blockDim.x * gridDim.x;

    if (type == 0) {
        if (device_to_host) {
            for (int k = id_y; k < n_z; k += offset_y) {
                for (int j = id_x; j < n_y; j += offset_x)
                    edge[j + n_y * k] = data[_i(index, j, k)];
            }  
        } else  {
            if (edge) {
                for (int k = id_y; k < n_z; k += offset_y) {
                    for (int j = id_x; j < n_y; j += offset_x) {
                        data[_i(index, j, k)] = edge[j + n_y * k];
                    }  
                }     
            } else {
                for (int k = id_y; k < n_z; k += offset_y) {
                    for (int j = id_x; j < n_y; j += offset_x) {
                        data[_i(index, j, k)] = u;
                    }
                    
                }    
            }
        }
    } else if(type == 1) {
        if (device_to_host) {
            for (int k = id_y; k < n_z; k += offset_y) {
                for (int i = id_x; i < n_x; i += offset_x) {
                    edge[i + n_x * k] = data[_i(i, index, k)];
                }  
            }    
        } else {
            if (edge) {
                for (int k = id_y; k < n_z; k += offset_y) {
                    for (int i = id_x; i < n_x; i += offset_x) {
                        data[_i(i, index, k)] = edge[i + n_x * k];
                    }     
                }     
            } else {
                for (int k = id_y; k < n_z; k += offset_y) {
                    for (int i = id_x; i < n_x; i += offset_x) {
                        data[_i(i, index, k)] = u;
                    }  
                }     
            }
        }   
    } else {
        if (device_to_host) {
            for (int j = id_y; j < n_y; j += offset_y) {
                for (int i = id_x; i < n_x; i += offset_x) {
                    edge[i + n_x * j] = data[_i(i, j, index)];
                }    
            }
        } else {
            if (edge) {
                for (int j = id_y; j < n_y; j += offset_y) {
                    for (int i = id_x; i < n_x; i += offset_x) {
                        data[_i(i, j, index)] = edge[i + n_x * j];
                    }        
                }
            } else {
                for (int j = id_y; j < n_y; j += offset_y) {
                    for (int i = id_x; i < n_x; i += offset_x) {
                        data[_i(i, j, index)] = u;
                    }       
                }       
            }
        }
    }
    
}
////////////

int main(int argc, char* argv[]) {
    int id;
    int number_processor, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &number_processor);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Get_processor_name(processor_name, &name_len);


    double epsilon, u_0;
    double* tmp;
    char output[128];

    double error, maximum_error = 0;
	
	
	int n_x, n_y, n_z;
	int block_x, block_y, block_z;
	double l_x, l_y, l_z;
	double u_down, u_up, u_left, u_right, u_front, u_back;

    if (id == 0) { 
        std::cin >> block_x >> block_y >> block_z;
        std::cerr << " " << block_x << " " << block_y << " " << block_z << "\n";
        std::cin >> n_x >> n_y >> n_z;
        std::cerr << " " << n_x << " " << n_y << " " << n_z << "\n";
        std::cin >> output;
        std::cerr << " " << output << "\n";
        std::cin >> epsilon;
        std::cerr << " " << epsilon << "\n";
        std::cin >> l_x >> l_y >> l_z;
        std::cerr << " " << l_x << " " << l_y << " " << l_z << "\n";
        std::cin >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back;
        std::cerr << " " << u_down << " " << u_up << " " << u_left << " " << u_right << " " << u_front << " " << u_back << "\n";
        std::cin >> u_0;
        std::cerr << " " << u_0 <<"\n";
    }

    //double start = MPI_Wtime();

    MPI_Bcast(&block_x, 1, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Bcast(&block_y, 1, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Bcast(&block_z, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    MPI_Bcast(&n_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n_z, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
    MPI_Bcast(&l_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&l_y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&l_z, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
    MPI_Bcast(&u_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
    MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(output, 128, MPI_CHAR, 0, MPI_COMM_WORLD);

    int tmp_x = n_x + 2;
    int tmp_y = n_y + 2;
    int tmp_z = n_z + 2;

    int cur_x = id % block_x;
    int cur_y = (id % (block_y * block_x)) / block_x;
    int cur_z = id / block_y / block_x;

    int dim_block_x = n_x * block_x;
    int dim_block_y = n_y * block_y;
    int dim_block_z = n_z * block_z;

    double h_x = l_x / dim_block_x;
    double h_y = l_y / dim_block_y;
    double h_z = l_z / dim_block_z;

    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(id % device_count);

    double* data = (double*)malloc(tmp_x * tmp_y * tmp_z * sizeof(double));
    double* device_data; 
    HANDLE_ERROR(cudaMalloc(&device_data, tmp_x * tmp_y * tmp_z * sizeof(double)));

    for (int i = 0; i < n_x; i++) {
        for (int j = 0; j < n_y; j++) {
            for (int k = 0; k < n_z; k++) {
                data[i + 1 + (j + 1) * tmp_x + (k + 1) * tmp_x * tmp_y] = u_0;
            }        
        }
            
    }

    double* next_data = (double*)malloc(tmp_x * tmp_y * tmp_z * sizeof(double));
    double* device_next_data;
    HANDLE_ERROR(cudaMalloc(&device_next_data, tmp_x * tmp_y * tmp_z * sizeof(double)))
    
    int maximum = std::max(n_x, std::max(n_y, n_z));

    double* buffer = (double*)malloc((maximum * maximum + 2) * sizeof(double));
    double* device_buffer; 
    HANDLE_ERROR(cudaMalloc(&device_buffer, (maximum * maximum + 2) * sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(device_data, data, tmp_x * tmp_y * tmp_z * sizeof(double), cudaMemcpyHostToDevice));
    
    do {

        //////////////////

        if (cur_x + 1 < block_x) {
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, n_x - 1, true, 0.0, 0);
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaMemcpy(buffer, device_buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyDeviceToHost));
            MPI_Send(buffer, n_y * n_z, MPI_DOUBLE, (cur_x + 1) + cur_y * block_x + cur_z * block_x * block_y, id, MPI_COMM_WORLD);
        }


        if (cur_y + 1 < block_y) {
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, n_y - 1, true, 0.0, 1);
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaMemcpy(buffer, device_buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyDeviceToHost));
            MPI_Send(buffer, n_x * n_z, MPI_DOUBLE, cur_x + (cur_y + 1) * block_x + cur_z * block_x * block_y, id, MPI_COMM_WORLD);
        }


        if (cur_z + 1 < block_z) {
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, n_z - 1, true, 0.0, 2);
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaMemcpy(buffer, device_buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyDeviceToHost));
            MPI_Send(buffer, n_x * n_y, MPI_DOUBLE, cur_x + cur_y * block_x + (cur_z + 1) * block_x * block_y, id, MPI_COMM_WORLD);
        }


        //////////////

        if (cur_x > 0) {
            MPI_Recv(buffer, n_y * n_z, MPI_DOUBLE, (cur_x - 1) + cur_y * block_x + cur_z * block_x * block_y, (cur_x - 1) + cur_y * block_x + cur_z * block_x * block_y, MPI_COMM_WORLD, &status);
            HANDLE_ERROR(cudaMemcpy(device_buffer, buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyHostToDevice));
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, -1, false, 0.0, 0);
            HANDLE_ERROR(cudaGetLastError());        
        } else {
            get_copy <<<32, 32>>> (NULL, device_data, n_x, n_y, n_z, -1, false, u_left, 0);  
            HANDLE_ERROR(cudaGetLastError());
        }

        if (cur_y > 0) {
            MPI_Recv(buffer, n_x * n_z, MPI_DOUBLE, cur_x + (cur_y - 1) * block_x + cur_z * block_x * block_y, cur_x + (cur_y - 1) * block_x + cur_z * block_x * block_y, MPI_COMM_WORLD, &status);
            HANDLE_ERROR(cudaMemcpy(device_buffer, buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyHostToDevice));
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, -1, false, 0.0, 1);
            HANDLE_ERROR(cudaGetLastError());
            
        } else {
            get_copy <<<32, 32>>> (NULL, device_data, n_x, n_y, n_z, -1, false, u_front, 1);
            HANDLE_ERROR(cudaGetLastError());
        }

        if (cur_z > 0) {
            MPI_Recv(buffer, n_x * n_y, MPI_DOUBLE, cur_x + cur_y * block_x + (cur_z - 1) * block_x * block_y, cur_x + cur_y * block_x + (cur_z - 1) * block_x * block_y, MPI_COMM_WORLD, &status);
            HANDLE_ERROR(cudaMemcpy(device_buffer, buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyHostToDevice));
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, -1, false, 0.0, 2);
            HANDLE_ERROR(cudaGetLastError());

        } else {
            get_copy <<<32, 32>>> (NULL, device_data, n_x, n_y, n_z, -1, false, u_down, 2);
            HANDLE_ERROR(cudaGetLastError());
        }

        /////////////

        if (cur_x > 0) {
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, 0, true, 0.0, 0);
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaMemcpy(buffer, device_buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyDeviceToHost));

            MPI_Send(buffer, n_y * n_z, MPI_DOUBLE, (cur_x - 1) + cur_y * block_x + cur_z * block_x * block_y, id, MPI_COMM_WORLD);
        }

        if (cur_y > 0) {
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, 0, true, 0.0, 1);
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaMemcpy(buffer, device_buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyDeviceToHost));   
            
            MPI_Send(buffer, n_x * n_z, MPI_DOUBLE, cur_x + (cur_y - 1) * block_x + cur_z * block_x * block_y, id, MPI_COMM_WORLD);
        }


        if (cur_z > 0) {
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, 0, true, 0.0, 2);
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaMemcpy(buffer, device_buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyDeviceToHost));

            MPI_Send(buffer, n_x * n_y, MPI_DOUBLE, cur_x + cur_y * block_x + (cur_z - 1) * block_x * block_y, id, MPI_COMM_WORLD);
        }

        /////////
        if (cur_x + 1 < block_x) {
            MPI_Recv(buffer, n_y * n_z, MPI_DOUBLE, (cur_x + 1) + cur_y * block_x + cur_z * block_x * block_y, (cur_x + 1) + cur_y * block_x + cur_z * block_x * block_y, MPI_COMM_WORLD, &status);
            HANDLE_ERROR(cudaMemcpy(device_buffer, buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyHostToDevice));
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, n_x, false, 0.0, 0);
            HANDLE_ERROR(cudaGetLastError());
        } else {
            get_copy <<<32, 32>>> (NULL, device_data, n_x, n_y, n_z, n_x, false, u_right, 0);
            HANDLE_ERROR(cudaGetLastError());
        }


        if (cur_y + 1 < block_y) {
            MPI_Recv(buffer, n_x * n_z, MPI_DOUBLE, cur_x + (cur_y + 1) * block_x + cur_z * block_x * block_y, cur_x + (cur_y + 1) * block_x + cur_z * block_x * block_y, MPI_COMM_WORLD, &status);
            HANDLE_ERROR(cudaMemcpy(device_buffer, buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyHostToDevice));
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, n_y, false, 0.0, 1);    
            HANDLE_ERROR(cudaGetLastError());
        } else {
            get_copy <<<32, 32>>> (NULL, device_data, n_x, n_y, n_z, n_y, false, u_back, 1);
            HANDLE_ERROR(cudaGetLastError());     
        }


        if (cur_z + 1 < block_z) {
            MPI_Recv(buffer, n_x * n_y, MPI_DOUBLE, cur_x + cur_y * block_x + (cur_z + 1) * block_x * block_y, cur_x + cur_y * block_x + (cur_z + 1) * block_x * block_y, MPI_COMM_WORLD, &status);
            HANDLE_ERROR(cudaMemcpy(device_buffer, buffer, (maximum * maximum + 2) * sizeof(double), cudaMemcpyHostToDevice));
            get_copy <<<32, 32>>> (device_buffer, device_data, n_x, n_y, n_z, n_z, false, 0.0, 2);    
            HANDLE_ERROR(cudaGetLastError());
        } else {
            get_copy <<<32, 32>>> (NULL, device_data, n_x, n_y, n_z, n_z, false, u_up, 2);
            HANDLE_ERROR(cudaGetLastError());
        }

    /////////////////////


        kernel<<<dim3(8, 8, 8), dim3(8, 8, 8)>>> (device_next_data, device_data, n_x, n_y, n_z, h_x, h_y, h_z);
		HANDLE_ERROR(cudaGetLastError());
        get_error<<<dim3(8, 8, 8), dim3(8, 8, 8)>>> (device_next_data, device_data, n_x, n_y, n_z);
        HANDLE_ERROR(cudaGetLastError());

        maximum_error = 0.0;
        error = 0.0;
        double* errors = (double*)malloc(block_x * block_y * block_z  * sizeof(double));

        thrust::device_ptr<double> device_data_pointer = thrust::device_pointer_cast(device_data);
        thrust::device_ptr<double> pointer_error = thrust::max_element(device_data_pointer, device_data_pointer + tmp_x * tmp_y * tmp_z);
        error = *pointer_error;

        MPI_Allgather(&error, 1, MPI_DOUBLE, errors, 1, MPI_DOUBLE, MPI_COMM_WORLD);

        for (int i = 0; i < block_x * block_y * block_z; i++) {
            if (maximum_error < errors[i]) {
                maximum_error = errors[i];    
            }
        }

        tmp = device_data;
        device_data = device_next_data;
        device_next_data = tmp;

    } while (maximum_error > epsilon);

    HANDLE_ERROR(cudaMemcpy(data, device_data, tmp_x * tmp_y * tmp_z * sizeof(double), cudaMemcpyDeviceToHost))
    HANDLE_ERROR(cudaFree(device_data));
    HANDLE_ERROR(cudaFree(device_buffer));
    HANDLE_ERROR(cudaFree(device_next_data));

    int value_size = 14;

    char* output_buffer = (char*)malloc(n_x * n_y * n_z * value_size * sizeof(char));

    for (int k = 0; k < n_z ; k++) {
        for (int j = 0; j < n_y; j++) {
            for (int i = 0; i < n_x; i++) {
                if (data[_i(i, j, k)] < 0) {
                    sprintf(output_buffer + (i + n_x * j + (n_x * n_y * k)) * value_size, "%.6e", data[_i(i, j, k)]);       
                } else {
                    sprintf(output_buffer + (i + n_x * j + (n_x * n_y * k)) * value_size, "%.7e", data[_i(i, j, k)]);    
                }
            }        
        }    
    }

    for (int i = 0; i < n_x * n_y * n_z * value_size; i++) {
        if (output_buffer[i] == '\0') {
            output_buffer[i] = ' ';
        }     
    }
    

    MPI_Datatype part;
    MPI_Type_contiguous(value_size, MPI_CHAR, &part);
    MPI_Type_commit(&part);

    MPI_File out;
    MPI_File_delete(output, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, output, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out);

    MPI_Datatype tmp_array;
    int tmp_array_0[3];
    
    for (int i = 0; i < 3; i++) {
        tmp_array_0[i] = 0;
    }
    int tmp_array_subsizes[3];
    
    tmp_array_subsizes[0] = n_x;
    tmp_array_subsizes[1] = n_y;
    tmp_array_subsizes[2] = n_z;

    int tmp_array_sizes[3];

    tmp_array_sizes[0] = n_x;
    tmp_array_sizes[1] = n_y;
    tmp_array_sizes[2] = n_z;

    MPI_Type_create_subarray(3, tmp_array_sizes, tmp_array_subsizes, tmp_array_0, MPI_ORDER_C, part, &tmp_array);
    MPI_Type_commit(&tmp_array);
    
    
    MPI_Datatype full_array;
    int full_array_0[3];
    full_array_0[0] = cur_x * n_x;
    full_array_0[1] = cur_y * n_y;
    full_array_0[2] = cur_z * n_z;

    int full_array_subsizes[3];
    full_array_subsizes[0] = n_x;
    full_array_subsizes[1] = n_y;
    full_array_subsizes[2] = n_z;

    int full_array_sizes[3];
    full_array_sizes[0] = n_x * block_x;
    full_array_sizes[1] = n_y * block_y;
    full_array_sizes[2] = n_z * block_z;

    MPI_Type_create_subarray(3, full_array_sizes, full_array_subsizes, full_array_0, MPI_ORDER_C, part, &full_array);
    MPI_Type_commit(&full_array);
    
    MPI_File_set_view(out, 0, MPI_CHAR, full_array, "native", MPI_INFO_NULL);
    MPI_File_write_all(out, output_buffer, 1, tmp_array, MPI_STATUS_IGNORE);
    MPI_File_close(&out);
    

    MPI_Finalize();

    free(data);
    free(next_data);
    free(buffer);
    free(output_buffer);

    return 0;
}