#include <iostream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <omp.h>

#define _i(i, j, k) (i + 1 + (j + 1) * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2))

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

    double maximum_error = 0;
	
	int i, j, k;
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

    double* data = (double*)malloc(tmp_x * tmp_y * tmp_z * sizeof(double));

    for (i = 0; i < n_x; i++) {
        for (j = 0; j < n_y; j++) {
            for (k = 0; k < n_z; k++) {
                data[i + 1 + (j + 1) * tmp_x + (k + 1) * tmp_x * tmp_y] = u_0;
            }        
        }
            
    }

    double* next_data = (double*)malloc(tmp_x * tmp_y * tmp_z * sizeof(double));

    double* errors = (double*)malloc(block_x * block_y * block_z  * sizeof(double));

    /*
    0 - left
    1 - right
    2 - front
    3 - back
    4 - up
    5 - down
    */

    int full_sizes[3];
    int sub_sizes[3];
    int size_0[3];

    full_sizes[0] = n_x + 2;
    full_sizes[1] = n_y + 2;
    full_sizes[2] = n_z + 2;

    MPI_Datatype send_yz_0, send_yz_1, recv_yz_0, recv_yz_1;

    sub_sizes[0] = 1;
    sub_sizes[1] = n_y;
    sub_sizes[2] = n_z;
    size_0[0] = 1;
    size_0[1] = 1;
    size_0[2] = 1;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &send_yz_0);
    MPI_Type_commit(&send_yz_0);

    size_0[0] = n_x;
    size_0[1] = 1;
    size_0[2] = 1;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &send_yz_1);
    MPI_Type_commit(&send_yz_1);

    size_0[0] = 0;
    size_0[1] = 1;
    size_0[2] = 1;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &recv_yz_0);
    MPI_Type_commit(&recv_yz_0);

    size_0[0] = n_x + 1;
    size_0[1] = 1;
    size_0[2] = 1;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &recv_yz_1);
    MPI_Type_commit(&recv_yz_1);



    MPI_Datatype send_xz_3, send_xz_2, recv_xz_3, recv_xz_2;

    sub_sizes[0] = n_x;
    sub_sizes[1] = 1;
    sub_sizes[2] = n_z;
    size_0[0] = 1;
    size_0[1] = n_y;
    size_0[2] = 1;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &send_xz_3);
    MPI_Type_commit(&send_xz_3);

    size_0[0] = 1;
    size_0[1] = 1;
    size_0[2] = 1;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &send_xz_2);
    MPI_Type_commit(&send_xz_2);

    size_0[0] = 1;
    size_0[1] = n_y + 1;
    size_0[2] = 1;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &recv_xz_3);
    MPI_Type_commit(&recv_xz_3);

    size_0[0] = 1;
    size_0[1] = 0;
    size_0[2] = 1;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &recv_xz_2);
    MPI_Type_commit(&recv_xz_2);



    MPI_Datatype send_xy_4, send_xy_5, recv_xy_4, recv_xy_5;

    sub_sizes[0] = n_x;
    sub_sizes[1] = n_y;
    sub_sizes[2] = 1;
    size_0[0] = 1;
    size_0[1] = 1;
    size_0[2] = n_z;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &send_xy_5);
    MPI_Type_commit(&send_xy_5);

    size_0[0] = 1;
    size_0[1] = 1;
    size_0[2] = 1;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &send_xy_4);
    MPI_Type_commit(&send_xy_4);

    size_0[0] = 1;
    size_0[1] = 1;
    size_0[2] = n_z + 1;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &recv_xy_5);
    MPI_Type_commit(&recv_xy_5);

    size_0[0] = 1;
    size_0[1] = 1;
    size_0[2] = 0;

    MPI_Type_create_subarray(3, full_sizes, sub_sizes, size_0, MPI_ORDER_FORTRAN, MPI_DOUBLE, &recv_xy_4);
    MPI_Type_commit(&recv_xy_4);

    do {
        maximum_error = 0.0;
        
        //////////////////

        if (cur_x + 1 < block_x) {
            MPI_Send(data, 1, send_yz_1, (cur_x + 1) + cur_y * block_x + cur_z * block_x * block_y, id, MPI_COMM_WORLD);
        }


        if (cur_y + 1 < block_y) {
            MPI_Send(data, 1, send_xz_3, cur_x + (cur_y + 1) * block_x + cur_z * block_x * block_y, id, MPI_COMM_WORLD);
        }


        if (cur_z + 1 < block_z) {
            MPI_Send(data, 1, send_xy_5, cur_x + cur_y * block_x + (cur_z + 1) * block_x * block_y, id, MPI_COMM_WORLD);
        }

        //////////////////


        /////////////////////
        if (cur_x > 0) {
            MPI_Recv(data, 1, recv_yz_0, (cur_x - 1) + cur_y * block_x + cur_z * block_x * block_y, (cur_x - 1) + cur_y * block_x + cur_z * block_x * block_y, MPI_COMM_WORLD, &status);
        } else {
            #pragma omp parallel for private(i, j, k) shared(data)
            for (int i = 0; i < n_z; i++) {
                for (int j = 0; j < n_y; j++) {
                    data[(j + 1) * (n_x + 2) + (i + 1) * (n_x + 2) * (n_y + 2)] = u_left;
                }
            }       
        }
        if (cur_y > 0) {
            MPI_Recv(data, 1, recv_xz_2, cur_x + (cur_y - 1) * block_x + cur_z * block_x * block_y, cur_x + (cur_y - 1) * block_x + cur_z * block_x * block_y, MPI_COMM_WORLD, &status);       
        } else {
            #pragma omp parallel for private(i, j, k) shared(data)
            for (int j = 0; j < n_z; j++) {
                for (int i = 0; i < n_x; i++) {
                    data[i + 1 + (j + 1) * (n_x + 2) * (n_y + 2)] = u_front;
                }
            }
        }
        if (cur_z > 0) {
            MPI_Recv(data, 1, recv_xy_4, cur_x + cur_y * block_x + (cur_z - 1) * block_x * block_y, cur_x + cur_y * block_x + (cur_z - 1) * block_x * block_y, MPI_COMM_WORLD, &status);
        } else {
            #pragma omp parallel for private(i, j, k) shared(data)
            for (int j = 0; j < n_y; j++) {
                for (int i = 0; i < n_x; i++) {
                    data[i + 1 + (j + 1) * (n_x + 2)] = u_down;
                }
            }
        }

        ////////////////////
        if (cur_x > 0) {
            MPI_Send(data, 1, send_yz_0, (cur_x - 1) + cur_y * block_x + cur_z * block_x * block_y, id, MPI_COMM_WORLD);
        }

        if (cur_y > 0) {  
            MPI_Send(data, 1, send_xz_2, cur_x + (cur_y - 1) * block_x + cur_z * block_x * block_y, id, MPI_COMM_WORLD);
        }

        if (cur_z > 0) {
            MPI_Send(data, 1, send_xy_4, cur_x + cur_y * block_x + (cur_z - 1) * block_x * block_y, id, MPI_COMM_WORLD);
        }

        ///////////////////

        if (cur_x + 1 < block_x) {
            MPI_Recv(data, 1, recv_yz_1, (cur_x + 1) + cur_y * block_x + cur_z * block_x * block_y, (cur_x + 1) + cur_y * block_x + cur_z * block_x * block_y, MPI_COMM_WORLD, &status);   
        } else {
            #pragma omp parallel for private(i, j, k) shared(data)
            for (int i = 0; i < n_z; i++) {
                for (int j = 0; j < n_y; j++) {
                    data[n_x + 1 + (j + 1) * (n_x + 2) + (i + 1) * (n_x + 2) * (n_y + 2)] = u_right;
                }
            }
        }


        if (cur_y + 1 < block_y) {
            MPI_Recv(data, 1, recv_xz_3, cur_x + (cur_y + 1) * block_x + cur_z * block_x * block_y, cur_x + (cur_y + 1) * block_x + cur_z * block_x * block_y, MPI_COMM_WORLD, &status);    
        } else {
            #pragma omp parallel for private(i, j, k) shared(data)
            for (int j = 0; j < n_z; j++) {
                for (int i = 0; i < n_x; i++) {
                    data[i + 1 + (n_y + 1) * (n_x + 2) + (j + 1) * (n_x + 2) * (n_y + 2)] = u_back;
                }    
            }     
        }


        if (cur_z + 1 < block_z) {
            MPI_Recv(data, 1, recv_xy_5, cur_x + cur_y * block_x + (cur_z + 1) * block_x * block_y, cur_x + cur_y * block_x + (cur_z + 1) * block_x * block_y, MPI_COMM_WORLD, &status);
        } else {
            #pragma omp parallel for private(i, j, k) shared(data)
            for (int j = 0; j < n_y; j++) {
                for (int i = 0; i < n_x; i++) {
                    data[i + 1 + (j + 1) * (n_x + 2) + (n_z + 1) * (n_x + 2) * (n_y + 2)] = u_up;
                }
            }
        }

    /////////////////////
        
        #pragma omp parallel for private(i, j, k) shared(data, next_data, n_x, n_y, n_z, h_x, h_y, h_z) reduction(max:maximum_error)
        for (int i = 0; i < n_x; i++) {
            for (int j = 0; j < n_y; j++) {
                for (int k = 0; k < n_z; k++) {
                    
                    double h_x_squared = h_x * h_x;
                    double h_y_squared = h_y * h_y;
                    double h_z_squared = h_z * h_z;

                    double add1 = (data[i + 2 + (j + 1) * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2)] + data[i + (j + 1) * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2)]) / h_x_squared;
                    double add2 = (data[i + 1 + (j + 2) * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2)] + data[i + 1 + j * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2)]) / h_y_squared; 
                    double add3 = (data[i + 1 + (j + 1) * (n_x + 2) + (k + 2) * (n_x + 2) * (n_y + 2)] + data[i + 1 + (j + 1) * (n_x + 2) + k * (n_x + 2) * (n_y + 2)]) / h_z_squared;
                    double devider = 2 * (1.0 / h_x_squared + 1.0 / h_y_squared + 1.0 / h_z_squared);

                    int tmp = i + 1 + (j + 1) * (n_x + 2) + (k + 1) * (n_x + 2) * (n_y + 2);
                    next_data[tmp] = (add1 + add2 + add3) / devider;
                    maximum_error = std::max(maximum_error, fabs(next_data[_i(i, j, k)] - data[_i(i, j, k)]));
                }
            }     
        }

        MPI_Allgather(&maximum_error, 1, MPI_DOUBLE, errors, 1, MPI_DOUBLE, MPI_COMM_WORLD);

        for (i = 0; i < block_x * block_y * block_z; i++) {
            if (maximum_error < errors[i]) {
                maximum_error = errors[i];    
            }
        }

        tmp = next_data;
        next_data = data;
        data = tmp;

    } while (maximum_error > epsilon);

    int value_size = 14;

    char* output_buffer = (char*)malloc(n_x * n_y * n_z * value_size * sizeof(char));

    for (k = 0; k < n_z ; k++) {
        for (j = 0; j < n_y; j++) {
            for (i = 0; i < n_x; i++) {
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

    MPI_Type_create_subarray(3, tmp_array_sizes, tmp_array_subsizes, tmp_array_0, MPI_ORDER_FORTRAN, part, &tmp_array);
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

    MPI_Type_create_subarray(3, full_array_sizes, full_array_subsizes, full_array_0, MPI_ORDER_FORTRAN, part, &full_array);
    MPI_Type_commit(&full_array);
    
    MPI_File_set_view(out, 0, MPI_CHAR, full_array, "native", MPI_INFO_NULL);
    MPI_File_write_all(out, output_buffer, 1, tmp_array, MPI_STATUS_IGNORE);
    MPI_File_close(&out);

    MPI_Finalize();

    free(data);
    free(next_data);
    free(output_buffer);
    return 0;
}