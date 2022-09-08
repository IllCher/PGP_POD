#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>

using namespace std;

#define HANDLE_ERROR(error) if (error != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(error)); exit(0);}

typedef unsigned char uchar;

struct vec3 {
    double x;
    double y;
    double z;
};

struct triangle {
    vec3 a;
    vec3 b;
    vec3 c;
    vec3 color;
};

__host__ __device__  vec3 operator + (vec3 a1, vec3 b2) {
    return  {a1.x + b2.x, a1.y + b2.y, a1.z + b2.z};
}

__host__ __device__  uchar4 operator + (uchar4 a1, uchar4 b2) {
    return  make_uchar4(a1.x + b2.x, a1.y + b2.y, a1.z + b2.z, b2.w);
}

__host__ __device__  vec3 operator - (vec3 a1, vec3 b2) {
    return  {a1.x - b2.x, a1.y - b2.y, a1.z - b2.z};
}

__host__ __device__  vec3 operator * (vec3 a, double number) {
    return {number * a.x, number * a.y, number * a.z};
}

__host__ __device__  vec3 operator * (vec3 a, vec3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__  vec3 operator / (vec3 a, vec3 b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__host__ __device__  vec3 operator / (vec3 a, int b) {
    return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__  uchar4 operator / (uchar4 a, int b) {
    return make_uchar4(a.x / b, a.y / b, a.z / b, 0);
}

__host__ __device__  uchar4 to_uchar4 (vec3 b) {
    return make_uchar4(b.x, b.y, b.z, 0);
}

__host__ __device__  vec3 to_vec3 (uchar4 b) {
    return vec3{b.x * 1.0, b.y * 1.0, b.z * 1.0};
}

__host__ __device__  double dot(vec3 a1, vec3 b2) {
    return a1.x * b2.x + a1.y * b2.y + a1.z * b2.z;
}

__host__ __device__  double vector_length(vec3 a) {
    return sqrt(dot(a, a));
}

__host__ __device__  vec3 negative(vec3 a) {
    return {a.x * (-1), a.y * (-1), a.z * (-1)};
}

__host__ __device__  vec3 norm(vec3 a) {
    double num = vector_length(a);
    return {a.x / num, a.y / num, a.z / num};
}

__host__ __device__ vec3 prod(vec3 a1, vec3 b2) {
    return {a1.y * b2.z - a1.z * b2.y,
            a1.z * b2.x - a1.x * b2.z,
            a1.x * b2.y - a1.y * b2.x};
}

__host__ __device__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 d) {
    return { a.x * d.x + b.x * d.y + c.x * d.z,
             a.y * d.x + b.y * d.y + c.y * d.z,
             a.z * d.x + b.z * d.y + c.z * d.z };
}


vec3 get_color(vec3 color) {
    return {color.x * 255., color.y * 255., color.z * 255.};
}

__host__ __device__ vec3 reflect(vec3 vec, vec3 normal) {
    double dot_mult = dot(vec, normal) * (-2.0);
    vec3 part = normal * dot_mult;
    return vec + part;
}

__host__ __device__ uchar4 get_texture_color(uchar4* texture, double x, double y, triangle* triangles) {
    uchar4 tmp = texture[(int)((triangles[36].a.x - x) / abs(triangles[36].a.x - triangles[36].c.x) * 200 + ((triangles[36].c.y - y) / abs(triangles[36].a.x - triangles[36].c.x) * 200) * 200)];
    //printf("%d\n", texture[39999]);
    //printf("%d\n", (int)((triangles[36].a.x - x) / floor_size * 200 + ((triangles[36].c.y - y) / floor_size * 200) * 200));  
    return tmp;
}

__host__ __device__ uchar4 ray(vec3 pos, vec3 dir, triangle* triangles, vec3 light_source, vec3 light_shade, int n, bool shadows_reflections, int recursion_step, uchar4* texture, bool with_texture) {
    int k_min = -1;
    double ts_min;
    pos = pos + dir * 0.01; //фикс зернистости
    //uchar4 color_min = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        vec3 e1 = triangles[i].b - triangles[i].a;
        vec3 e2 = triangles[i].c - triangles[i].a;
        vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10)
            continue;
        vec3 t = pos - triangles[i].a;
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0)
            continue;

        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;

        double ts = dot(q, e2) / div;
        if (ts < 0.0)
            continue;

        if (k_min == -1 || ts < ts_min) {
            k_min = i;
            ts_min = ts;
        }

    }

    if (k_min == -1) {
        return {0, 0, 0, 0};
    }

    if (shadows_reflections) {
        vec3 pos_tmp = dir * ts_min + pos;
        vec3 new_direction = norm(light_source - pos_tmp);
        for (int i = 0; i < n; i++) {
            vec3 e1 = triangles[i].b - triangles[i].a;
            vec3 e2 = triangles[i].c - triangles[i].a;
            vec3 p = prod(new_direction, e2);
            double div = dot(p, e1);
            if (fabs(div) < 1e-10)
                continue;
            
            vec3 t = pos_tmp - triangles[i] .a;
            double u = dot(p, t) / div;
            
            if (u < 0.0 || u > 1.0)
                continue;
            vec3 q = prod(t, e1);
            double v = dot(q, new_direction) / div;
            if (v < 0.0 || v + u > 1.0)
                continue;
            double ts = dot(q, e2) / div;
            if (ts > 0.0 && ts < vector_length(light_source - pos_tmp) && i != k_min) {
                return {0, 0, 0, 0};
            }
    
        }
        uchar4 color_min = {0, 0, 0, 0};
        vec3 result = triangles[k_min].color;
        vec3 reflections = triangles[k_min].color;
        if ((k_min == 36 || k_min == 37) && with_texture) {
            result = to_vec3(get_texture_color(texture, pos_tmp.x, pos_tmp.y, triangles)); 
        }
        if (recursion_step > 0) {
            vec3 reflection_dir = reflect(dir, norm(prod(triangles[k_min].b - triangles[k_min].a, triangles[k_min].c - triangles[k_min].a)));
            double reflection_scale = 0.5;
            double transparency_scale = 0.5;
            reflections = (reflections * (1.0 - reflection_scale) + to_vec3(ray(pos_tmp, reflection_dir, triangles, light_source, light_shade, n, true, recursion_step - 1, texture, with_texture)) * reflection_scale);
            result = (reflections * (1.0 - transparency_scale) + to_vec3(ray(pos_tmp, dir, triangles, light_source, light_shade, n, true, recursion_step - 1, texture, with_texture)) * transparency_scale);
        }

        if ((k_min == 36 || k_min == 37) && with_texture) {
            color_min.x += result.x * light_shade.x;
            color_min.y += result.y * light_shade.y;
            color_min.z += result.z * light_shade.z;    
        } else {
            color_min.x += result.x* light_shade.x;
            color_min.y += result.y* light_shade.y;
            color_min.z += result.z* light_shade.z;    
        }
        color_min.w = 0;
        return color_min;  
    } else {
        return {220, 220, 220};
    }
}

//done//
void render_cpu(vec3 pc, vec3 pv, triangle* triangles, uchar4* points, int width, int height, double angle, vec3 light_source, vec3 light_shade, int n, int recursion_step, uchar4* texture) {
    double dw = 2.0 / (width - 1);
    double dh = 2.0 / (height - 1);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(pv - pc);
    vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = prod(bx, bz);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            vec3 a = {-1.0 + dw * i, (-1.0 + dh * j) * height / width, z};
            vec3 dir = norm(mult(bx, by, bz, a));
            points[(height - 1 - j) * width + i] = ray(pc, dir, triangles, light_source, light_shade, n, true, recursion_step, texture, true);
        }
    }
}

__global__ void render_gpu(vec3 pc, vec3 pv, triangle* triangles, uchar4* points, int width, int height, double angle, vec3 light_source, vec3 light_shade, int n, int recursion_step, uchar4* texture) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    double dw = 2.0 / (width - 1);
    double dh = 2.0 / (height - 1);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(pv - pc);
    vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = prod(bx, bz);
    for (int i = idx; i < width; i += offsetx) {
        for (int j = idy; j < height; j += offsety) {
            vec3 a = {-1.0 + dw * i, (-1.0 + dh * j) * height / width, z};
            vec3 dir = norm(mult(bx, by, bz, a));
            points[(height - 1 - j) * width + i] = ray(pc, dir, triangles, light_source, light_shade, n, true, recursion_step, texture, false); //меняем индексацию чтобы не получить перевернутое изображение
        }
    }

}

///done//

void smoothing_cpu(uchar4* points, uchar4* smoothing_points, int width, int height, int multiplier) {
    int multiplier2 = multiplier * multiplier;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            vec3 mid = {0, 0, 0};
            for (int j = 0; j < multiplier; j++) {
                for (int i = 0; i < multiplier; i++) {
                    mid = mid + to_vec3(smoothing_points[i + j * width * multiplier + x * multiplier + y * width * multiplier2]);
                }
            }
            points[x + width * y] = to_uchar4(mid / (multiplier2));
        }
    }
}

__global__ void smoothing_gpu(uchar4* points, uchar4* smoothing_points, int width, int height, int multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;    
    int multiplier2 = multiplier * multiplier;
    for (int y = idy; y < height; y += offsety) {
        for (int x = idx; x < width; x += offsetx) {
            vec3 mid = {0, 0, 0};
            for (int j = 0; j < multiplier; j++) {
                for (int i = 0; i < multiplier; i++) {
                    mid = mid + to_vec3(smoothing_points[i + j * width * multiplier + x * multiplier + y * width * multiplier2]);
                }
            }
            points[x + width * y] = to_uchar4(mid / (multiplier2));
        }
    }
}



void create_figures(vec3 center1, vec3 center2, vec3 center3, double r1, double r2, double r3, vec3 color1, vec3 color2, vec3 color3, vector<triangle>& triangles) {
    
    vec3 color = get_color(color1);
    
    vec3 point_a  = { r1 / sqrt(3) + center1.x, r1 / sqrt(3) + center1.y, r1/ sqrt(3) + center1.z};
    vec3 point_b  = { r1 / sqrt(3) + center1.x, -r1 / sqrt(3) + center1.y, -r1/ sqrt(3) + center1.z};
    vec3 point_c  = {-r1 / sqrt(3) + center1.x, r1 / sqrt(3) + center1.y, -r1/ sqrt(3) + center1.z};
    vec3 point_d  = {-r1 / sqrt(3) + center1.x, -r1 / sqrt(3) + center1.y, r1/ sqrt(3) + center1.z};

    triangles.push_back({point_a, point_b, point_d, color});
    triangles.push_back({point_a, point_c, point_d, color});
    triangles.push_back({point_b, point_c, point_d, color});
    triangles.push_back({point_a, point_b, point_c, color}); 

    color = get_color(color2);

    point_a = {-r2 / sqrt(3) + center2.x, -r2 / sqrt(3) + center2.y, -r2 / sqrt(3) + center2.z};
    point_b = {-r2 / sqrt(3) + center2.x, -r2 / sqrt(3) + center2.y,  r2 / sqrt(3) + center2.z};
    point_c = {-r2 / sqrt(3) + center2.x,  r2 / sqrt(3) + center2.y, -r2 / sqrt(3) + center2.z};
    point_d = {-r2 / sqrt(3) + center2.x,  r2 / sqrt(3) + center2.y,  r2 / sqrt(3) + center2.z};
    vec3 point_e = { r2 / sqrt(3) + center2.x, -r2 / sqrt(3) + center2.y, -r2 / sqrt(3) + center2.z};
    vec3 point_f = { r2 / sqrt(3) + center2.x, -r2 / sqrt(3) + center2.y,  r2 / sqrt(3) + center2.z};
    vec3 point_g = { r2 / sqrt(3) + center2.x,  r2 / sqrt(3) + center2.y, -r2 / sqrt(3) + center2.z};
    vec3 point_h = { r2 / sqrt(3) + center2.x,  r2 / sqrt(3) + center2.y,  r2 / sqrt(3) + center2.z};
 

    triangles.push_back({point_a, point_b, point_d, color});
    triangles.push_back({point_a, point_c, point_d, color});
    triangles.push_back({point_b, point_f, point_h, color});
    triangles.push_back({point_b, point_d, point_h, color});
    triangles.push_back({point_e, point_f, point_h, color});
    triangles.push_back({point_e, point_g, point_h, color});
    triangles.push_back({point_a, point_e, point_g, color});
    triangles.push_back({point_a, point_c, point_g, color});
    triangles.push_back({point_a, point_b, point_f, color});
    triangles.push_back({point_a, point_e, point_f, color});
    triangles.push_back({point_c, point_d, point_h, color});
    triangles.push_back({point_c, point_g, point_h, color});

    color = get_color(color3);
    double a = (1 + sqrt(5)) / 2;

    point_a = {center3.x   , -r3 / sqrt(3) + center3.y, r3 * a / sqrt(3) + center3.z};
    point_b = {center3.x   , r3 / sqrt(3) + center3.y, r3 * a / sqrt(3) + center3.z};
    point_c = {center3.x - r3 * a / sqrt(3), center3.y, r3 / sqrt(3) + center3.z};
    point_d = {center3.x + r3 * a / sqrt(3), center3.y, r3 / sqrt(3) + center3.z};
    point_e = {center3.x - r3 / sqrt(3), r3 * a / sqrt(3) + center3.y, center3.z};
    point_f = {center3.x + r3 / sqrt(3), r3 * a / sqrt(3) + center3.y, center3.z};
    point_g = {center3.x + r3 / sqrt(3), -r3 * a / sqrt(3) + center3.y, center3.z};
    point_h = {center3.x - r3 / sqrt(3), -r3 * a / sqrt(3) + center3.y, center3.z};
    vec3 point_i = {center3.x - r3 * a / sqrt(3), center3.y, -r3 / sqrt(3) + center3.z};
    vec3 point_j = {center3.x + r3 * a / sqrt(3), center3.y, -r3 / sqrt(3) + center3.z};
    vec3 point_k = {center3.x, -r3 / sqrt(3) + center3.y, -r3 * a / sqrt(3) + center3.z};
    vec3 point_l = {center3.x, r3 / sqrt(3) + center3.y, -r3 * a / sqrt(3) + center3.z};

    triangles.push_back({ point_a,  point_b,  point_c, color});
    triangles.push_back({ point_b,  point_a,  point_d, color});
    triangles.push_back({ point_a,  point_c,  point_h, color});
    triangles.push_back({ point_c,  point_b,  point_e, color});
    triangles.push_back({ point_e,  point_b,  point_f, color});
    triangles.push_back({ point_g,  point_a,  point_h, color});
    triangles.push_back({ point_d,  point_a,  point_g, color});
    triangles.push_back({ point_b,  point_d,  point_f, color});
    triangles.push_back({ point_e,  point_f,  point_l, color});
    triangles.push_back({ point_g,  point_h,  point_k, color});
    triangles.push_back({ point_d,  point_g,  point_j, color});
    triangles.push_back({ point_f,  point_d,  point_j, color});
    triangles.push_back({ point_h,  point_c,  point_i, color});
    triangles.push_back({ point_c,  point_e,  point_i, color});
    triangles.push_back({ point_j,  point_k,  point_l, color});
    triangles.push_back({ point_k,  point_i,  point_l, color});
    triangles.push_back({ point_f,  point_j,  point_l, color});
    triangles.push_back({ point_j,  point_g,  point_k, color});
    triangles.push_back({ point_h,  point_i,  point_k, color});
    triangles.push_back({ point_i,  point_e,  point_l, color});
}


void floor(vec3 a, vec3 b, vec3 c, vec3 d, vec3 color, vector<triangle>& triangles) {
    color = get_color(color);
    triangles.push_back(triangle{a, b, c, color});
    triangles.push_back(triangle{c, d, a, color});
}

int main(int argc, char* argv[]) {
    string cmd = argv[1];
    
    if (cmd == "--default") {
        cout << "20\n ./frames_data\n640 480 120\n5.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0 2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0\n 0.0 0.0 1.0            1.0 0.5 1.0               2.0 1.0 1.0 1.0\n -2.0 3.0 1.0            0.7 0.5 0.5               2.0 1.0 1.0 1.0\n -3.0 -3.0 1.0          0.5 0.5 1.0               2.0 1.0 1.0 1.0\n -10.0 -10.0 -1.0\n  -10.0 10.0 -1.0\n 10.0 10.0 -1.0\n 10.0 -10.0 -1.0\n texture.data\n  1.0 1.0 1.0 0.5\n 1\n 20 -20 100\n   1.0 1.0 1.0\n 3 2\n";
        return 0;
    }
    int frames_count, width, height, angle;
    string path;
    double r0c, z0c, phi0c, Arc, Azc, wrc, wzc, wphic, prc, pzc, r0n, z0n, phi0n, Arn, Azn, wrn, wzn, wphin, prn, pzn;
    string texture_png;
    string tbc;
    vec3 floor_a, floor_b, floor_c, floor_d;
    vec3 light_source, light_shade;
    vector<triangle> figures;
    uchar4* points;
    uchar4* points_smoothing;
    uchar4* texture;
    int recursion_step;
    int multiplier;
    vec3 center1, center2, center3;
    vec3 color1, color2, color3, color4;
    double r1, r2, r3;
    cin >> frames_count;
    cin >> path;
    cin >> width >> height >> angle;
    double rc, zc, phic, rn, zn, phin;
    vec3 pc, pv;
    int smoothing_rays;

    cin >> r0c >> z0c >> phi0c >> Arc >> Azc >> wrc >> wzc >> wphic >> prc >> pzc >> r0n >> z0n >> phi0n >> Arn >> Azn >> wrn >> wzn >> wphin >> prn >> pzn;

    cin >> center1.x >> center1.y >> center1.z >> color1.x >> color1.y >> color1.z >> r1 >> tbc >> tbc >> tbc;
    cin >> center2.x >> center2.y >> center2.z >> color2.x >> color2.y >> color2.z >> r2 >> tbc >> tbc >> tbc;
    cin >> center3.x >> center3.y >> center3.z >> color3.x >> color3.y >> color3.z >> r3 >> tbc >> tbc >> tbc;

    create_figures(center1, center2, center3, r1, r2, r3, color1, color2, color3, figures);

    cin >> floor_a.x >> floor_a.y >> floor_a.z;
    cin >> floor_b.x >> floor_b.y >> floor_b.z;
    cin >> floor_c.x >> floor_c.y >> floor_c.z;
    cin >> floor_d.x >> floor_d.y >> floor_d.z;

    cin >> texture_png;

    FILE* file = fopen(texture_png.c_str(), "rb");

    int width_texture, height_texture;

    fread(&width_texture, sizeof(int), 1, file);
    fread(&height_texture, sizeof(int), 1, file);
    texture = (uchar4 *)malloc(width_texture * height_texture * sizeof(uchar4));
    fread(texture, sizeof(uchar4), width_texture * height_texture, file);

    fclose(file);

    cin >> color4.x >> color4.y >> color4.z;
    cin >> tbc;

    floor(floor_a, floor_b, floor_c, floor_d, color4, figures);

    cin >> tbc;
    cin >> light_source.x >> light_source.y >> light_source.z;
    cin >> light_shade.x >> light_shade.y >> light_shade.z;

    cin >> recursion_step >> multiplier;

    int smoothing_width = multiplier * width;
    int smoothing_height = multiplier * height;

    texture = (uchar4*)malloc(width * height * sizeof(uchar4));
    points = (uchar4*)malloc(width * height * sizeof(uchar4));
    points_smoothing = (uchar4*)malloc(smoothing_width * smoothing_height * sizeof(uchar4));

    double total_time = 0;

    cout << "Total triangles: " << 38 << " Resolution: " << width << "x" << height;
    cout << " Total frames: " << frames_count << "\n";
    cout << "Frame         Time                 Total rays\n";

    for (int i = 0; i < frames_count; i++) {

        cudaEvent_t start_cuda, stop_cuda;
        cudaEventCreate(&start_cuda);
        cudaEventCreate(&stop_cuda);
        cudaEventRecord(start_cuda);
        auto start = chrono::steady_clock::now();
        
        double time_step = 2.0 * M_PI / frames_count;
        double t = time_step * i;

        rc = r0c + Arc * sin(wrc * t + prc);
        zc = z0c + Azc * sin(wzc * t + pzc);
        phic = phi0c + wphin * t;

        rn = r0n + Arn * sin(wrn * t + prn);
        zn = z0n + Azn * sin(wzn * t + pzn);
        phin = phi0n + wphin * t;

        pc = {rc * cos(phic), rc * sin(phic), zc};
        pv = {rn * cos(phin), rn * sin(phin), zn};

        smoothing_rays = smoothing_width * smoothing_height;

        if (cmd == "--gpu") {
            uchar4* gpu_texture;
            HANDLE_ERROR(cudaMalloc((uchar4**)(&gpu_texture), width_texture * height_texture * sizeof(uchar4)));
            uchar4* gpu_points;
            HANDLE_ERROR(cudaMalloc((uchar4**)(&gpu_points), width * height * sizeof(uchar4)));
            uchar4* gpu_points_smoothing;
            HANDLE_ERROR(cudaMalloc((uchar4**)(&gpu_points_smoothing), smoothing_width * smoothing_height * sizeof(uchar4)));
            triangle* gpu_figures;
            HANDLE_ERROR(cudaMalloc((triangle**)(&gpu_figures), figures.size() * sizeof(triangle)));
            HANDLE_ERROR(cudaMemcpy(gpu_figures, figures.data(), figures.size() * sizeof(triangle), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(gpu_points_smoothing, points_smoothing, smoothing_width * smoothing_height * sizeof(uchar4), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(gpu_points, points, width * height * sizeof(uchar4), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(gpu_texture, texture, width_texture * height_texture * sizeof(uchar4), cudaMemcpyHostToDevice));
            render_gpu<<<128, 128>>>(pc, pv, gpu_figures, gpu_points_smoothing, smoothing_width, smoothing_height, angle, light_source, light_shade, figures.size(), recursion_step, gpu_texture);
            HANDLE_ERROR(cudaGetLastError());           
            smoothing_gpu<<<128, 128>>>(gpu_points, gpu_points_smoothing, width, height, multiplier);
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaMemcpy(points, gpu_points, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));           
            HANDLE_ERROR(cudaFree(gpu_points));
            HANDLE_ERROR(cudaFree(gpu_points_smoothing));
            HANDLE_ERROR(cudaFree(gpu_figures));
            HANDLE_ERROR(cudaFree(gpu_texture));

        } else {
            render_cpu(pc, pv, figures.data(), points_smoothing, smoothing_width, smoothing_height, angle, light_source, light_shade, figures.size(), recursion_step, texture);
            smoothing_cpu(points, points_smoothing, width, height, multiplier);    
        }

        cudaEventRecord(stop_cuda);
        auto end = chrono::steady_clock::now();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda);
        cout << i + 1 << "         ";
        total_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milliseconds        ";
        cout << smoothing_rays << "    \n";

        string filename = to_string(i + 1) + ".data";
        FILE* file = fopen(filename.c_str(), "wb");
        fwrite(&width, sizeof(int), 1, file);
        fwrite(&height, sizeof(int), 1, file);
        fwrite(points, sizeof(uchar4), width * height, file);
        fclose(file);
    }
    free(points);
    free(points_smoothing);

    cout << "Total time: " << total_time << "ms\n";
    return 0;
}