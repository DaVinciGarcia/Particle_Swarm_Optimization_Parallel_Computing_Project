#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <ctime>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <curand_kernel.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Particle {
    float* best_position_inx; 
    float* best_position_iny;

    float* current_position_inx; 
    float* current_position_iny;

    float* velocity_inx;
    float* velocity_iny;

    float* current_value;
    float* pBest;
};

__global__ float calcValue(float* position_x, float*  position_y, float* value) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float aux = (20 + (position_x[tid] * position_x[tid]) + (position_y[tid] * position_y[tid]) - 10*(cosf(2 * M_PI * position_x[tid]) + cosf(2 * M_PI * position_y[tid]))):
    value[tid] = aux;
}