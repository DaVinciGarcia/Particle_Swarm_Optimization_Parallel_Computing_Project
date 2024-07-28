#include <iostream>
#include <stdlib.h>
#include <cmath> 
#include <string>
#include <ctime> 
#include <cuda.h> 
#include <cuda_runtime.h>
#include <iomanip>
#include <curand_kernel.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const unsigned int parts_qty = 1000;
const unsigned int iterations = 1000;
const float min_range_value = -5.12f;
const float max_range_value = 5.12f;
const float w = 0.7f;
const float c1 = 1.5f;
const float c2 = 1.5f;

const int threadsPerBlock = 256;
const int blocksPerGrid = (parts_qty + threadsPerBlock - 1) / threadsPerBlock;

struct Particle {
    float* current_position_inx; 
    float* current_position_iny;

    float* best_position_inx; 
    float* best_position_iny;

    float* velocity_inx;
    float* velocity_iny;

    float* current_value;
    float* pBest;
};

__device__ float calcFunct(float pos_x, float pos_y) {
    return (20 + (pos_x * pos_x) + (pos_y * pos_y) - 
            10 * (cosf(2 * M_PI * pos_x) + cosf(2 * M_PI * pos_y)));
}

__global__ void initializeRandomPositions(float* position_x, 
                                          float* position_y, 
                                          float* velocity_x, 
                                          float* velocity_y,
                                          float* personal_best_x,
                                          float* personal_best_y, 
                                          int seed, float minVal, 
                                          float maxVal, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init(seed, tid, 0, &state);

    if (tid < N) { 
        position_x[tid] = minVal + (maxVal - minVal) * curand_uniform(&state);
        position_y[tid] = minVal + (maxVal - minVal) * curand_uniform(&state);
        velocity_x[tid] = minVal + (maxVal - minVal) * curand_uniform(&state);
        velocity_y[tid] = minVal + (maxVal - minVal) * curand_uniform(&state);
        personal_best_x[tid] = position_x[tid];
        personal_best_y[tid] = position_y[tid];
    }
}

__global__ void evalFunct(float* position_x, float* position_y, float* value) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < parts_qty) {
        value[tid] = calcFunct(position_x[tid], position_y[tid]);
    }
}

__global__ void copyTwoFloatValues(float* values_from, float* values_to, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        values_to[tid] = values_from[tid];
    }
}

__global__ void reduceMin(float* input, float* output, int* outputIndex, int n) {
    extern __shared__ float sdata[];
    int* sindex = (int*)&sdata[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = input[i];
        sindex[tid] = i;
    } else {
        sdata[tid] = FLT_MAX;
        sindex[tid] = -1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            if (sdata[tid] > sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
                sindex[tid] = sindex[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
        outputIndex[blockIdx.x] = sindex[0];
    }
}

__global__ void updateBestGlobal(float* personal_best, 
                                 float* global_best, 
                                 int* global_best_index, 
                                 int parts_qty) {
    *global_best = personal_best[0];
    *global_best_index = 0;
    for (int i = 1; i < parts_qty; i++) {
        if (personal_best[i] < *global_best) {
            *global_best = personal_best[i];
            *global_best_index = i;
        }
    }
}

__global__ void updateVelocity(float* d_current_position_inx,
                               float* d_current_position_iny,
                               float* d_best_position_inx,
                               float* d_best_position_iny,
                               float* d_velocity_inx,
                               float* d_velocity_iny,
                               int *d_team_best_index, 
                               float w, float c1, float c2, 
                               int parts_qty, curandState *state) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < parts_qty) {
        float r_ind = curand_uniform(state);
        float r_team = curand_uniform(state);
        d_velocity_inx[idx] = w * d_velocity_inx[idx] +
                              r_ind * c1 * (d_best_position_inx[idx] - d_current_position_inx[idx]) +
                              r_team * c2 * (d_best_position_inx[*d_team_best_index] - d_current_position_inx[idx]);

        d_velocity_iny[idx] = w * d_velocity_iny[idx] +
                              r_ind * c1 * (d_best_position_iny[idx] - d_current_position_iny[idx]) +
                              r_team * c2 * (d_best_position_iny[*d_team_best_index] - d_current_position_iny[idx]);
    }
}

__global__ void updatePosition(float* d_current_position_inx,
                               float* d_current_position_iny,
                               float* d_best_position_inx,
                               float* d_best_position_iny,
                               float* d_velocity_inx,
                               float* d_velocity_iny,
                               float* d_pBest,
                               int parts_qty) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < parts_qty) {
        d_current_position_inx[idx] += d_velocity_inx[idx];
        d_current_position_iny[idx] += d_velocity_iny[idx];

        float newValue = calcFunct(d_current_position_inx[idx], d_current_position_iny[idx]);
        if (newValue < d_pBest[idx]) {
            d_pBest[idx] = newValue;
            d_best_position_inx[idx] = d_current_position_inx[idx];
            d_best_position_iny[idx] = d_current_position_iny[idx];
        }
    }
}

int main() {

    Particle particle;

    particle.current_position_inx = new float[parts_qty];
    particle.current_position_iny = new float[parts_qty];
    particle.best_position_inx = new float[parts_qty];
    particle.best_position_iny = new float[parts_qty];
    particle.velocity_inx = new float[parts_qty];
    particle.velocity_iny = new float[parts_qty];
    particle.current_value = new float[parts_qty];
    particle.pBest = new float[parts_qty];
    float gBest;
    int gBestIndex;
    int h_best_global_indexes[blocksPerGrid];
    float h_best_global[blocksPerGrid];
    size_t sharedMemSize = threadsPerBlock * sizeof(float) + threadsPerBlock * sizeof(int);

    float* d_current_position_inx;
    float* d_current_position_iny;
    float* d_best_position_inx;
    float* d_best_position_iny;
    float* d_velocity_inx;
    float* d_velocity_iny;
    float* d_current_value;
    float* d_pBest;
    float* d_gBest;
    float* d_blocks_global_bests;
    int* d_blocks_global_bests_index;
    int* d_gBestIndex;
    curandState *state;

    cudaMalloc((void**)&d_current_position_inx, parts_qty * sizeof(float));
    cudaMalloc((void**)&d_current_position_iny, parts_qty * sizeof(float));
    cudaMalloc((void**)&d_best_position_inx, parts_qty * sizeof(float));
    cudaMalloc((void**)&d_best_position_iny, parts_qty * sizeof(float));
    cudaMalloc((void**)&d_velocity_inx, parts_qty * sizeof(float));
    cudaMalloc((void**)&d_velocity_iny, parts_qty * sizeof(float));
    cudaMalloc((void**)&d_current_value, parts_qty * sizeof(float));
    cudaMalloc((void**)&d_pBest, parts_qty * sizeof(float));
    cudaMalloc((void**)&d_gBest, sizeof(float));
    cudaMalloc((void**)&d_gBestIndex, sizeof(int));
    cudaMalloc(&state, sizeof(curandState) * parts_qty);
    cudaMalloc(&d_blocks_global_bests, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_blocks_global_bests_index, blocksPerGrid * sizeof(int));

    initializeRandomPositions<<<blocksPerGrid, threadsPerBlock>>>(d_current_position_inx, 
                                                                  d_current_position_iny, 
                                                                  d_velocity_inx, 
                                                                  d_velocity_iny,
                                                                  d_best_position_inx,
                                                                  d_best_position_iny, 
                                                                  time(NULL), 
                                                                  min_range_value, max_range_value,
                                                                  parts_qty);
    cudaDeviceSynchronize();

    evalFunct<<<blocksPerGrid, threadsPerBlock>>>(d_current_position_inx, 
                                                  d_current_position_iny, 
                                                  d_current_value);
    cudaDeviceSynchronize();

    copyTwoFloatValues<<<blocksPerGrid, threadsPerBlock>>>(d_current_value, d_pBest, parts_qty);
    cudaDeviceSynchronize();

    updateBestGlobal<<<1,1>>>(d_pBest, d_gBest, d_gBestIndex, parts_qty);
    cudaDeviceSynchronize();

    reduceMin<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_pBest, 
                                                                 d_blocks_global_bests, 
                                                                 d_blocks_global_bests_index, 
                                                                 parts_qty);
    cudaDeviceSynchronize();

    cudaMemcpy(h_best_global, d_blocks_global_bests, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_global_indexes, d_blocks_global_bests_index, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    for (int k = 0; k < blocksPerGrid; k++) {
        std::cout << "Block " << k << " best value: " << h_best_global[k] << ", index: " << h_best_global_indexes[k] << std::endl;
    }

    for (int i = 0; i < iterations; i++) {
        updateVelocity<<<blocksPerGrid, threadsPerBlock>>>(d_current_position_inx,
                                                           d_current_position_iny,
                                                           d_best_position_inx,
                                                           d_best_position_iny,
                                                           d_velocity_inx,
                                                           d_velocity_iny,
                                                           d_gBestIndex, 
                                                           w, c1, c2, 
                                                           parts_qty, 
                                                           state);
        cudaDeviceSynchronize();
        
        updatePosition<<<blocksPerGrid, threadsPerBlock>>>(d_current_position_inx,
                                                           d_current_position_iny,
                                                           d_best_position_inx,
                                                           d_best_position_iny,
                                                           d_velocity_inx,
                                                           d_velocity_iny,
                                                           d_pBest,
                                                           parts_qty);
        cudaDeviceSynchronize();

        updateBestGlobal<<<1,1>>>(d_pBest, d_gBest, d_gBestIndex, parts_qty);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(&gBest, d_gBest, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gBestIndex, d_gBestIndex, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle.best_position_inx, d_best_position_inx, sizeof(float)*parts_qty, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle.best_position_iny, d_best_position_iny, sizeof(float)*parts_qty, cudaMemcpyDeviceToHost);

    std::cout << "Global best: " << std::fixed << std::setprecision(5) << gBest << std::endl;
    std::cout << "Global Best Position: (" << std::fixed << std::setprecision(5) << particle.best_position_inx[gBestIndex] << ", " 
              << particle.best_position_iny[gBestIndex] << ")" << std::endl;

    delete[] particle.current_position_inx;
    delete[] particle.current_position_iny;
    delete[] particle.velocity_inx;
    delete[] particle.velocity_iny;
    delete[] particle.current_value;
    delete[] particle.pBest;

    cudaFree(d_current_position_inx);
    cudaFree(d_current_position_iny);
    cudaFree(d_best_position_inx);
    cudaFree(d_best_position_iny);
    cudaFree(d_velocity_inx);
    cudaFree(d_velocity_iny);
    cudaFree(d_current_value);
    cudaFree(d_pBest);
    cudaFree(d_gBest);
    cudaFree(d_gBestIndex);
    cudaFree(state);
    cudaFree(d_blocks_global_bests);
    cudaFree(d_blocks_global_bests_index);

    return 0;
}
