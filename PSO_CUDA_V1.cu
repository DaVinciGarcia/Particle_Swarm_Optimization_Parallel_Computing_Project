#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
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

    float randomValue1 = minVal + (maxVal - minVal) * curand_uniform(&state);
    float randomValue2 = minVal + (maxVal - minVal) * curand_uniform(&state);
    float randomValue3 = minVal + (maxVal - minVal) * curand_uniform(&state);
    float randomValue4 = minVal + (maxVal - minVal) * curand_uniform(&state);

    if (tid < N) { 
        position_x[tid] = randomValue1;
        personal_best_x[tid] = randomValue1;
        position_y[tid] = randomValue2;
        personal_best_y[tid] = randomValue2;
        velocity_x[tid] = randomValue3;
        velocity_y[tid] = randomValue4;
    }
}

__global__ void evalFunct(float* position_x, float*  position_y, float* value) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float aux = (20 + (position_x[tid] * position_x[tid]) 
                    + (position_y[tid] * position_y[tid]) 
                    - 10*(cosf(2 * M_PI * position_x[tid]) 
                    + cosf(2 * M_PI * position_y[tid])));
    value[tid] = aux;
}

__global__ void copyTwoFloatValues(float* values_from, float* values_to, int N){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N){
        float aux = values_from[tid];
        values_to[tid] = aux;
    }
}

__global__ void updateBestGlobal(float* personal_best, float* global_best, int* global_best_index, int parts_qty) {
    *global_best = 10;
    *global_best_index = 5;
    for (int i = 1; i < parts_qty-1; i++) {
        if (personal_best[i] < *global_best) {
            *global_best = personal_best[i];
            *global_best_index = i;
        }
    }
}

int main() {
    int threadsPerBlock = 256;
    int blocksPerGrid = (parts_qty + threadsPerBlock - 1) / threadsPerBlock;


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


    float* d_current_position_inx;
    float* d_current_position_iny;
    float* d_best_position_inx;
    float* d_best_position_iny;
    float* d_velocity_inx;
    float* d_velocity_iny;
    float* d_current_value;
    float* d_pBest;
    float* d_gBest;
    int* d_gBestIndex;


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


    initializeRandomPositions<<<blocksPerGrid, threadsPerBlock>>>(d_current_position_inx, 
                                                                    d_current_position_iny, 
                                                                    d_velocity_inx, 
                                                                    d_velocity_iny,
                                                                    d_best_position_inx,
                                                                    d_best_position_iny, 
                                                                    time(NULL), 
                                                                    min_range_value, max_range_value,
                                                                    parts_qty);

    evalFunct<<<blocksPerGrid, threadsPerBlock>>>(d_current_position_inx, 
                                                    d_current_position_iny, 
                                                    d_current_value);

    copyTwoFloatValues<<<blocksPerGrid, threadsPerBlock>>>(d_current_value, d_pBest, parts_qty);

    updateBestGlobal<<<1,1>>>(d_pBest, d_gBest, d_gBestIndex, parts_qty);

    
    cudaMemcpy(&gBest, d_gBest, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gBestIndex, d_gBestIndex, sizeof(int), cudaMemcpyDeviceToHost);


    std::cout << "Global best: "<< gBest << " "; 
    std::cout << std::endl;
    std::cout << "Global best index: "<< gBestIndex << " "; 
    std::cout << std::endl;



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

    return 0;
}
