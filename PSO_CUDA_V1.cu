#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


const unsigned int particulas = 10000;
const unsigned int ITERATIONS = 1000;
const float SEARCH_MIN = -5.12f;
const float SEARCH_MAX = 5.12f;
const float w = 0.7f;
const float c_ind = 1.5f;
const float c_team = 1.5f;


struct Position {
    float x, y;
    std::string toString() {
        return "(" + std::to_string(x) + "," + std::to_string(y) + ")";
    }
    __device__ __host__ void operator+=(const Position& a) {
        x = x + a.x;
        y = y + a.y;
    }
    __device__ __host__ void operator=(const Position& a) {
        x = a.x;
        y = a.y;
    }
};




struct Particula {
    Position best_position;
    Position current_position;
    Position velocity;
    float best_value;
};




float randomFloat(float low, float high) {
    float range = high-low;
    float pct = static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
    return low + pct * range;
}


__device__ __host__ float calcValue(Position p) {


    return (p.x * p.x - 10 * cosf(2 * M_PI * p.x)) + (p.y * p.y - 10 * cosf(2 * M_PI * p.y));
}


__global__ void init_kernel(curandState *state, long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, state);
}


__global__ void updateTeamBestIndex(Particula *d_particles, float *d_team_best_value, int *d_team_best_index, int particulas) {
    *d_team_best_value = d_particles[0].best_value;
    *d_team_best_index = 0;
    for (int i = 1; i < particulas; i++) {
        if (d_particles[i].best_value < *d_team_best_value) {
            *d_team_best_value = d_particles[i].best_value;
            *d_team_best_index = i;
        }
    }
}


__global__ void updateVelocity(Particula* d_particles, int *d_team_best_index, float w, float c_ind, float c_team, int particulas, curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < particulas) {
        float r_ind = curand_uniform(state);
        float r_team = curand_uniform(state);
        d_particles[idx].velocity.x = w * d_particles[idx].velocity.x +
                       r_ind * c_ind * (d_particles[idx].best_position.x - d_particles[idx].current_position.x) +
                       r_team * c_team * (d_particles[*d_team_best_index].best_position.x - d_particles[idx].current_position.x);
        d_particles[idx].velocity.y = w * d_particles[idx].velocity.y +
                       r_ind * c_ind * (d_particles[idx].best_position.y - d_particles[idx].current_position.y) +
                       r_team * c_team * (d_particles[*d_team_best_index].best_position.y - d_particles[idx].current_position.y);
    }
}


__global__ void updatePosition(Particula *d_particles, int particulas) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < particulas) {
        d_particles[idx].current_position += d_particles[idx].velocity;
        float newValue = calcValue(d_particles[idx].current_position);
        if (newValue < d_particles[idx].best_value) {
            d_particles[idx].best_value = newValue;
            d_particles[idx].best_position = d_particles[idx].current_position;
        }
    }
}




int main(void) {


    long start = std::clock();


    std::srand(std::time(NULL));
    curandState *state;
    cudaMalloc(&state, sizeof(curandState));
    init_kernel<<<1,1>>>(state, clock());




    Particula* h_particles = new Particula[particulas];
    Particula* d_particles;  


    for (int i = 0; i < particulas; i++) {
        h_particles[i].current_position.x = randomFloat(SEARCH_MIN, SEARCH_MAX);
        h_particles[i].current_position.y = randomFloat(SEARCH_MIN, SEARCH_MAX);
        h_particles[i].best_position.x = h_particles[i].current_position.x;
        h_particles[i].best_position.y = h_particles[i].current_position.y;
        h_particles[i].best_value = calcValue(h_particles[i].best_position);
   
        h_particles[i].velocity.x = randomFloat(SEARCH_MIN, SEARCH_MAX);
        h_particles[i].velocity.y = randomFloat(SEARCH_MIN, SEARCH_MAX);
    }


    size_t particleSize = sizeof(Particula) * particulas;
    cudaMalloc((void **)&d_particles, particleSize);
    cudaMemcpy(d_particles, h_particles, particleSize, cudaMemcpyHostToDevice); // dest, source, size, direction




    int *d_team_best_index;
    float *d_team_best_value;


    cudaMalloc((void **)&d_team_best_index, sizeof(int));
    cudaMalloc((void **)&d_team_best_value, sizeof(float));


    updateTeamBestIndex<<<1,1>>>(d_particles, d_team_best_value, d_team_best_index, particulas);


    int blockSize = 1024;
    int gridSize = (particulas + blockSize - 1) / blockSize;




    for (int i = 0; i < ITERATIONS; i++) {
        updateVelocity<<<gridSize, blockSize>>>(d_particles, d_team_best_index, w, c_ind, c_team, particulas, state);
        updatePosition<<<gridSize, blockSize>>>(d_particles, particulas);
        updateTeamBestIndex<<<1,1>>>(d_particles, d_team_best_value, d_team_best_index, particulas);
    }


    int team_best_index;
    cudaMemcpy(&team_best_index, d_team_best_index, sizeof(int), cudaMemcpyDeviceToHost);
   
    cudaMemcpy(h_particles, d_particles, particleSize, cudaMemcpyDeviceToHost);


    long stop = std::clock();
    long elapsed = (stop - start) * 1000 / CLOCKS_PER_SEC;


    // print results


    std::cout << "Valor Optimo: " << h_particles[team_best_index].best_value << std::endl;
    std::cout << "Posicion del Valor: " << h_particles[team_best_index].best_position.toString() << std::endl;
   
    std::cout << "Run time: " << elapsed << "ms" << std::endl;


    cudaFree(d_particles);
    cudaFree(d_team_best_index);
    cudaFree(d_team_best_value);
    cudaFree(state);
    return 0;
}
