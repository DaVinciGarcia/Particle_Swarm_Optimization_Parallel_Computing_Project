#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void initializeRandomPositions(float* d_output_1, float* d_output_2, float* d_output_3, float* d_output_4, int seed, float minVal, float maxVal) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Inicializar el generador de números aleatorios
    curandState state;
    curand_init(seed, tid, 0, &state);

    // Generar números aleatorios y almacenarlos en los arreglos
    float randomValue1 = minVal + (maxVal - minVal) * curand_uniform(&state);
    float randomValue2 = minVal + (maxVal - minVal) * curand_uniform(&state);
    float randomValue3 = minVal + (maxVal - minVal) * curand_uniform(&state);
    float randomValue4 = minVal + (maxVal - minVal) * curand_uniform(&state);

    // Escribir los valores generados en los arreglos correspondientes
    if (tid < N) { // Asegúrate de que tid esté dentro del rango
        d_output_1[tid] = randomValue1;
        d_output_2[tid] = randomValue2;
        d_output_3[tid] = randomValue3;
        d_output_4[tid] = randomValue4;
    }
}

int main() {
    const int N = 1024; 
    const int numArrays = 4; 
    float *h_arrays[numArrays];
    float *d_arrays[numArrays];
    const int arraySize = N * sizeof(float);

    for (int i = 0; i < numArrays; ++i) {
        h_arrays[i] = new float[N];
        cudaMalloc((void**)&d_arrays[i], arraySize);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Llamar al kernel
    initializeRandomPositions<<<blocksPerGrid, threadsPerBlock>>>(d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], time(NULL), 0.0f, 1.0f);

    // Copiar los resultados al host
    for (int i = 0; i < numArrays; ++i) {
        cudaMemcpy(h_arrays[i], d_arrays[i], arraySize, cudaMemcpyDeviceToHost);
    }

    // Imprimir un ejemplo de valores
    for (int i = 0; i < numArrays; ++i) {
        std::cout << "Array " << i << " values:\n";
        for (int j = 0; j < 10; ++j) { // Solo imprimir los primeros 10 valores
            std::cout << h_arrays[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Liberar memoria
    for (int i = 0; i < numArrays; ++i) {
        cudaFree(d_arrays[i]);
        delete[] h_arrays[i];
    }

    return 0;
}
