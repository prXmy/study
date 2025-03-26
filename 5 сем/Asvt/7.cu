#include <iostream>
#include <cuda_runtime.h>
#include <ctime>
#define ARRAY_SIZE 10

__global__ void findMinValue(int* array, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < ARRAY_SIZE) {
        atomicMin(result, array[tid]);
    }
}
__global__ void findMaxValue(int* array, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < ARRAY_SIZE) {
        atomicMax(result, array[tid]);
    }
}

int main() {
    srand(time(0));
    int h_array[ARRAY_SIZE];
    int* d_array, *d_min, *d_max;
    int h_min = INT_MAX;
    int h_max = INT_MIN;
    // Заполнение массива случайными числами
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_array[i] = rand() % 100; // Заполнение числами от 0 до 99
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // Выделение памяти на GPU
    cudaMalloc(&d_array, ARRAY_SIZE * sizeof(int));
    cudaMalloc(&d_min, sizeof(int));
    cudaMalloc(&d_max, sizeof(int));
    // Копирование массива на GPU
    cudaMemcpy(d_array, h_array, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Установим начальное значение для минимума
    cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice);
    // Вычисление минимального значения через atomicMin
    int numThreadsPerBlock = 1024;
//определяем необходимое количество блоков
    int numBlocks = (ARRAY_SIZE+numThreadsPerBlock-1)/numThreadsPerBlock;
//подставляем переменные numBlocks и numThreadsPerBlock в ядро
    findMinValue<<<numBlocks,numThreadsPerBlock>>>(d_array, d_min);
    findMaxValue<<<numBlocks,numThreadsPerBlock>>>(d_array, d_max);
    // Копирование результата на хост
    cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Минимальное значение: " << h_min << std::endl;
    std::cout << "Максимальное значение: " << h_max << std::endl;
    // Освобождение памяти на GPU
    cudaFree(d_array);
    cudaFree(d_min);
    cudaFree(d_max);
    return 0;
}