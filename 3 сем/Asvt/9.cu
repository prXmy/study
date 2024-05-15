#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define N 1000//количество элементов массивов

// Функция умножения матрицы на вектор на устройстве
__global__ void matrixVectorMul(int* matrix, int* vector, int* result)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        result[tid] = 0;

        for (int i = 0; i < N; i++)
        {
            result[tid] += matrix[tid * N + i] * vector[i];
        }
    }
}

int main()
{
    int* h_matrix = new int[N * N];
    int* h_vector = new int[N];
    int* h_result = new int[N];

    // Заполнение матрицы и вектора единицами
    for (int i = 0; i < N; i++)
    {
        h_vector[i] = 1;
        for (int j = 0; j < N; j++)
        {
            h_matrix[i * N + j] = 1;
        }
    }

    // Выделение памяти на устройстве
    int* d_matrix, * d_vector, * d_result;
    cudaMalloc(&d_matrix, N * N * sizeof(int));
    cudaMalloc(&d_vector, N * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(int));

    // Копирование данных с хоста на устройство
    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, N * sizeof(int), cudaMemcpyHostToDevice);

    // Запуск ядра умножения матрицы на вектор
    int numThreadsPerBlock = 1024;
//определяем необходимое количество блоков
    int numBlocks = (N+numThreadsPerBlock-1)/numThreadsPerBlock;
//подставляем переменные numBlocks и numThreadsPerBlock в ядро
    matrixVectorMul <<<numBlocks,numThreadsPerBlock>>> (d_matrix, d_vector, d_result);

    // Копирование результата с устройства на хост
    cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Вывод первых и последних пяти элементов результирующего вектора
    std::cout << "Первые пять элементов результирующего вектора:\n";
    for (int i = 0; i < 5; i++)
    {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Последние пять элементов результирующего вектора:\n";
    for (int i = N-5; i < N; i++)
    {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    // Освобождение памяти
    delete[] h_matrix;
    delete[] h_vector;
    delete[] h_result;

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return 0;
}