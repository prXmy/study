#include <iostream>
#include <cuda_runtime.h>
using namespace std;
__global__ void matrixVectorMultiplication(int* A, int* B, int* C, int N, int M, int K)
{
    // Вычисление индекса текущего элемента матрицы C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K)
    {
        int value = 0;
        for (int k = 0; k < M; k++)
        {
            value += A[row * M + k] * B[k * K + col];
        }
        C[row * K + col] = value;
    }
}
void Multiplication(int N, int M, int K, int *aMatrix, int *bMatrix, int *product) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            product[i*N + j] = 0;
            for (int z = 0; z < K; z++) {
                product[i*N +j] += aMatrix[i*N +z] * bMatrix[z*K +j];
            }
            cout << product[i*N +j] << " ";
        }
        cout << endl;
    }
}
int main()
{
    // Создаем события для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N = 3, M =3 , K = 3; // Размеры матриц (N x M) и (M x K)

    // Проверка возможности перемножения матриц
    if (M <= 0 || N <= 0 || K <= 0)
    {
        std::cout << "Ошибка: Некорректные размеры матриц\n";
        return 1;
    }

    // Выделение и заполнение памяти для матриц на хосте
    int* h_A = new int[N * M];
    int* h_B = new int[M * K];
    int* h_C = new int[N * K];

    // Заполнение матриц случайными значениями
    for (int i = 0; i < N * M; i++)
    {
        h_A[i] = rand() % 10; // Ограничим числа до 9 для простоты
    }

    for (int i = 0; i < M * K; i++)
    {
        h_B[i] = rand() % 10; // Ограничим числа до 9 для простоты
    }

    // Выделение памяти на устройстве
    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, N * M * sizeof(int));
    cudaMalloc(&d_B, M * K * sizeof(int));
    cudaMalloc(&d_C, N * K * sizeof(int));

    // Копирование данных с хоста на устройство
    cudaMemcpy(d_A, h_A, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, M * K * sizeof(int), cudaMemcpyHostToDevice);

    // Задаем конфигурацию блоков и потоков для вычислений
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Запуск ядра умножения матриц
    cudaEventRecord(start);
    matrixVectorMultiplication <<<numBlocks, threadsPerBlock>>> (d_A, d_B, d_C, N, M, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Копирование результата с устройства на хост
    cudaMemcpy(h_C, d_C, N * K * sizeof(int), cudaMemcpyDeviceToHost);

    // Вывод результата
    std::cout << "Результат умножения матриц:\n";
    // Выводим только первые 5 элементов для наглядности
    std::cout<<"Матрица 1: \n";
    for(int i = 0 ; i<N;i++){
        for (int j = 0; j<M;j++) {
            std::cout<<h_A[i*N +j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\nМатрица 2: \n";
    for(int i = 0 ; i<M;i++){
        for (int j = 0; j<K;j++) {
            std::cout<<h_B[i*M +j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\nРезультат: \n";
    for(int i = 0 ; i<N;i++){
        for (int j = 0; j<K;j++) {
            std::cout<<h_C[i*N +j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"Результат выполнения на GPU: "<<milliseconds<<" ms\n";
    // Освобождение памяти
    cudaEventRecord(start);
    Multiplication(N,M,K,h_A,h_B,h_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"Результат выполнения на CPU: "<<milliseconds<<" ms\n";
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}