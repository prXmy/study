
#define N 1000000//количество элементов массивов
#define numThreadsPerBlock 1024 // количество нитей на один блок
//ядро, которое выполняет скалярное произведение
#include "iostream"
#include <cuda_runtime.h>
using namespace std;
__global__ void kernel( int *a, int *b, int *c )
{
    unsigned long long int tid = threadIdx.x + blockIdx.x * blockDim.x;
//массивы в разделяемой памяти
    __shared__ int tempA[numThreadsPerBlock];
    __shared__ int tempB[numThreadsPerBlock];
//копирование из глобальной в разделяемую память
if(tid<N) {
    tempA[threadIdx.x] = a[tid];
    tempB[threadIdx.x] = b[tid];
    __syncthreads(); //синхронизация нитей в одном блоке

    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < numThreadsPerBlock; i++) {
            sum += tempA[i] * tempB[i];
        }
        atomicAdd(c, sum);
    }
} else{
    tempA[threadIdx.x] = 0;
    tempB[threadIdx.x] = 0;
}
}
__global__ void kernel2( int *a, int *b, int *c )
{
    unsigned long long int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<N){
        int temp = a[threadIdx.x] * b[threadIdx.x];
        atomicAdd(c, temp);
    }
}
int main( void )
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *a, *b, c =0;
    int *dev_a, *dev_b, *dev_c;
    int size = N * sizeof( int );
    cudaMalloc( (void**)&dev_a, size );
    cudaMalloc( (void**)&dev_b, size );
    cudaMalloc( (void**)&dev_c, sizeof( int ) );
    a = (int *)malloc( size );
    b = (int *)malloc( size );
    for(int i = 0 ; i<N;i++){
        a[i] = 1;
        b[i] =1;
    }
// копируем массивы на device
    cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );
//запускаем на выполнение kernel() с 1 блоком и N нитями
//определяем необходимое количество блоков
    int numBlocks = (N+numThreadsPerBlock-1)/numThreadsPerBlock;
//подставляем переменные numBlocks и numThreadsPerBlock в ядро
    cudaEventRecord(start);
    kernel<<<numBlocks,numThreadsPerBlock>>>( dev_a, dev_b, dev_c );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
//при необходимости можно добавить синхронизацию cudaDeviceSynchronize(), для ожидания
   // завершения работы на device
//копируем результат работы device на host
    cudaMemcpy( &c, dev_c, sizeof( int ) , cudaMemcpyDeviceToHost );
    cout<<"1 задание \nСкалярное произведение: "<<c<<endl;
    cout<<"Затраченное время на 1 000 000 элементов: "<< milliseconds<<" ms"<<endl;
    c =0;
    cudaMemcpy( dev_c, &c, sizeof(int), cudaMemcpyHostToDevice );
    cudaEventRecord(start);
    kernel2<<<numBlocks,numThreadsPerBlock>>>( dev_a, dev_b, dev_c );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy( &c, dev_c, sizeof( int ) , cudaMemcpyDeviceToHost );
    cout<<"2 задание \nСкалярное произведение: "<<c<<endl;
    cout<<"Затраченное время на 1 000 000 элементов: "<< milliseconds<<" ms"<<endl;
    free( a );
    free( b );
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
    return 0;
}