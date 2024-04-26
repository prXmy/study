#include <iostream>
#include <cuda_runtime.h>

#define N 100000
__device__ double atomicAddDouble(double* address, double val){
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
__global__ void sumArrayInt(int* array)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid<N&&tid!=0){
        atomicAdd(&array[0], array[tid]);
    }
}
__global__ void sumArrayDouble(double * array, unsigned long long int* result,double * res)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid<N){
//        atomicAdd(result, __double_as_longlong(array[tid]));
//        *res = __longlong_as_double(*result);
        atomicAddDouble(res,array[tid]);
    }

}


int main()
{
    int* h_array = new int[N];
    int size = N * sizeof(int);

    int* d_array;
    cudaMalloc((void**)&d_array, size);

    for (int i = 0; i < N; i++)
    {
        h_array[i] = 1;
    }

    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    int numThreadsPerBlock = 1024;
//определяем необходимое количество блоков
    int numBlocks = (N+numThreadsPerBlock-1)/numThreadsPerBlock;

    sumArrayInt<<<numBlocks,numThreadsPerBlock>>>(d_array);
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);


    double *h_arrayD = new double [N];
    int sizeD = N * sizeof(double );

    double *d_arrayD;
    cudaMalloc((void **) &d_arrayD, sizeD);

    for (int i = 0; i < N; i++) {
        h_arrayD[i] = 1.0;
    }
    unsigned long long h_result = 0;
    unsigned long long *d_result;
    cudaMalloc((void **) &d_result, sizeof(unsigned long long));
    cudaMemcpy(d_result, &h_result, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    double h_res = 0.0;
    double *d_res;
    cudaMalloc((void **) &d_res, sizeof(double ));
    cudaMemcpy(d_res, &h_res, sizeof(double ), cudaMemcpyHostToDevice);

    cudaMemcpy(d_arrayD, h_arrayD, sizeD, cudaMemcpyHostToDevice);

    sumArrayDouble<<<numBlocks, numThreadsPerBlock>>>(d_arrayD, d_result,d_res);

    cudaMemcpy(h_arrayD, d_arrayD, sizeD, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_res, d_res, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    std::cout << "Сумма интового массива: " << h_array[0] << std::endl;
    std::cout << "Сумма дабл массива: ";
    printf("%f \n",h_res);

    delete[] h_array;
    cudaFree(d_array);
    delete[] h_arrayD;
    cudaFree(d_arrayD);
    cudaFree(d_result);
    cudaFree(d_res);

    return 0;
}