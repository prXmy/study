#include <iostream>
using namespace std;

//макрос для отлова ошибок
static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        cout << cudaGetErrorString( err ) << " in file '" << file << "' at line " << line << endl;
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define N 800

//1. Матрицы хранятся в виде одномерных массивов по строкам;
__global__ void addMatrix1(float *a, float *b, float *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int index = row * N + col;
        c[index] = a[index] + b[index];
    }
}

//2. Матрицы хранятся в виде двумерных массивов.
__global__ void addMatrix2(float (*a)[N], float (*b)[N], float (*c)[N]) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        c[row][col] = a[row][col] + b[row][col];
    }
}

int main(void){
////1
        float *a, *b, *c;
        float *dev_a, *dev_b, *dev_c;

        int size = N * N * sizeof(float);

        a = (float*)malloc(size);
        b = (float*)malloc(size);
        c = (float*)malloc(size);

        for (int i = 0; i < N * N; i++) {
            a[i] = 0.5;
            b[i] = 0.5;
        }

        HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
        HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
        HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

        HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));
//определяем необходимое количество блоков
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    addMatrix1<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c); ////dim3 это встроенная переменная CUDA, которая является
/// трехмерной индикаторной переменной, которая соответствует трем измерениям x, y и z соответственно. Используя эту
/// переменную, мы можем контролировать общее количество блоков и потоков, вызываемых программой
//копируем массив 'c' с GPU на CPU
    HANDLE_ERROR( cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost) );
//выводим результат
cout<<"Первый метод :\n";
    for(int i=0; i<5; i++)
    {
        cout <<i<<" : "<< a[i] << " + " << b[i] << " = " << c[i] << endl;
    }
    for(int i=N*N-5; i<N*N; i++)
    {
        cout <<i<<" : "<< a[i] << " + " << b[i] << " = " << c[i] << endl;
    }
    free( a );
    free( b );
    free( c );
//освобождаем память, выделенную на GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );


////2
    float a1[N][N], b1[N][N], c1[N][N];
    float (*dev_a1)[N], (*dev_b1)[N], (*dev_c1)[N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a1[i][j] = 0.5;
            b1[i][j] = 0.5;
        }
    }

    cudaMalloc((void**)&dev_a1, N * N * sizeof(float));
    cudaMalloc((void**)&dev_b1, N * N * sizeof(float));
    cudaMalloc((void**)&dev_c1, N * N * sizeof(float));

    cudaMemcpy(dev_a1, a1, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b1, b1, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock1(16, 16);
    dim3 dimGrid1((N + dimBlock1.x - 1) / dimBlock1.x, (N + dimBlock1.y - 1) / dimBlock1.y);

    addMatrix2<<<dimGrid1, dimBlock1>>>(dev_a1, dev_b1, dev_c1);

    cudaMemcpy(c1, dev_c1, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_c1);
    cout<<"Второй метод :\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cout<<c1[i][j]<<" ";
        }
        cout<<" ...... \n";
    }
    for (int i = N-3; i < N; i++) {
        cout<<"...... ";
        for (int j = N-3; j < N; j++) {
            cout<<c1[i][j]<<" ";
        }
        cout<<"\n";
    }

    return 0;
}