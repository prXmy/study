//сложение двух векторов на GPU
#include <iostream>
#include <ctime>
using namespace std;
#define N 10000000 //используется как количество элементов в массивах, и как количество задач для GPU
//(каждый элемент массива обрабатывается на отдельной нити)
//макрос для отлова ошибок
static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        cout << cudaGetErrorString( err ) << " in file '" << file << "' at line " << line << endl;
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void add(int *a, int *b, int *c , int *res)
{
//обработать данные, находящиеся по этому индексу блока
    unsigned long long int tid = blockIdx.x * blockDim.x + threadIdx.x;;
//threadIdx - координаты нити в блоке нитей (threadIdx.x, threadIdx.y, threadIdx.z), значение не должно
  //  превышать 1023
//blockIdx - координаты блока нитей в сетке (blockIdx.x, blockIdx.y, blockIdx.z), значение не должно
  //  превышать 65535 по одному из измерений
//blockDim - размеры блока нитей (blockDim.x, blockDim.y, blockDim.z)
//gridDim - размеры сетки блоков нитей (gridDim.x, gridDim.y, gridDim.z)
    if(tid < N) {//необходимо чтобы не выйти за пределы массива, если потоков создано больше N
        if(a[tid]>b[tid])
            res[tid]=a[tid];
        else res[tid] = b[tid];
        if (res[tid]<c[tid])
        res[tid] = c[tid];
    }
}
int main(void)
{
//объявляем указатели на массивы в памяти CPU и GPU
    int *a, *b, *c , *res;
    int *dev_a, *dev_b, *dev_c , *dev_res;
//выделяем память на CPU под массивы 'a', 'b' и 'c'
    a = (int *) malloc (N * sizeof(int));
    b = (int *) malloc (N * sizeof(int));
    c = (int *) malloc (N * sizeof(int));
    res = (int *) malloc (N * sizeof(int));
//выделяем память на GPU под массивы dev_a, dev_b, dev_c
    HANDLE_ERROR(cudaMalloc( (void**)&dev_a, N * sizeof(int) ));
    HANDLE_ERROR(cudaMalloc( (void**)&dev_b, N * sizeof(int) ));
    HANDLE_ERROR(cudaMalloc( (void**)&dev_c, N * sizeof(int) ));
    HANDLE_ERROR(cudaMalloc( (void**)&dev_res, N * sizeof(int) ));
//заполняем массивы 'a' и 'b' на CPU
    srand(time(0));
    for(int i=0; i<N; i++)
    {
        a[i] = rand()%10;
        b[i] = rand()%10;
        c[i] = rand()%10;
    }

//копируем массивы 'a' и 'b' на GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice));
//запускаем ядро на N блоках
//записываем количество нитей на один блок в numThreadsPerBlock (максимум нитей на блок вы уже
   // определяли в первом задании с помощью maxThreadsPerBlock, обычно это 1024)
    int numThreadsPerBlock = 1024;
//определяем необходимое количество блоков
    int numBlocks = (N+numThreadsPerBlock-1)/numThreadsPerBlock;
//подставляем переменные numBlocks и numThreadsPerBlock в ядро
    add<<<numBlocks,numThreadsPerBlock>>> (dev_a, dev_b, dev_c , dev_res) ;
 //   add<<<N,1>>>(dev_a, dev_b, dev_c);
//копируем массив 'c' с GPU на CPU
    cudaMemcpy(res, dev_res, N * sizeof(int), cudaMemcpyDeviceToHost);
//выводим результат
    for(int i=0; i<5; i++)
    {
        cout <<i+1<<" : a: "<< a[i] << "   b: " << b[i] << "   c: " << c[i] <<"   max: "<<res[i]<<endl;
    }
    for(int i=N-5; i<N; i++)
    {
        cout <<i+1<<" : a: "<< a[i] << "   b: " << b[i] << "   c: " << c[i] <<"   max: "<<res[i]<<endl;
    }
//освобождаем память, выделенную на CPU
    free( a );
    free( b );
    free( c );
    free( res );
//освобождаем память, выделенную на GPU
    HANDLE_ERROR(cudaFree( dev_a ));
            HANDLE_ERROR(cudaFree( dev_b ));
            HANDLE_ERROR(cudaFree( dev_c ));
            HANDLE_ERROR(cudaFree( dev_res ));
    return 0;
}