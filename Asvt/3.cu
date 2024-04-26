//сложение векторов на GPU
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

#define N 10 //используется как количество элементов в массивах, и как количество задач для GPU
//(каждый элемент массива обрабатывается на отдельной нити)

__global__ void add(int *a, int *b, int *c)
{
//обработать данные, находящиеся по этому индексу
    int tid = blockIdx.x;
    if(tid < N)
        c[tid] = a[tid] + b[tid];
}
int main(void){
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

//выделяем память на GPU под массивы a,b,c
    HANDLE_ERROR( cudaMalloc((void**)&dev_a, N * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_b, N * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_c, N * sizeof(int)) );//// ошибка была здесь, память была выделена под 1 int а не под N

//заполняем массивы 'a' и 'b' на CPU
    for(int i=0; i<N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }
//копируем массивы 'a' и 'b' на GPU
    HANDLE_ERROR( cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice) );
//запускаем ядро на N блоках
    add<<<N,1>>>(dev_a, dev_b, dev_c);
//копируем массив 'c' с GPU на CPU
    HANDLE_ERROR( cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost) );
//выводим результат
    for(int i=0; i<N; i++)
    {
        cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
    }
//освобождаем память, выделенную на GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );
    return 0;
}