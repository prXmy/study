#include <iostream>
#include <cuda_runtime.h>

#define SIZE 66666666 // Размер блока памяти
#define int long long
using namespace std;
signed main() {
    int size = SIZE * sizeof(int);
    cout<<"Передается "<<size/1024/1024<<" Mb\n";
    // Выделяем память на хосте и на устройстве
    int* host_buffer_src = (int*)malloc(SIZE * sizeof(int));
    int* host_buffer_dest = (int*)malloc(SIZE * sizeof(int));
    int* device_buffer_src;
    cudaMalloc(&device_buffer_src, SIZE * sizeof(int));
    int* device_buffer_dest;
    cudaMalloc(&device_buffer_dest, SIZE * sizeof(int));

    // Создаем события для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Инициализируем исходный буфер на хосте
    for (int i = 0; i < SIZE; i++) {
        host_buffer_src[i] = i;
    }

    // Копируем данные между двумя буферами в оперативной памяти (Host -> Host)
    cudaEventRecord(start);
    memcpy(host_buffer_dest, host_buffer_src, SIZE * sizeof(int));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float ps =((size / milliseconds)/1024)/1024;
    cout<<"Host -> Host  "<<ps<<"mb/s"<<endl;
    // Копируем данные из хоста в устройство (Host -> Device) с использованием обычной памяти
    cudaEventRecord(start);
    cudaMemcpy(device_buffer_src, host_buffer_src, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    ps =((size / milliseconds)/1024)/1024;
    cout<<"Host -> Device  "<<ps<<"mb/s"<<endl;
    // Копируем данные из устройства на хост (Device -> Host) с использованием обычной памяти
    cudaEventRecord(start);
    cudaMemcpy(host_buffer_dest, device_buffer_src, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    ps =((size / milliseconds)/1024)/1024;
    cout<<"Device -> Host  "<<ps<<"mb/s"<<endl;
    // Копируем данные из хоста в устройство (Host -> Device) с использованием pagelocked памяти
    cudaHostRegister(host_buffer_src, SIZE * sizeof(int), cudaHostRegisterDefault);
    cudaEventRecord(start);
    cudaMemcpy(device_buffer_src, host_buffer_src, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    ps =((size / milliseconds)/1024)/1024;
    cout<<"Host -> Device pagelocked "<<ps<<"mb/s"<<endl;
    cudaHostUnregister(host_buffer_src);

    // Копируем данные из устройства на хост (Device -> Host) с использованием pagelocked памяти
    cudaHostRegister(host_buffer_dest, SIZE * sizeof(int), cudaHostRegisterDefault);
    cudaEventRecord(start);
    cudaMemcpy(host_buffer_dest, device_buffer_src, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    ps =((size / milliseconds)/1024)/1024;
    cout<<"Device -> Host pagelocked "<<ps<<"mb/s"<<endl;
    cudaHostUnregister(host_buffer_dest);

    ////Функция cudaHostRegister является одной из функций из библиотеки CUDA Runtime API и предназначена для
    /// регистрации области памяти на хосте (ЦПУ) как pagelocked

    // Копируем данные между двумя буферами в глобальной памяти видеокарты (Device -> Device)
    cudaEventRecord(start);
    cudaMemcpy(device_buffer_dest, device_buffer_src, SIZE * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    ps =((size / milliseconds)/1024)/1024;
    cout<<"Device -> Device  "<<ps<<"mb/s"<<endl;
    // Проверяем результаты копирования
    for (int i = 0; i < SIZE; i++) {
        if (host_buffer_src[i] != host_buffer_dest[i]) {
            std::cout << "Ошибка при копировании данных\n";
            break;
        }
    }

    // Освобождаем память
    free(host_buffer_src);
    free(host_buffer_dest);
    cudaFree(device_buffer_src);
    cudaFree(device_buffer_dest);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}