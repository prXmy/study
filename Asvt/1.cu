#include<iostream>
using namespace std;
int main()
{
    int device_count; //количество GPU устройств
    cudaGetDeviceCount(&device_count); //функция для определения числа установленных в
   // системе видеокарт с поддержкой технологии CUDA
//Для определения свойств видеокарты используется функция
//    cudaGetDeviceProperties();
// принимающая в качестве параметра номер видеокарты и
//возвращающая в структуре cudaDeviceProp интересующие значения
    cudaDeviceProp dp; //структура свойств
//cudaGetDeviceProperties(&dp, 0); Определение параметров GPU с номером 0
    cout << "CUDA device count: " << device_count << "\n";
    for (int i=0; i<device_count; i++)
    {
        cudaGetDeviceProperties(&dp, i);
        //имя устройства dp.name
        //тактовая частота ядра dp.clockRate
        //общий объем графической памяти dp.totalGlobalMem
        //объем памяти констант dp.totalConstMem
        //число потоковых мультипроцессоров dp.multiProcessorCount
        //объем разделяемой памяти в пределах блока dp.sharedMemPerBlock
        //число регистров в пределах блока dp.regsPerBlock
        //размер WARP’а (нитей в варпе) dp.warpSize
        //максимально допустимое число нитей в блоке dp.maxThreadsPerBlock
        //максимальную размерность при конфигурации нитей в блоке (максимальное
        //                                  количество нитей по трем измерениям) dp.maxThreadsDim[ ]
        //максимальную размерность при конфигурации блоков в сетке (максимальные
        //                                     размеры сетки по трем измерениям) dp.maxGridSize[ ]
        cout<<"\n";
        cout << i << ": " <<"имя устройства "<<dp.name<<"\nтактовая частота ядра "<<dp.clockRate<<"\nобщий объем графической памяти "<<dp.totalGlobalMem;
                            cout<<"\nобъем памяти констант "<<dp.totalConstMem;
                            cout<<"\nчисло потоковых мультипроцессоров "<<dp.multiProcessorCount;
                            cout<<"\nобъем разделяемой памяти в пределах блока "<<dp.sharedMemPerBlock;
                            cout<<"\nчисло регистров в пределах блока "<<dp.regsPerBlock;
                            cout<<"\nразмер WARP’а (нитей в варпе) "<<dp.warpSize;
                            cout<<"\nмаксимально допустимое число нитей в блоке "<<dp.maxThreadsPerBlock;
                            cout<<"\nмаксимальную размерность при конфигурации нитей в блоке (максимальное количество нитей по трем измерениям) [ "<<dp.maxThreadsDim[0]<<" , "<<dp.maxThreadsDim[1]<<" , "<<dp.maxThreadsDim[2]<<" ]";
                            cout<<"\nмаксимальную размерность при конфигурации блоков в сетке (максимальные размеры сетки по трем измерениям) [ "<<dp.maxGridSize[0]<<" , "<<dp.maxGridSize[1]<<" , "<<dp.maxGridSize[2]<<" ]\n";
    }
    return 0;
}