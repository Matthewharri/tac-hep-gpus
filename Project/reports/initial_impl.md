'''Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/nvtx_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain NV Tools Extension (NVTX) data.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/osrt_sum.py]...

 ** OS Runtime Summary (osrt_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)           Name
 --------  ---------------  ---------  ------------  -------------  --------  -----------  ------------  ---------------------
     92.6    4,706,607,597         58  81,148,406.8  100,153,265.0     4,790  100,360,747  36,587,443.0  poll
      7.1      360,805,532        673     536,115.2       51,461.0       780   39,620,678   1,783,326.6  ioctl
      0.1        3,782,085         57      66,352.4       10,891.0     2,410      945,929     186,396.6  open64
      0.1        3,544,527         32     110,766.5       25,760.5    15,360    2,127,891     369,645.0  mmap64
      0.0        2,075,900         10     207,590.0       74,735.5    28,880      977,750     296,433.8  sem_timedwait
      0.0        2,013,147         37      54,409.4        7,430.0     2,180    1,065,520     200,063.5  fopen
      0.0        1,670,687          4     417,671.8      346,238.5   240,583      737,627     220,880.2  pthread_create
      0.0          538,675         65       8,287.3          650.0       320      258,183      42,896.6  fcntl
      0.0          419,743         31      13,540.1        9,270.0     4,670      104,491      17,745.3  fclose
      0.0          411,163          6      68,527.2       16,080.0       440      217,422      94,408.9  fread
      0.0          346,762         21      16,512.5        9,430.0     3,270       95,931      20,312.5  mmap
      0.0           83,431          7      11,918.7        9,390.0     8,020       22,060       5,029.3  munmap
      0.0           71,280         23       3,099.1           90.0        70       69,280      14,426.9  fgets
      0.0           65,950          6      10,991.7       13,290.0     2,950       15,260       5,175.7  open
      0.0           54,260         11       4,932.7        5,130.0     1,040        8,120       2,502.5  write
      0.0           35,932          4       8,983.0        8,615.5     6,370       12,331       3,051.7  mprotect
      0.0           32,290         14       2,306.4        2,280.0       620        5,490       1,375.8  read
      0.0           27,450          2      13,725.0       13,725.0    12,330       15,120       1,972.8  socket
      0.0           13,900          1      13,900.0       13,900.0    13,900       13,900           0.0  pipe2
      0.0           13,510          1      13,510.0       13,510.0    13,510       13,510           0.0  connect
      0.0            5,070         64          79.2           50.0        50          390          58.9  pthread_mutex_trylock
      0.0            4,590          8         573.8          585.0       280          970         266.9  dup
      0.0            2,530          1       2,530.0        2,530.0     2,530        2,530           0.0  bind
      0.0              900          1         900.0          900.0       900          900           0.0  listen

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/cuda_api_sum.py]...

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)    Max (ns)     StdDev (ns)            Name
 --------  ---------------  ---------  ------------  -----------  ---------  -----------  -------------  ----------------------
     91.1      356,992,995          5  71,398,599.0  1,142,611.0    126,101  352,834,035  157,330,789.7  cudaMalloc
      7.5       29,416,070          8   3,677,008.8    554,845.5    524,885   25,120,899    8,665,894.2  cudaMemcpy
      0.8        3,283,872          1   3,283,872.0  3,283,872.0  3,283,872    3,283,872            0.0  cuLibraryLoadData
      0.5        2,074,540          5     414,908.0    327,863.0    316,013      752,098      188,926.1  cudaFree
      0.0           62,831          3      20,943.7      8,901.0      8,050       45,880       21,599.7  cudaLaunchKernel
      0.0            3,050          1       3,050.0      3,050.0      3,050        3,050            0.0  cuModuleGetLoadingMode

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/cuda_gpu_kern_sum.py]...

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                      Name
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  --------------------------------------------
     98.6       24,472,143          1  24,472,143.0  24,472,143.0  24,472,143  24,472,143          0.0  mult_square_matrix(int *, int *, int *, int)
      1.4          347,803          2     173,901.5     173,901.5     173,694     174,109        293.4  stencil_2D(int *, int *)

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/cuda_gpu_mem_time_sum.py]...

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)      Operation
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ------------------
     58.7        2,097,125      5  419,425.0  415,387.0   410,010   434,970      9,727.8  [CUDA memcpy HtoD]
     41.3        1,473,453      3  491,151.0  504,538.0   457,850   511,065     29,023.6  [CUDA memcpy DtoH]

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/cuda_gpu_mem_size_sum.py]...

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
     21.218      5     4.244     4.244     4.244     4.244        0.000  [CUDA memcpy HtoD]
     12.731      3     4.244     4.244     4.244     4.244        0.000  [CUDA memcpy DtoH]

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/openmp_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain OpenMP event data.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/opengl_khr_range_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain KHR Extension (KHR_DEBUG) data.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/opengl_khr_gpu_range_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain GPU KHR Extension (KHR_DEBUG) data.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/vulkan_marker_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain Vulkan Debug Extension (Vulkan Debug Util) data.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/vulkan_gpu_marker_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain GPU Vulkan Debug Extension (GPU Vulkan Debug markers) data.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/dx11_pix_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain DX11 CPU debug markers.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/dx12_gpu_marker_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain DX12 GPU debug markers.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/dx12_pix_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain DX12 CPU debug markers.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/wddm_queue_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain WDDM context data.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/um_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/um_total_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/um_cpu_page_faults_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [2D_stencil_gpu.sqlite] with [/opt/nvidia/nsight-systems/2023.1.2/host-linux-x64/reports/openacc_sum.py]...
SKIPPED: 2D_stencil_gpu.sqlite does not contain OpenACC event data.'''