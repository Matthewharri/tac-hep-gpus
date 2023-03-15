all:
	g++ -std=c++11 2D_stencil_cpu.cpp -o 2D_stencil_cpu
	nvcc 2D_stencil_cuda.cu -o 2D_stencil_gpu
	nvcc 2D_stencil_cuda_managed.cu -o 2D_stencil_gpu_managed
	nvcc 2D_stencil_cuda_async.cu -o 2D_stencil_gpu_async