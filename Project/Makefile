CUDA_BASE   = /usr/local/cuda
BOOST_BASE  = /afs/hep.wisc.edu/home/abocci/public/boost
ALPAKA_BASE = /afs/hep.wisc.edu/home/abocci/public/alpaka

CXX  := g++
NVCC := $(CUDA_BASE)/bin/nvcc

CXXFLAGS       := -std=c++17 -O2 -g -I$(BOOST_BASE)/include -I$(ALPAKA_BASE)/include
CXX_HOST_FLAGS := -pthread
CXX_CUDA_FLAGS := --expt-relaxed-constexpr

all:
	g++ -std=c++17 2D_stencil_cpu.cpp -o 2D_stencil_cpu
	nvcc 2D_stencil_cuda.cu -o 2D_stencil_gpu
	nvcc 2D_stencil_cuda_managed.cu -o 2D_stencil_gpu_managed
	nvcc 2D_stencil_cuda_async.cu -o 2D_stencil_gpu_async
	$(NVCC) -x cu -ccbin $(CXX) $(CXXFLAGS) $(CXX_CUDA_FLAGS) -Xcompiler '$(CXX_HOST_FLAGS)' 2D_stencil_alpaka.cpp -DALPAKA_ACC_GPU_CUDA_ENABLED -o 2D_stencil_alpaka
