#ifndef config_h
#define config_h

#include <alpaka/alpaka.hpp>

// index type
using Idx = uint32_t;

// dimensions
using Dim0D = alpaka::DimInt<0u>;
using Dim1D = alpaka::DimInt<1u>;
using Dim2D = alpaka::DimInt<2u>;
using Dim3D = alpaka::DimInt<3u>;

// vectors 
template <typename TDim>
using Vec = alpaka::Vec<TDim, uint32_t>;
using Scalar = Vec<Dim0D>;
using Vec1D = Vec<Dim1D>;
using Vec2D = Vec<Dim2D>;
using Vec3D = Vec<Dim3D>;

// work division
template <typename TDim>
using WorkDiv = alpaka::WorkDivMembers<TDim, uint32_t>;
using WorkDiv1D = WorkDiv<Dim1D>;
using WorkDiv2D = WorkDiv<Dim2D>;
using WorkDiv3D = WorkDiv<Dim3D>;

// host types
using Host = alpaka::DevCpu;
using HostPlatform = alpaka::PltfCpu;

// different backends and device types

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
// CUDA backend
using Device = alpaka::DevCudaRt;
using Platform = alpaka::Pltf<Device>;
using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;
using Event = alpaka::Event<Queue>;

template <typename TDim>
using Acc = alpaka::AccGpuCudaRt<TDim, uint32_t>;
using Acc1D = Acc<Dim1D>;
using Acc2D = Acc<Dim2D>;
using Acc3D = Acc<Dim3D>;

#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
// HIP/ROCm backend
using Device = alpaka::DevHipRt;
using Platform = alpaka::Pltf<Device>;
using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;
using Event = alpaka::Event<Queue>;

template <typename TDim>
using Acc = alpaka::AccGpuHipRt<TDim, uint32_t>;
using Acc1D = Acc<Dim1D>;
using Acc2D = Acc<Dim2D>;
using Acc3D = Acc<Dim3D>;

#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
// CPU serial backend
using Device = alpaka::DevCpu;
using Platform = alpaka::Pltf<Device>;
using Queue = alpaka::Queue<Device, alpaka::Blocking>;
using Event = alpaka::Event<Queue>;

template <typename TDim>
using Acc = alpaka::AccCpuSerial<TDim, uint32_t>;
using Acc1D = Acc<Dim1D>;
using Acc2D = Acc<Dim2D>;
using Acc3D = Acc<Dim3D>;

#else
// no backend specified
#error Please define one of ALPAKA_ACC_GPU_CUDA_ENABLED, ALPAKA_ACC_GPU_HIP_ENABLED, ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#endif

#endif  // config_h