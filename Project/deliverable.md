# Deliverable for Project

# Setup Environment
- First, we log into UWisc cluster and connect to a GPU node via '''ssh g38nXX''' where XX is the node number.
    - Once connected we run the following commands:
    '''export LD_LIBRARY_PATH=/usr/local/cuda/lib'''
    '''export PATH=$PATH:/usr/local/cuda/bin'''
    '''source scl_source enable devtoolset-11'''
- If we then change the the directory, tac-hep-gpus/Project we can run '''make''' which will compile all of the source code for the project.
- This will create executables for every part of the project

# C++ and CPU profiling
- The c++ only implementation is located in 2D_stencil_cpu.cpp, and can be run with the command '''./2D_stencil_cpu'''.
- Unfortunately, I could not get VTune to work, so I do not have a profile for this part of the project.

# Porting to CUDA (Profiling can be found in Projects/reports)
- For the explict memory copies part of the CUDA implmentation, this can be found in the file 2D_stencil_cuda.cu, and can be run using the command '''.2D_stencil_gpu'''.
- In the profiling of this implementation, the majory of the time (91.1%) of the cuda API calls is spent in cudaMalloc, with the next highest being CUDA Memcpy (7.5%),
  Everything else is rather negligible.
- When it comes to the Cuda kernels, the majority of the time is spent doing matrix multiplication at 98.6% of the time.
- The total run time of this implementation looks to take around 5 seconds in total to run.

- For the managed memory part of the CUDA implmentation, this can be found in the file 2D_stencil_cuda_managed.cu, and can be run using the command 
'''./2D_stencil_gpu_managed'''.
- In the profiling of this implementation, the majority of the time (96.5%) of the cuda API calls is spent in cudaMallocManaged, with the next highest being cudaDeviceSychronize (2.4%). Everything else is rather negligible.
- When it comes to the Cuda kernels, the time between matrix multiplication and the stencil are more comparable at 64.4% and 35.6% respectively.
- The total run time of this implementation looks to take around 2 seconds in total to run.

# Optimizing performance in CUDA
- For the non-default CUDA streams part of the CUDA implmentation, this can be found in the file 2D_stencil_cuda_async.cu, and can be run using the command 
'''./2D_stencil_gpu_async'''.
- In the profiling of this implementation, Most of the time is spent is two API calls at similar percentages, cudaHostAlloc(58.1%) and CudaDeviceSynchronize(41.0%).
  Everything else is pretty negligible.
- In regards to cuda kernels, we see that Matrix multiplication takes up the majority of the time at 98.1% with the stencil operation at 1.9%.
- The total run time of this implementation is around 2 seconds total.

- In comparison to the initial cuda implementation, this is significantly faster, with the total run time being around 2 seconds, compared to the initial 5 seconds.
  This is a 60% improvement in performance. Not only this, but we spend about 0.7 seconds in cuda API calls in this method compared to ~0.4 seconds in the initial implemntation, which is slower, but we gain a big speed up elsewhere.

# Making use of Alpaka
  - Switching from the other versions to Alpaka was not a ton of work, but did require very subtle changes.
    - The first thing that needed to be figured out was how to actually get a very simple example to compile while including an alpaka header library,
      Thankfully, Andrea left his alpaka (and boost library) on the cluster ready to go, so I was just able to include it in my Makefile when compiling.
    - In order to get the code to compile it did require a few more steps, namely specifying the type of device I want to compile on, 
    - The next thing I had to do was be able to rewrite the stencil and matrix multiplication making use of alpaka. This wasn't terribly hard but definitely took some thinking. The first part of it was straightforward which just required throwing the kernel into a struct, and inside of a templated operator function. What was really the most difficult part of rewriting the kernels was figuring out how the indexing worked. After a while I was able to find out how to actually access indexes, but to determine the x,y index took me a little bit to reason out that it made sense. But the rest of writing the kernels wasn't much work as it was pretty similar to what I had done before.
    - I then had to also allocate memory using alpaka but this was pretty straight forward.
    - I struggled for a while on figuring out how to actually get the kernels to be executed. The main problem came from the line '''auto div = make_workdiv<Acc1D>(1024, 1024);'''. It wasn't very clear to me what this did, and my kernels wouldn't finish executing, it would just get part way through and stop. I did figure out that it all boiled down to this line and was able to make it work.

