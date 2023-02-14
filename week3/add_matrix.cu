#include <stdio.h>


const int DSIZE_X = 256;
const int DSIZE_Y = 256;

__global__ void add_matrix(const float *A, const float *B, float *C)
{
    //FIXME:
    // Express in terms of threads and blocks
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    // Add the two matrices - make sure you are not out of range
    if (idx <  DSIZE_X && idy < DSIZE_Y ){
        C[idy * DSIZE_Y + idx] =  A[idy * DSIZE_Y + idx] + B[idy * DSIZE_Y + idx];

    }
}

int main()
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Create and allocate memory for host and device pointers 
    h_A = new float[DSIZE_X * DSIZE_Y];
    h_B = new float[DSIZE_X * DSIZE_Y];
    h_C = new float[DSIZE_X * DSIZE_Y];

    cudaMalloc(&d_A, DSIZE_X * DSIZE_Y * sizeof(float));
    cudaMalloc(&d_B, DSIZE_X * DSIZE_Y * sizeof(float));
    cudaMalloc(&d_C, DSIZE_X * DSIZE_Y * sizeof(float));

    // Fill in the matrices
    // FIXME
    for (int i = 0; i < DSIZE_X; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_C[i] = 0;
    }
;
    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE_X * DSIZE_Y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE_X * DSIZE_Y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, DSIZE_X * DSIZE_Y * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    // dim3 is a built in CUDA type that allows you to define the block 
    // size and grid size in more than 1 dimentions
    // Syntax : dim3(Nx,Ny,Nz)
    // dim3 blockSize(blockIdx.x, blockIdx.y,1); 
    // dim3 gridSize(DSIZE_X/blockSize.x,DSIZE_Y/blockSize.y,1); 
    dim3 blockSize(32,32,1);
    dim3 gridSize(DSIZE_X/blockSize.x,DSIZE_Y/blockSize.y,1);
    
    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C);

    // Copy back to host 
    cudaMemcpy(h_C, d_C, DSIZE_X * DSIZE_Y * sizeof(float), cudaMemcpyDeviceToHost);
    // Print and check some elements to make the addition was succesfull
    printf("h_A[0] = %f h_B[0] = %f h_C[0] = %f, h_A[1] = %f h_B[1] = %f h_C[1] = %f, h_A[96] = %f h_B[96] = %f h_C[96] = %f ", h_A[0], h_B[0], h_C[0], h_A[1], h_B[1], h_C[1], h_A[96], h_B[96], h_C[96]);
    // Free the memory     

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}