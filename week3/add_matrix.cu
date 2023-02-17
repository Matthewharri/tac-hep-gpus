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
    for (int i = 0; i < DSIZE_X*DSIZE_Y; i++) {
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
    dim3 blockSize(32,32,1);
    dim3 gridSize(DSIZE_X/blockSize.x,DSIZE_Y/blockSize.y,1);
    
    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C);

    // Copy back to host 
    cudaMemcpy(h_C, d_C, DSIZE_X * DSIZE_Y * sizeof(float), cudaMemcpyDeviceToHost);
    // Print and check some elements to make the addition was succesfull
    // First I will make the host arrays into 2D matrices

    float h_A_matrix[DSIZE_X][DSIZE_Y];
    float h_B_matrix[DSIZE_X][DSIZE_Y];
    float h_C_matrix[DSIZE_X][DSIZE_Y];

    for (int i = 0; i < DSIZE_X; i++) {
        for (int j = 0; j < DSIZE_Y; j++) {
            h_A_matrix[i][j] = h_A[i * DSIZE_Y + j];
            h_B_matrix[i][j] = h_B[i * DSIZE_Y + j];
            h_C_matrix[i][j] = h_C[i * DSIZE_Y + j];
        }
    }
    // Print the first 4x4 elements of the matrices
    printf("\n");
    printf("Matrix A:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", h_A_matrix[i][j]);
            
        }
        printf("\n");
    }

    printf("\n");
    printf("Matrix B:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", h_B_matrix[i][j]);
            
        }
        printf("\n");
    }
    printf("\n");
    printf("Matrix C:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", h_C_matrix[i][j]);
            
        }
        printf("\n");
    }

    // Free the memory     

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}