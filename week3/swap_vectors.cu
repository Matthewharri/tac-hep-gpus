#include <stdio.h>


const int DSIZE = 40960;
const int block_size = 256;
const int grid_size = DSIZE/block_size;


__global__ void vector_swap(float *d_A, float *d_B, float *d_C) {

    //FIXME:
    // Express the vector index in terms of threads and blocks
    int idx =  threadIdx.x + blockIdx.x * block_size;
    // Swap the vector elements - make sure you are not out of range
    if (idx < DSIZE) {
        d_C[idx] = d_B[idx];
        d_B[idx] = d_A[idx];
        d_A[idx] = d_C[idx];
    }
}


int main() {


    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    //allocated memory for host pointers
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];


    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_C[i] = 0;
    }

    //Print first/last 3 elements of the vectors
    printf("h_A[0] = %f h_B[0] = %f\n", h_A[0], h_B[0]);
    printf("h_A[1] = %f h_B[1] = %f\n", h_A[1], h_B[1]);
    printf("h_A[2] = %f h_B[2] = %f\n", h_A[2], h_B[2]);

    printf("h_A[40957] = %f h_B[40957] = %f\n", h_A[40957], h_B[40957]);
    printf("h_A[40958] = %f h_B[40958] = %f\n", h_A[40958], h_B[40958]);
    printf("h_A[40959] = %f h_B[40959] = %f\n", h_A[40959], h_B[40959]);
    printf("\n");
    
    // Allocate memory for device pointers 
    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * sizeof(float));

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    vector_swap<<<grid_size, block_size>>>(d_A, d_B, d_C);
    // Copy back to host 

    cudaMemcpy(h_A, d_A, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make sure swapping was successfull
    printf("h_A[0] = %f h_B[0] = %f\n", h_A[0], h_B[0]);
    printf("h_A[1] = %f h_B[1] = %f\n", h_A[1], h_B[1]);
    printf("h_A[2] = %f h_B[2] = %f\n", h_A[2], h_B[2]);

    printf("h_A[40957] = %f h_B[40957] = %f\n", h_A[40957], h_B[40957]);
    printf("h_A[40958] = %f h_B[40958] = %f\n", h_A[40958], h_B[40958]);
    printf("h_A[40959] = %f h_B[40959] = %f\n", h_A[40959], h_B[40959]);

    // Free the memory 
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
