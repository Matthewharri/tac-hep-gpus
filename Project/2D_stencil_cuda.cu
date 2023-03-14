#include <stdio.h>
#include <algorithm>


#define BLOCK_SIZE 32
#define DSIZE 512
#define RADIUS 3

__global__ void stencil_2D(int *in, int *out){

    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];
    int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex_x = threadIdx.x + RADIUS;
    int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;
    int lindex_y = threadIdx.y + RADIUS;

    //read input elements into shared memory
    int size = DSIZE + 2 * RADIUS;
    temp[lindex_x][lindex_y] = in[gindex_y + size * gindex_x];

    //Take care of boundary conditions
    if (threadIdx.x < RADIUS){
        temp[lindex_x - RADIUS][lindex_y] = in[gindex_y + size * (gindex_x - RADIUS)];
        temp[lindex_x + BLOCK_SIZE][lindex_y] = in[gindex_y + size * (gindex_x + BLOCK_SIZE)];
    }
    if (threadIdx.y < RADIUS){
        temp[lindex_x][lindex_y - RADIUS] = in[gindex_y - RADIUS + size * gindex_x];
        temp[lindex_x][lindex_y + BLOCK_SIZE] = in[gindex_y + BLOCK_SIZE + size * gindex_x];
    }

    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++){
        result += temp[lindex_x + offset][lindex_y];
        result += temp[lindex_x][lindex_y + offset];
        }
    result -= temp[lindex_x][lindex_y];

    out[gindex_y + size * gindex_x] = result;
}

__global__ void mult_square_matrix(int *a, int *b, int *c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // Make sure we are not out of range
    if ((idx < size) && (idy < size)) {
        int temp = 0;
        for (int i = 0; i < size; i++){
            temp += a[idy*size+i] * b[i*size+idx];
        }
        c[idy*size+idx] = temp;                    
    }

    if (idx == 1000 && idy == 1000){
        printf("c[0][0] = %d", c[1000]);
    }
}

int main(void){
    int *A, *B, *C, *D, *E;
    int *d_A, *d_B, *d_C, *d_D, *d_E;

    int size = (DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS) * sizeof(int);
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);
    D = (int *)malloc(size);
    E = (int *)malloc(size);

    //Fill arrays with integers
    for(int i = 0; i < (DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS); i++){
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = A[i];
        D[i] = B[i];
        E[i] = 0;
    }

    //Allocate memory on the device for stencil operation
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    cudaMalloc((void **)&d_D, size);
    cudaMalloc((void **)&d_E, size);

    // //Copy from host to device for the 2D stencil operation
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E, size, cudaMemcpyHostToDevice);

    //Set up the execution configuration for the 2D stencil operation
    int gridSize = (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(gridSize, gridSize);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    //Launch 2D stencil kernel for both A and B
    stencil_2D<<<dimGrid, dimBlock>>>(d_A + RADIUS + RADIUS * (DSIZE + 2 * RADIUS), d_C + RADIUS + RADIUS * (DSIZE + 2 * RADIUS));
    stencil_2D<<<dimGrid, dimBlock>>>(d_B + RADIUS + RADIUS * (DSIZE + 2 * RADIUS), d_D + RADIUS + RADIUS * (DSIZE + 2 * RADIUS));
    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(D, d_D, size, cudaMemcpyDeviceToHost);

    //d_C and d_D already exist on device, so we can keep these allocated on the device without copying back
    dim3 dimBlock2(32,32);
    dim3 dimGrid2((DSIZE + 2 * RADIUS), (DSIZE + 2 * RADIUS));
    mult_square_matrix<<<dimGrid2, dimBlock2>>>(d_C, d_D, d_E, DSIZE+2*RADIUS);

    //copy d_E back to host
    cudaMemcpy(E, d_E, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < DSIZE + 2 * RADIUS; ++i) {
		for (int j = 0; j < DSIZE + 2 * RADIUS; ++j) {

			if (i < RADIUS || i >= DSIZE + RADIUS) {
				if (C[j+i*(DSIZE + 2 * RADIUS)] != A[j+i*(DSIZE + 2 * RADIUS)]) {
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*(DSIZE + 2 * RADIUS)], A[j+i*(DSIZE + 2 * RADIUS)]);
					return -1;
				}
			}
        }
    }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);

    //Free host memory
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);

}

