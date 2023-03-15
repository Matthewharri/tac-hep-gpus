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
}

bool check_matrix_mult(int *a, int *b, int *c, int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            int temp = 0;
            for (int k = 0; k < size; k++){
                temp += a[i*size+k] * b[k*size+j];
            }
            if (temp != c[i*size+j]){
                printf("Error! at %d, %d. Expected %d, got %d", i, j, temp, c[i*size+j]);
                return false;
            }
        }
    }
    return true;
}


int main(void){
    int *A, *B, *C, *D, *E;

    cudaMallocManaged(&A, (DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS) * sizeof(int));
    cudaMallocManaged(&B, (DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS) * sizeof(int));
    cudaMallocManaged(&C, (DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS) * sizeof(int));
    cudaMallocManaged(&D, (DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS) * sizeof(int));
    cudaMallocManaged(&E, (DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS) * sizeof(int));

    //Fill arrays with integers
    for(int i = 0; i < (DSIZE + 2 * RADIUS) * (DSIZE + 2 * RADIUS); i++){
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = A[i];
        D[i] = B[i];
        E[i] = 0;
    }

        //Set up the execution configuration for the 2D stencil operation
    int gridSize = (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(gridSize, gridSize);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    //Launch 2D stencil kernel for both A and B
    stencil_2D<<<dimGrid, dimBlock>>>(A + RADIUS + RADIUS * (DSIZE + 2 * RADIUS), C + RADIUS + RADIUS * (DSIZE + 2 * RADIUS));
    cudaDeviceSynchronize();
    stencil_2D<<<dimGrid, dimBlock>>>(B + RADIUS + RADIUS * (DSIZE + 2 * RADIUS), D + RADIUS + RADIUS * (DSIZE + 2 * RADIUS));
    cudaDeviceSynchronize();

    //Set up the execution configuration for the matrix multiplication
    dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2((DSIZE + 2 * RADIUS), (DSIZE + 2 * RADIUS));
    mult_square_matrix<<<dimGrid2, dimBlock2>>>(C, D, E, DSIZE+2*RADIUS);
    cudaDeviceSynchronize();

    if(not check_matrix_mult(C,D,E,DSIZE + 2 * RADIUS)){
        printf("Matrix multiplication failed\n");
        exit(1);
    }
    else{
        printf("Matrix multiplication Successful\n");
    }

    for (int i = 0; i < DSIZE + 2 * RADIUS; ++i) {
		for (int j = 0; j < DSIZE + 2 * RADIUS; ++j) {

			if (i < RADIUS || i >= DSIZE + RADIUS) {
				if (C[j+i*(DSIZE + 2 * RADIUS)] != A[j+i*(DSIZE + 2 * RADIUS)]) {
					printf("Mismatch at index [%d,%d] for matric C, was: %d, should be: %d\n", i,j, C[j+i*(DSIZE + 2 * RADIUS)], A[j+i*(DSIZE + 2 * RADIUS)]);
					return -1;
				}
                if (D[j+i*(DSIZE + 2 * RADIUS)] != B[j+i*(DSIZE + 2 * RADIUS)]) {
                    printf("Mismatch at index [%d,%d] for matrix D, was: %d, should be: %d\n", i,j, D[j+i*(DSIZE + 2 * RADIUS)], B[j+i*(DSIZE + 2 * RADIUS)]);
                    return -1;
                }
			}
            else if (j < RADIUS || j >= DSIZE + RADIUS) {
                if (C[j+i*(DSIZE + 2 * RADIUS)] != A[j+i*(DSIZE + 2 * RADIUS)]) {
                    printf("Mismatch at index [%d,%d] for matrix C, was: %d, should be: %d\n", i,j, C[j+i*(DSIZE + 2 * RADIUS)], A[j+i*(DSIZE + 2 * RADIUS)]);
                    return -1;
                }
                if (D[j+i*(DSIZE + 2 * RADIUS)] != B[j+i*(DSIZE + 2 * RADIUS)]) {
                    printf("Mismatch at index [%d,%d] for matrix D, was: %d, should be: %d\n", i,j, D[j+i*(DSIZE + 2 * RADIUS)], B[j+i*(DSIZE + 2 * RADIUS)]);
                    return -1;
                }
            }
        }
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
    cudaFree(E);
}