#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    int Row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int Col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float temp = 0.0;

    for (int x = 0; x < (k-1 + TILE_SIZE)/TILE_SIZE ; ++x) {
        if ( (Row < m) && ( ( x*TILE_SIZE+ threadIdx.x)<k ) ){
            A_shared[threadIdx.y][threadIdx.x] = A[ Row*k + x*TILE_SIZE + threadIdx.x ];
        }else{
            A_shared[threadIdx.y][threadIdx.x] = 0;
        }

        if ( ( (x*TILE_SIZE + threadIdx.y)< k ) && (Col< n) ){
            B_shared[ threadIdx.y ][ threadIdx.x ] = B[ ( x*TILE_SIZE + threadIdx.y ) * n + Col];
        }else{
            B_shared[ threadIdx.y ][ threadIdx.x ] = 0;
        }

        __syncthreads();

        for (int y = 0; y < TILE_SIZE; ++y){
            temp += A_shared[ threadIdx.y ][y] * B_shared[y][ threadIdx.x ];
        }
        __syncthreads();

    }

    if (Row<m && Col<n)
    {
        C[Row*n+Col] = temp;
    }
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    if((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	    printf("unsupported value of alpha\n");
	    return;
    }
    if((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	    printf("unsupported value of beta\n");
	    return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE

    dim3 dimGrid((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,1);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<dimGrid,dimBlock>>>(m, n, k, A, B, C);
}


