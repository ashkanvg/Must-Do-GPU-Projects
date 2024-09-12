#include <stdio.h>

#define TILE_SIZE 16

__global__ void matAdd(int dim, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (dim x dim) matrix
     *   where B is a (dim x dim) matrix
     *   where C is a (dim x dim) matrix
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i<dim && j<dim){
        int index = i*dim + j;
        C[index] = A[index] + B[index];
    } 
    /*************************************************************************/

}

void basicMatAdd(int dim, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE


    /*************************************************************************/
	
	// Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
	dim3 DimGrid((dim-1)/BLOCK_SIZE + 1, (dim-1)/BLOCK_SIZE + 1, 1);   
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);                      
    
    matAdd<<<DimGrid,DimBlock>>>(dim, A, B, C);
    /*************************************************************************/

}

