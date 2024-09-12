#include <stdio.h>

#define BLOCK_SIZE 512

__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{
    
    extern __shared__ int histogram_prv[];
    unsigned int m = (num_bins - 1 / blockDim.x) + 1;     // calculates the number of iterations needed to initialize the shared memory array.
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int str = blockDim.x * gridDim.x;

    
    if ( threadIdx.x < num_bins) {
        for (unsigned int j = 0; j < m && ( threadIdx.x + (j) * blockDim.x ) < num_bins; j++){
            histogram_prv[ threadIdx.x + j * blockDim.x ] = 0;
        }
    }
    __syncthreads();
  
    while (i < num_elements) {
        atomicAdd( &(histogram_prv[input[i]]), 1 );
        i = i + str;
    }
    __syncthreads();

    if ( threadIdx.x < num_bins ) {
        for (unsigned int j = 0; j <= m && ( threadIdx.x + (j) * blockDim.x) < num_bins; j++){
            atomicAdd(&(bins[threadIdx.x + j * blockDim.x]), histogram_prv[threadIdx.x + j * blockDim.x]);
        }
    }
    
}

void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

    /*************************************************************************/
    //INSERT CODE HERE  
    dim3 dim_grid( ((num_elements - 1) / BLOCK_SIZE) + 1, 1, 1 );
    dim3 dim_block( BLOCK_SIZE, 1, 1 );
  
    int histogram_prv_size = num_bins * ( sizeof(int) );
    histo_kernel<<<dim_grid, dim_block, histogram_prv_size>>>(input, bins, num_elements, num_bins);

	/*************************************************************************/

}


