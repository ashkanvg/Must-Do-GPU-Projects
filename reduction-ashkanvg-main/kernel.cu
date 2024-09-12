/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512


__global__ void naiveReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // NAIVE REDUCTION IMPLEMENTATION
    __shared__ float partialSum[2*BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    // first 
    if(start + t < size){
        partialSum[t] = in[start + t];
    }else{
        partialSum[t] = 0.0;
    }

    // second
    if(start + blockDim.x + t < size){
        partialSum[blockDim.x + t] = in[start + blockDim.x + t];
    }else{
        partialSum[blockDim.x + t] = 0.0;
    }

    // sum
    for (unsigned int stride = 1; stride <= blockDim.x;  stride *= 2){
        __syncthreads();
        if (t % stride == 0)
            partialSum[2*t]+= partialSum[2*t+stride];
    }
    if(t == 0)
        out[blockIdx.x] = partialSum[0];

}

__global__ void optimizedReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // OPTIMIZED REDUCTION IMPLEMENTATION

    __shared__ float partialSum[2*BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    // first 
    if(start + t < size){
        partialSum[t] = in[start + t];
    }else{
        partialSum[t] = 0.0;
    }

    // second
    if(start + blockDim.x + t < size){
        partialSum[blockDim.x + t] = in[start + blockDim.x + t];
    }else{
        partialSum[blockDim.x + t] = 0.0;
    }
    
    __syncthreads();
    for (unsigned int stride = blockDim.x; stride >= 1;  stride /= 2){
        if (t < stride)
            partialSum[t] += partialSum[t+stride];
            __syncthreads();
    }
    if(t == 0)
        out[blockIdx.x] = partialSum[0];

}
