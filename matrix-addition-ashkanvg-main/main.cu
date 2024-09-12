#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t mat_sz;
    unsigned matDim;
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        matDim = 1000;
    } else if (argc == 2) {
        matDim = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./mat-add                # All matrices are 1000 x 1000"
      "\n    Usage: ./mat-add <m>            # All matrices are m x m"
      "\n");
        exit(0);
    }
   
    mat_sz = matDim*matDim;

    A_h = (float*) malloc( sizeof(float)*mat_sz );
    for (unsigned int i=0; i < mat_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*mat_sz );
    for (unsigned int i=0; i < mat_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*mat_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matDim, matDim, matDim, matDim, matDim, matDim);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE
    // int size = n * sizeof(float);    defined in the beginning of code : mat_sz
    int size_malloc = mat_sz * sizeof(float);
    // float *A_d, *B_d, *C_d;          defined in the beginning of code
    cudaMalloc((void **) &A_d, size_malloc);
    cudaMalloc((void **) &B_d, size_malloc);
    cudaMalloc((void **) &C_d, size_malloc);

    /*************************************************************************/
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);
	
    /*************************************************************************/
    //INSERT CODE HERE
    cudaMemcpy(A_d, A_h, size_malloc, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_malloc, cudaMemcpyHostToDevice);


    /*************************************************************************/
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard mat-add interface ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    basicMatAdd(matDim, A_d, B_d, C_d);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE
    cudaMemcpy(C_h, C_d, size_malloc, cudaMemcpyDeviceToHost);   



    /*************************************************************************/
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results...\n"); fflush(stdout);

    verify(A_h, B_h, C_h, matDim);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    /*************************************************************************/
    //INSERT CODE HERE
    cudaFree(A_d); 
    cudaFree(B_d);
    cudaFree(C_d);

    
    /*************************************************************************/
    return 0;
}

