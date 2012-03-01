//#include "stdafx.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
//#include <cuda_runtime.h>

__global__ void sgemmNN( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta ) {
	A += blockIdx.x * 64 + threadIdx.x + threadIdx.y*16; 
	B += threadIdx.x + ( blockIdx.y * 16 + threadIdx.y ) * ldb; 
	C += blockIdx.x * 64 + threadIdx.x + (threadIdx.y + blockIdx.y * ldc ) * 16; 
	__shared__ float bs[16][17]; 
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; 
	const float *Blast = B + k; 
	
	do {
		#pragma unroll 32
		for( int i = 0; i < 16; i += 4 )
			bs[threadIdx.x][threadIdx.y+i] = B[i*ldb]; 
		B += 16;
		__syncthreads();
		
		#pragma unroll 32
		for( int i = 0; i < 16; i++, A += lda ){
			c[0] += A[0]*bs[i][0]; c[1] += A[0]*bs[i][1]; c[2] += A[0]*bs[i][2]; c[3] += A[0]*bs[i][3];
			c[4] += A[0]*bs[i][4]; c[5] += A[0]*bs[i][5]; c[6] += A[0]*bs[i][6]; c[7] += A[0]*bs[i][7];
			c[8] += A[0]*bs[i][8]; c[9] += A[0]*bs[i][9]; c[10] += A[0]*bs[i][10]; c[11] += A[0]*bs[i][11];
			c[12] += A[0]*bs[i][12]; c[13] += A[0]*bs[i][13]; c[14] += A[0]*bs[i][14]; c[15] += A[0]*bs[i][15];
			}
		__syncthreads();
	
	} while(B < Blast);
	
	for(int i=0; i<16; i++,C+=ldc){
		c[0]=alpha*c[i]+beta*c[0];
		}
}


// Kernel that executes on the CUDA device
__global__ void square_matrix(float* A, float* B, float* C, int N)
{
  //int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //if (idx<N) a[idx] = a[idx] * a[idx];
  int idx = blockIdx.x;
  int idy = blockIdx.y * blockDim.x + blockIdx.x;
  C[idx*N+idy]= A[blockIdx.x*N+threadIdx.x] * B[threadIdx.x*N+blockIdx.x];
}

// main routine that executes on the host
int main(int argc, char* argv[]) {
  float *A_h, *A_d;  // Pointer to host & device arrays
  float *B_h, *B_d;
  float *C_h, *C_d;
  float *Res,start,end;
  int N = atoi(argv[1]);
  int size = N * N * sizeof(float);
  
  A_h = (float *)malloc(size);        // Allocate array on host
  B_h = (float *)malloc(size); 
  C_h = (float *)malloc(size*N);
  Res = (float *)malloc(size); 
  
  cudaMalloc((void **) &A_d, size);   // Allocate array on device
  cudaMalloc((void **) &B_d, size);
  cudaMalloc((void **) &C_d, size*N);
  
  // Initialize host array and copy it to CUDA device
  for (int i=0; i<N*N; i++) {
	A_h[i] = (float)i;
  	B_h[i] = (float)i;
	}
  for (int i=0; i<N*N*N; i++){ 
	C_h[i] = 0;
	}
  start = omp_get_wtime();
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C_h, size*N, cudaMemcpyHostToDevice);
  
  
  // Do calculation on device:
  //int block_size = N;
  //int n_blocks = N*N
  dim3 dimBlock(N,1,1);
  dim3 dimGrid(N,N,1);
  square_matrix <<< dimGrid, dimBlock >>> (A_d, B_d, C_d, N);
  
  // Retrieve result from device and store it in host array
  cudaMemcpy(C_h, C_d, sizeof(float)*N*N*N, cudaMemcpyDeviceToHost);
  
  // Print results
  //for (int i=0; i<N; i++) printf("%d %f\n", i, a_h[i]);
  
  // Cleanup
  free(A_h);
  free(B_h); 
  //free(C_h); 
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  
  int chunk = (N*N)/24;
  int id=0;
  int i=0;
  int j=0;
  float accum=0;
  
  #pragma omp parallel shared (Res, chunk) private (id, i,j)
{
  
  id=omp_get_thread_num();
  
  for(i=id*chunk; i<id*chunk+chunk; i++){
  	for(j=0;j<N; j++){
  		accum+=C_h[j];
  		}
  	Res[i]=accum;
  	accum=0;
  		
  	}
  
  end = omp_get_wtime();
  
  printf("%f",end-start);
  
 free(C_h);
  }
 }



