#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_WIDTH = 32;

int N = 0;
int num_bytes = 0;
double Mflop = 0;
double half_Mflop = 0;

float *h_a = 0;   // device pointers
float *h_b = 0;   // device pointers
float *h_ab = 0;  // device pointers

void initMatrix(float* A) 
{	
	#pragma omp parallel for
	for (int ii=0; ii<N;ii++)
		for (int jj=0; jj<N; jj++) 
			A[ii*N+jj] = (float)rand()/(float)RAND_MAX;
}

void initMatrix2(float* A, int n) {	
#pragma omp parallel for
	for (int ii=0; ii<N; ii++)
		for (int jj=0; jj<N; jj++) 
			A[ii*N+jj] = n;
	
}

void clearMatrix(float* A) 
{
	#pragma omp parallel for
	for (int ii=0; ii<N; ii++)
		for (int jj=0; jj<N; jj++) 
			A[ii*N+jj] = 0.0;
}

void printMatrix(float* A) {
	printf("***********\n");
	for (int ii=0; ii<N; ii++){
		printf("\n");
		for (int jj=0; jj<N; jj++) 
			printf("%f\t",A[ii*N+jj]);
	}
	printf("***********\n");
	
}

void cmpMatrix(float* A,float* B) {
	int err = 0;
	printf("Checking errors... ");
	#pragma omp parallel for
    for (int ii=0; ii<N; ii++)
            for (int jj=0; jj<N; jj++){
            	if(A[ii*N+jj] != B[ii*N+jj]) err++;
	}
	printf("%d positions are wrong\n", err);
}

void thrMM(float* A, float* B, float* C) {
	// Note: OpenMP allows
	//    more robust workload distribution among threads
	//    assign threads to specific cores (affinity)
	
	#pragma omp parallel for
	for (int ii=0; ii<N; ++ii) {
		for (int kk=0; kk<N; ++kk) {
			float r =  A[ii*N+kk];
			for (int jj=0; jj<N; ++jj) {
				C[ii*N+jj] += r * B[kk*N+jj];
			}
		}
	}
}

__global__ void mat_mul(float *a, float *b, float *ab, int width){

// calculate the row & col index of the element
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  float result = 0;

// do dot product between row of a and col of b
  for(int k = 0; k < width; ++k)
          result += a[row*width+k] * b[k*width+col];

  ab[row*width+col] = result;
}

__global__ void mat_mul_shared(float *a, float *b, float *ab, int width){

  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
 
  float result = 0;

 	for(int m = 0; m < width/TILE_WIDTH; ++m){
		As[threadIdx.y][threadIdx.x] = a[(m*TILE_WIDTH+threadIdx.x) + width*row];
		Bs[threadIdx.y][threadIdx.x] = b[col + width*(m*TILE_WIDTH+threadIdx.y)];
  
		__syncthreads();

		for(int k = 0; k < TILE_WIDTH; ++k) 
			result += As[threadIdx.y][k] * Bs[k][threadIdx.x];

		__syncthreads();
		}
	ab[row*width+col] = result;
}

__global__ void mat_mul_shared_unroll(float *A, float *B, float *C, int width){

	__shared__ float As[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;
	
	float result = 0;
	
	for (int m = 0; m < width/TILE_WIDTH; ++m) {
		As[threadIdx.y][threadIdx.x] = A[Row*width + (m*TILE_WIDTH + threadIdx.x)];
		Bs[threadIdx.y][threadIdx.x] = B[(m*TILE_WIDTH + threadIdx.y) * width + Col];
		
		__syncthreads();
		
		#pragma unroll 4
		for (int k = 0; k < TILE_WIDTH; ++k) {
			result += As[threadIdx.y][k] * Bs[k][threadIdx.x];
		}
		
		__syncthreads();
	}
	C[Row*width+Col] = result;
}

// Device multiplication function called by Mul() 
// Compute C = A * B 
//	wA is the width of A 
//	wB is the width of B
__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
	// Block index 
	int bx = blockIdx.x; 
	int by = blockIdx.y;
	// Thread index 
	int tx = threadIdx.x; 
	int ty = threadIdx.y;
	
	int aBegin = wA * TILE_WIDTH * by;
	int aEnd	= aBegin + wA - 1;
	int aStep = TILE_WIDTH; 
	int bBegin = TILE_WIDTH * bx;
	int bStep = TILE_WIDTH * wB;
	
	// by the thread 
	float Csub = 0;
	// Loop over all the sub-matrices of A and B required to 
	// compute the block sub-matrix 
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		__shared__ float As[TILE_WIDTH][TILE_WIDTH];
		__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
		As[ty][tx] = A[a + wA * ty + tx]; Bs[ty][tx] = B[b + wB * ty + tx];
		
		// Synchronize to make sure the matrices are loaded 
		__syncthreads();
		
		// Multiply the two matrices together; 
		// each thread computes one element 
		// of the block sub-matrix 
		#pragma unroll 32
		for (int k = 0; k < TILE_WIDTH; ++k)
			Csub += As[ty][k] * Bs[k][tx];
		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	// Write the block sub-matrix to global memory;
	// each thread writes one element
	int c = wB * TILE_WIDTH * by + TILE_WIDTH * bx;
	C[c + wB * ty + tx] = Csub;
}

// splits work to 2 device
void launch_thread(int tid, float* A, float* B, float* C)
{

	//clears all the runtime state for the current thread
	cudaThreadExit();

	float *d_a = 0;  // device pointers
	float *d_b = 0;   // device pointers
	float *d_ab = 0; // device pointers

	dim3 grid( (int)ceil((float)N/TILE_WIDTH), (int)ceil((float)N/(2*TILE_WIDTH)) );
	dim3 block(TILE_WIDTH, TILE_WIDTH);

	// set working device
	if( cudaSetDevice(tid) != cudaSuccess)
	{
		printf("Failed to set to device #%d!\n", tid);
		return;
	}

	//printf("Starting on device %d...\n", tid);

	int byte_rows = (N/2)*N*sizeof(float);

	cudaMalloc( (void**) &d_a, byte_rows );
	cudaMalloc( (void**) &d_b, num_bytes);
	cudaMalloc( (void**) &d_ab, byte_rows );

	double start = omp_get_wtime(); 
	
	// send data to device
	cudaMemset( d_ab, 0, byte_rows);
	cudaMemcpy( d_a, A+(tid*(N/2)*N), byte_rows, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, B, num_bytes, cudaMemcpyHostToDevice );

	// launch kernel
	
	//mat_mul<<<grid,block>>>(d_a, d_b, d_ab, N);
	mat_mul_shared<<<grid,block>>>(d_a, d_b, d_ab, N);
	//mat_mul_shared_unroll<<<grid,block>>>(d_a, d_b, d_ab, N);
	//Muld<<<grid, block>>>(d_a, d_b, N, N, d_ab );

	//printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );

	// retreive data from device
	cudaMemcpy( 
		C+(tid*(N/2)*N), 
		d_ab,
		byte_rows,
		cudaMemcpyDeviceToHost );  

	double elapsed = omp_get_wtime() - start;
	
	printf("Device %d: %d\t%f\t%f\n", tid, N, elapsed, (half_Mflop / (elapsed * 1E9) ) ) ;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_ab);
	
}


int main(int argc, char* argv[]) {

	if(argc != 2){ printf("Please supply N.\n"); return 0; }

	N = atoi(argv[1]);
	num_bytes = N*N*sizeof(float);
	Mflop = 2*powl(N,3);
	half_Mflop = powl(N,3);

	h_a = (float*) malloc(num_bytes);
	h_b = (float*) malloc(num_bytes);
	h_ab = (float*) malloc(num_bytes);
	
	float *D = (float*)malloc(sizeof(float)*N*N);

	omp_set_num_threads( omp_get_num_procs() );

	initMatrix(h_a);
	initMatrix(h_b);
	clearMatrix(h_ab);
	//clearMatrix(D);
	
	//thrMM(h_a, h_b,D);

	int GPU_N=0;
	cudaGetDeviceCount(&GPU_N);
	//printf("Avaiable GPUs: %d\n", GPU_N);

	int tid;
	double begin = omp_get_wtime(); 
	
	#pragma omp parallel num_threads(2) private(tid)
	{
		tid=omp_get_thread_num();
		launch_thread(tid, h_a, h_b, h_ab);
	}

	double end = omp_get_wtime() - begin;
	
	printf("Total: %d\t%f\t%f\n", N, end, (Mflop / (end * 1E9) ) ) ;
	
	//cmpMatrix(h_ab, D);
	
	//printMatrix(h_ab);
	//printMatrix(D);

	free(h_a);
	free(h_b);
	free(h_ab);
	free(D);

	return 0;
}

