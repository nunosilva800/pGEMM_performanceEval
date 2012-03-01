#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

int N = 1024;

void initMatrix(float* A) {
	
#pragma omp parallel for
	for (int ii=0; ii<N;ii++)
		for (int jj=0; jj<N; jj++) 
			A[ii*N+jj] = (float)rand()/(float)RAND_MAX;
}

void clearMatrix(float* A) {

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
            	if( (A[ii*N+jj] - B[ii*N+jj]) < 1 ) err++;
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

#define UNROLL_FACTOR 4

__global__ void mat_mul_shared_unroll(float *A, float *B, float *C, int width){

	__shared__ float As[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
	float res[UNROLL_FACTOR] = {//0,0,0,0,0,0,0,0,
								//0,0,0,0,
								0,0,
								0,
								0};

	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;
	
	for (int m = 0; m < width/TILE_WIDTH; m++) {
		As[threadIdx.y][threadIdx.x] = A[Row*width + (m*TILE_WIDTH + threadIdx.x)];
		Bs[threadIdx.y][threadIdx.x] = B[(m*TILE_WIDTH + threadIdx.y) * width + Col];
		
		__syncthreads();
		
		for(int k = 0; k < TILE_WIDTH; k+=UNROLL_FACTOR) {
			res[0] += As[threadIdx.y][k] * Bs[k][threadIdx.x];
			res[1] += As[threadIdx.y][k+1] * Bs[k+1][threadIdx.x];
			res[2] += As[threadIdx.y][k+2] * Bs[k+2][threadIdx.x];
			res[3] += As[threadIdx.y][k+3] * Bs[k+3][threadIdx.x];
			/*res[4] += As[threadIdx.y][k+4] * Bs[k+4][threadIdx.x];
			res[5] += As[threadIdx.y][k+5] * Bs[k+5][threadIdx.x];
			res[6] += As[threadIdx.y][k+6] * Bs[k+6][threadIdx.x];
			res[7] += As[threadIdx.y][k+7] * Bs[k+7][threadIdx.x];			
			res[8] += As[threadIdx.y][k+8] * Bs[k+8][threadIdx.x];
			res[9] += As[threadIdx.y][k+9] * Bs[k+9][threadIdx.x];
			res[10] += As[threadIdx.y][k+10] * Bs[k+10][threadIdx.x];
			res[11] += As[threadIdx.y][k+11] * Bs[k+11][threadIdx.x];
			res[12] += As[threadIdx.y][k+12] * Bs[k+12][threadIdx.x];
			res[13] += As[threadIdx.y][k+13] * Bs[k+13][threadIdx.x];
			res[14] += As[threadIdx.y][k+14] * Bs[k+14][threadIdx.x];
			res[15] += As[threadIdx.y][k+15] * Bs[k+15][threadIdx.x];*/

	}
		
		__syncthreads();
	}
	
	for(int k = 0; k < UNROLL_FACTOR; k++)
		C[Row*width+Col] += res[k];
}


extern "C" __global__ void mmkernel2 (float* a, float* b, float* c, int n, int m, int p) {
	int tx = threadIdx.x; 
	int i = blockIdx.x*128 + tx; 
	int j = blockIdx.y*4;

	__shared__ float cb0[128][8]; //, cb1[128], cb2[128], cb3[128];	
	__shared__ float sb[128];

	__shared__ float sum0[8];
	// {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	//  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; 
	// , sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

	for (int ks = 0; ks < p; ks += 128) {
		//#pragma unroll 4
		//for(int ur=0; ur < 4; ur++) 
		cb0[tx][0] = c[ks+tx+p*(j+0)];
		cb0[tx][1] = c[ks+tx+p*(j+1)];
		cb0[tx][2] = c[ks+tx+p*(j+2)];
		cb0[tx][3] = c[ks+tx+p*(j+3)];
		//cb0[tx][4] = c[ks+tx+p*(j+4)];
		//cb0[tx][5] = c[ks+tx+p*(j+5)];
		//cb0[tx][6] = c[ks+tx+p*(j+6)];
		//cb0[tx][7] = c[ks+tx+p*(j+7)];
		
		//cb1[tx] = c[ks+tx+p*(j+1)];
		//cb0[tx] = c[ks+tx+p*(j+2)];
		//cb1[tx] = c[ks+tx+p*(j+3)];
		sb[tx] = b[i+n*(ks+tx)];
		__syncthreads();
		for (int k = 0; k < 128; ++k) { 
		//for (int k = ks; k < ks+128; ++k) { 
			//int pos = ks+k;
			//float rb = bs[k];
			//#pragma unroll 4 
			//for(int ur=0; ur < 4; ur++)
			sum0[0] += sb[k] * cb0[k][0];
			sum0[1] += sb[k] * cb0[k][1];
			sum0[2] += sb[k] * cb0[k][2];
			sum0[3] += sb[k] * cb0[k][3]; 
			//sum0[4] += sb[k] * cb0[k][4];
			//sum0[5] += sb[k] * cb0[k][5];
			//sum0[6] += sb[k] * cb0[k][6];
			//sum0[7] += sb[k] * cb0[k][7]; 
			//sum1 += sb[k] * cb1[k];
			//sum2 += sb[k] * cb2[k];
			//sum3 += sb[k] * cb3[k]; 
		}
		__syncthreads();
	}
	
	

	//if ((i+n*j)<n2) {
		//#pragma unroll 4
		//for(int ur=0; ur < 4; ur++) 			
		a[i+n*(j+0)] = sum0[0];
		a[i+n*(j+1)] = sum0[1];
		a[i+n*(j+2)] = sum0[2];
		a[i+n*(j+3)] = sum0[3];
		//a[i+n*(j+4)] = sum0[4];
		//a[i+n*(j+5)] = sum0[5];
		//a[i+n*(j+6)] = sum0[6];
		//a[i+n*(j+7)] = sum0[7];
	//	if ( (i+n*(j+1))< n2 ) 
		//a[i+n*(j+1)] = sum1;
		//a[i+n*(j+2)] = sum2;
		//a[i+n*(j+3)] = sum3;
	//}
	//printf("J %d\n",j);

}

int main(int argc, char* argv[]) {

	if(argc != 2){ printf("Please supply N.\n"); return 0; }

	N = atoi(argv[1]);

	int num_bytes= N*N*sizeof(float);
	double Mflop = 2*powl(N,3) ;

	float *d_a = 0, *h_a = 0;   // device and host pointers
	float *d_b = 0, *h_b = 0;   // device and host pointers
	float *d_ab = 0, *h_ab = 0; // device and host pointers
	float *h_c = 0;
	
	h_a = (float*) malloc(num_bytes);
	h_b = (float*) malloc(num_bytes);
	h_ab = (float*) malloc(num_bytes);
	h_c = (float*) malloc(num_bytes);

	cudaMalloc( (void**) &d_a, num_bytes);
	cudaMalloc( (void**) &d_b, num_bytes);
	cudaMalloc( (void**) &d_ab, num_bytes);

	omp_set_num_threads( omp_get_num_procs() );

	initMatrix(h_a);
	initMatrix(h_b);
	clearMatrix(h_ab);
	clearMatrix(h_c);
	
	// #blocks and #threads per block
	dim3 grid( (int)ceil((float)N/TILE_WIDTH), (int)ceil((float)N/(TILE_WIDTH)) );
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	
	dim3 grid_U( (int)ceil((float)N/(TILE_WIDTH)), (int)ceil((float)N/(TILE_WIDTH*UNROLL_FACTOR)) );
	dim3 block_U(TILE_WIDTH, TILE_WIDTH);

	dim3 grid2(floor(N/128), N/4);

	//printf("Starting ...\n");
	
	double start = omp_get_wtime(); 
	
	cudaMemset( d_ab, 0, num_bytes);

	// send data to device
	cudaMemcpy( d_a, h_a, num_bytes, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, h_b, num_bytes, cudaMemcpyHostToDevice );

	// launch kernel
	mat_mul_shared<<<grid,block>>>(d_a, d_b, d_ab, N);
	//mat_mul_shared_unroll<<<grid_U,block_U>>>(d_a, d_b, d_ab, N);
	//mmkernel2<<<grid2,128>>>(d_ab,d_a,d_b,N,N,N);
	//Muld<<<grid, block>>>(d_a, d_b, N, N, d_ab);
	//sgemmNN<<<grid, block>>>(d_a, N, d_b, N, d_ab, N, TILE_WIDTH, 1, 1 );
	//mat_mul<<<grid,block>>>(d_a, d_b, d_ab, N);
	
	// retreive data from device
	cudaMemcpy( h_ab, d_ab, num_bytes, cudaMemcpyDeviceToHost );  

	double elapsed = omp_get_wtime() - start;
	
	printf("%d\tunroll%d\t%f\t%f\n", N, UNROLL_FACTOR, elapsed, (Mflop / (elapsed * 1E9) ) ) ;
/*
	thrMM(h_a, h_b, h_c);
	cmpMatrix(h_ab, h_c);

printf("%f\t",h_ab[0]);
printf("%f\t",h_ab[N]);
printf("%f\t",h_ab[N*(N-1)]);
printf("%f\n",h_ab[N*N-1]);
printf("%f\t",h_c[0]);
printf("%f\t",h_c[N]);
printf("%f\t",h_c[N*(N-1)]);
printf("%f\n",h_c[N*N-1]);*/


	//printf("Clean-up\n");
	free(h_a);
	free(h_b);
	free(h_ab);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_ab);

	return 0;
}

