#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

//#include <cuda.h>
//#include <cuda_runtime.h>

/* Includes, cuda */
#include "cublas.h"

#include "GotoBLAS2/common.h"
#include "GotoBLAS2/cblas.h"

#define K 3
#define TOL 0.02
#define MAX_TESTS 20
#define MIN_TESTS 5

int N; 
double Mflop ;
double half_Mflop ;
float quota;

void initMatrix(float* A) {
	
#pragma omp parallel for
	for (int ii=0; ii<N; ii++)
		for (int jj=0; jj<N; jj++) 
			A[ii*N+jj] = (float)rand()/(float)RAND_MAX;
	
}

void initMatrix2(float* A, int n) {
	
#pragma omp parallel for
	for (int ii=0; ii<N; ii++)
		for (int jj=0; jj<N; jj++) 
			A[ii*N+jj] = n;
	
}

void clearMatrix(float* A) {
	
#pragma omp parallel for
	for (int ii=0; ii<N; ii++)
		for (int jj=0; jj<N; jj++) 
			A[ii*N+jj] = 0.0;
	
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

int withinTol(double * array){
  
  int flag=1;
  double median=0;
  for (int i=0;i<K;i++){
    median += array[i];
  }
  median = median/K;
  
  for (int i=0;i<K;i++){
    flag &= (fabs(array[i]-median) <= TOL*median);
  }
  
  return flag;
}

void insertTest(double time, double *array){

   int pos=0; double tmp=0;
   for(int i=0;i<K;i++){
      if (array[i]>tmp) {tmp=array[i];pos=i;}
   }
   if (time<tmp) array[pos]=time;
}


void launch_thread(int tid, float* A, float* B, float* C)
{

	//clears all the runtime state for the current thread
	cudaThreadExit();
	
	// set working device
	if( cudaSetDevice(tid) != cudaSuccess)
	{
		printf("Failed to set to device #%d!\n", tid);
		return;
	}

	cublasStatus status;
	/* Initialize CUBLAS */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error\n");
		return;
	}

	// CuBLAS specific
	float* d_A = 0;
	float* d_B = 0;
	float* d_C = 0;
	float alpha = 1.0f;
	float beta = 0.0f;
	int workload = N * (N/2);
	
	/* Allocate device memory for the matrices */
	status = cublasAlloc(workload, sizeof(A[0]), (void**)&d_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error- alloc A\n");
		cublasShutdown();
		return;
		
	}
	status = cublasAlloc(N * N * quota, sizeof(B[0]), (void**)&d_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error- alloc B\n");
		cublasFree(d_A);
		cublasShutdown();
		return;
	}
	status = cublasAlloc(workload * quota, sizeof(C[0]), (void**)&d_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error- alloc C\n");
		cublasFree(d_A);
		cublasFree(d_B);
		cublasShutdown();
		return;
	}

	double start = omp_get_wtime();
	
	//              row, col, elemsize, *a, lda, *b, ldb
	status = cublasSetMatrix(N/2, N, sizeof(A[0]), A+(tid*(N/2)*N), N, d_A, N);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error- set matrix A\n");
		cublasShutdown();
		return;
	}
	status = cublasSetMatrix(N, (int)(N*quota), sizeof(B[0]), B, N, d_B, N);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error- set matrix B\n");
		cublasFree(d_A);
		cublasShutdown();
		return;
	}
	status = cublasSetMatrix(N, (int)(N*quota/2), sizeof(C[0]), C, N, d_C, N);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error - set matrix C\n");
		cublasFree(d_A);
		cublasFree(d_B);
		cublasShutdown();
		return;
	}

	/* Performs operation using cublas */
	cublasSgemm('n', 'n', 	//  cublas assumes column major
		N, (int)((N*quota)/2), N,	// m, n, k -> A:mxk; B:kxn; C:mxn
		alpha,		
		d_A, N,
		d_B, N,
		beta,
		d_C, N );

	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	/* Read the result back */
	cublasGetMatrix(N, (int)(N*quota/2), sizeof(C[0]), d_C, N, C+(tid*(N/2)*N), N);

	double elapsed = omp_get_wtime() - start;
	
	printf("Device %d: %d\t%f\t%f\n", tid, N, elapsed, (half_Mflop / (elapsed * 1E9) ) ) ;

	status = cublasFree(d_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! error free A\n");
		return;
	}

	status = cublasFree(d_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! error free B\n");
		return;
	}

	status = cublasFree(d_C);
	if (status != CUBLAS_STATUS_SUCCESS) {

		fprintf (stderr, "!!!! error free C\n");
		return;
	}

	/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! shutdown error (A)\n");
		return;
	}
	
}

int main (int argc, const char * argv[]) {

	if(argc != 3){ printf("Please supply N and quota.\n"); return 0; }

	N = atoi(argv[1]);
	Mflop = 2*powl(N,3) ;
	half_Mflop = powl(N,3) ;
	
	float quota = atof(argv[2]);
	if(quota>1){ printf("Quota must be < 1\n"); return 0;}
	
	float *A = (float*)malloc(sizeof(float)*N*N);
	float *B = (float*)malloc(sizeof(float)*N*N);
	float *C = (float*)malloc(sizeof(float)*N*N);
	float *D = (float*)malloc(sizeof(float)*N*N);
	
	printf("# N\tQuota\tRuntime\t GFlop/s\n");
	
	omp_set_num_threads( omp_get_num_procs() );

	initMatrix(A);
	initMatrix(B);
	//initMatrix2(A, 2);
	//initMatrix2(B, 3);
	clearMatrix(C);
	clearMatrix(D);

	int attempt=0;
	double ktests[K];
	int domoretests=1;
	float alpha = 1.0f;
	float beta = 0.0f;

	while (domoretests){
		// Returns a value in seconds of the time elapsed
		double start = omp_get_wtime(); 

		#pragma omp parallel 
		{
			if(omp_get_thread_num() == 0 )
				launch_thread(0, A, B, C);
			if(omp_get_thread_num() == 1 )
				launch_thread(1, A, B, C);
			else{
				/* Performs operation using GotoBLAS */
		cblas_sgemm (
			CblasColMajor, // order
			CblasNoTrans,  // transpose or not?
			CblasNoTrans,  // or CblasTrans
			N, (int)ceil(N*(1-quota)), N,       // since A, B, C are square, #rows==#col in each
			alpha,
			A, N,          // matrix A and size of its row
			B+(int)floor(N*(N*quota)), N,          // matrix B and size of its row
			beta,
			C+(int)floor(N*(N*quota)), N );     // matrix C and size of its row
			
			}
		}

		double end = omp_get_wtime();
		double elapsed = end - start;
		
		printf("%d\t%.1f\t%f\t%f\n", N, quota, elapsed, (Mflop / (elapsed * 1E9) ) ) ;

		//thrMM(A,B,D);
		//cmpMatrix(C, D);
		
		//printMatrix(C);
		//printMatrix(D);
		
		
		if (attempt<K){ktests[attempt]=elapsed;}
	    else {insertTest(elapsed,ktests);}
	    if ( attempt==MAX_TESTS) domoretests=0;
	    if (attempt>=MIN_TESTS && withinTol(ktests)) 
			domoretests=0;
	    attempt++;
	}
	
		// clean up
	free(A);
	free(B);
	free(C);
	free(D);

	return EXIT_SUCCESS;
}

