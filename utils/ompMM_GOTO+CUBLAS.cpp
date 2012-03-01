#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/* Includes, cuda */
#include "cublas.h"

#include "GotoBLAS2/common.h"
#include "GotoBLAS2/cblas.h"

int N; 

void initMatrix(float* A) {
	
#pragma omp parallel for
	for (int ii=0; ii<N; ii++)
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
	for (int ii=0; ii<N; ii++)
		for (int jj=0; jj<N; jj++) 
			printf("%f\t",A[ii*N+jj]);
	
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

void cmpMatrix(float* A,float* B) {
        for (int ii=0; ii<N; ii++)
                for (int jj=0; jj<N; jj++){
                	if(ii == jj) printf("!!!%f vs %f \n", A[ii*N+jj], B[ii*N+jj]);
	}
}

inline int min(int a, int b)
{ return a < b ? a : b; }

int main (int argc, const char * argv[]) {

	if(argc != 3){ printf("Please supply N and number of cores.\n"); return 0; }

	N = atoi(argv[1]);
	int Nb = 32;
	double Mflop = 2*pow(N,3) ;

	float *A = (float*)malloc(sizeof(float)*N*N);
	float *B = (float*)malloc(sizeof(float)*N*N);
	float *C = (float*)malloc(sizeof(float)*N*N);
	float *D = (float*)malloc(sizeof(float)*N*N);
	
	// CuBLAS specific
	float* d_A = 0;
	float* d_B = 0;
	float* d_C = 0;
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasStatus status;
	int n2 = N * N;

	/* Initialize CUBLAS */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}
	
	/* Allocate device memory for the matrices */
	cublasAlloc(N * N, sizeof(A[0]), (void**)&d_A);
	cublasAlloc(N * N, sizeof(B[0]), (void**)&d_B);
	cublasAlloc(N * N, sizeof(C[0]), (void**)&d_C);
	
	printf("# N\tThreads\tRuntime\t GFlop/s\n");
	
	// #threads = #cores
	int nthreads = atoi(argv[2]);
	omp_set_num_threads(nthreads);

	initMatrix(A);
	initMatrix(B);
	clearMatrix(C);
	clearMatrix(D);
	
	//for (int ii=0; ii<N; ii+=Nb){
	//	for (int kk=0; kk<N; kk+=Nb){
	//		for (int jj=0; jj<N; jj+=Nb){
			
				// Returns a value in seconds of the time elapsed
				double start = omp_get_wtime(); 

				//              row, col, elemsize, *a, lda, *b, ldb
				cublasSetMatrix(N, N, sizeof(A[0]), A, N, d_A, N);
				cublasSetMatrix(N, (int)(N*0.75), sizeof(B[0]), B, N, d_B, N);
				cublasSetMatrix(N, (int)(N*0.75), sizeof(C[0]), C, N, d_C, N);

				/* Performs operation using cublas */
				cublasSgemm('t', 't', 	// transpose because cublas assumes column major
					N, (int)(N*0.75), N,	// m, n, k -> A:mxk; B:kxn; C:mxn
					alpha,		
					d_A, N,
					d_B, N,
					beta,
					d_C, N );

				status = cublasGetError();
				if (status != CUBLAS_STATUS_SUCCESS) {
					fprintf (stderr, "!!!! kernel execution error.\n");
					return EXIT_FAILURE;
				}
		
				/* Performs operation using GotoBLAS */
				cblas_sgemm (
					CblasRowMajor, // order
					CblasNoTrans,  // transpose or not?
					CblasNoTrans,  // or CblasTrans
					N, (int)(N*0.25), N,       // since A, B, C are square, #rows==#col in each
					alpha,
					A, N,          // matrix A and size of its row
					B+(int)(N*0.75), N,          // matrix B and size of its row
					beta,
					C+(int)(N*0.75), N );     // matrix C and size of its row
				
				/* Read the result back */
				cublasGetMatrix(N, (int)(N*0.75), sizeof(C[0]), d_C, N, C, N);

				double end = omp_get_wtime();
				double elapsed = end - start;
				printf("%d\t%d\t%f\t%f\n", N, nthreads, elapsed, (Mflop / (elapsed * 1E9) ) ) ;
				thrMM(A,B,D);
	//		}
	//	}
	//}
	cmpMatrix(D, C);
	
	// clean up
	free(A);
	free(B);
	free(C);
	free(D);
	cublasFree(d_A);
	cublasFree(d_B);
	cublasFree(d_C);

	/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! shutdown error (A)\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

