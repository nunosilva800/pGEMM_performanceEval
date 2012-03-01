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

void thrMM_col_major(float* A, float* B, float* C) {
	// Note: OpenMP allows
	//    more robust workload distribution among threads
	//    assign threads to specific cores (affinity)
	
	#pragma omp parallel for
	for (int ii=0; ii<N; ++ii) {
		for (int kk=0; kk<N; ++kk) {
			float r =  A[ii+N*kk];
			for (int jj=0; jj<N; ++jj) {
				C[ii+N*jj] += r * B[kk+N*jj];
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
	double diff = 0;
	printf("Checking errors... ");
	#pragma omp parallel for
    for (int ii=0; ii<N; ii++)
            for (int jj=0; jj<N; jj++){
	            diff = fabs( A[ii*N+jj] - B[ii*N+jj] );
            	if( diff > 1.0) err++;
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


int main (int argc, const char * argv[]) {

	if(argc != 3){ printf("Please supply N and quota.\n"); return 0; }

	N = atoi(argv[1]);
	double Mflop = 2*powl(N,3) ;
	
	float quota = atof(argv[2]);
	if(quota>1){ printf("Quota must be < 1\n"); return 0;}
	
	omp_set_num_threads( omp_get_num_procs() );
	
	cublasStatus status;
	/* Initialize CUBLAS */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error\n");
		return 0;
	}
	
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
	
	if(quota != 0){
		/* Allocate device memory for the matrices */
		status = cublasAlloc(N * N, sizeof(A[0]), (void**)&d_A);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! CUBLAS initialization error- alloc A\n");
			cublasShutdown();
			return 0;
		
		}
		status = cublasAlloc(N * N * quota, sizeof(B[0]), (void**)&d_B);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! CUBLAS initialization error- alloc B\n");
			cublasFree(d_A);
			cublasShutdown();
			return 0;
		}
		status = cublasAlloc(N * N * quota, sizeof(C[0]), (void**)&d_C);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! CUBLAS initialization error- alloc C\n");
			cublasFree(d_A);
			cublasFree(d_B);
			cublasShutdown();
			return 0;
		}
	}
	
	//printf("# N\tQuota\tRuntime\t GFlop/s\n");

	initMatrix(A);
	initMatrix(B);
	//initMatrix2(A, 2);
	//initMatrix2(B, 3);
	clearMatrix(C);
	//clearMatrix(D);

	int attempt=0;
	double ktests[K];
	int domoretests=1;

	/*
	Output format:
	N	Quota	GPU time	total time	 GFlop/s
	todo: need cudaEvents to get gpu time
	*/
	// print N and quota
	printf("%d\t%.1f", N, quota ) ;

	// Returns a value in seconds of the time elapsed
	//double gpu_start = 0;
	double start = omp_get_wtime(); 

	if(quota != 0){
		//              row, col, elemsize, *a, lda, *b, ldb
		status = cublasSetMatrix(N, N, sizeof(A[0]), A, N, d_A, N);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! CUBLAS initialization error- set matrix A\n");
			cublasShutdown();
			return 0;
		}
		status = cublasSetMatrix(N, N * quota, sizeof(B[0]), B, N, d_B, N);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! CUBLAS initialization error- set matrix B\n");
			cublasFree(d_A);
			cublasShutdown();
			return 0;
		}
		status = cublasSetMatrix(N, (int)(N*quota), sizeof(C[0]), C, N, d_C, N);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! CUBLAS initialization error - set matrix C\n");
			cublasFree(d_A);
			cublasFree(d_B);
			cublasShutdown();
			return 0;
		}
		
		//gpu_start = omp_get_wtime();
		
		/* Performs operation using cublas */
		cublasSgemm('n', 'n', 	//  cublas assumes column major
			N, (int)(N*quota), N,	// m, n, k -> A:mxk; B:kxn; C:mxn
			alpha,		
			d_A, N,
			d_B, N,
			beta,
			d_C, N );

		status = cublasGetError();
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! kernel execution error.\n");
			return 0;
		}
		
	}
	// print gpu time
	//if(quota != 0) printf("\t%.3f", (omp_get_wtime()-gpu_start) ) ;
	//else printf("\t0") ;

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

	if(quota != 0){
		/* Read the result back */
		cublasGetMatrix(N, (int)(N*quota), sizeof(C[0]), d_C, N, C, N);
	}

	double end = omp_get_wtime();
	double elapsed = end - start;
	
	//print total time
	printf("\t%.3f\t%f\n", elapsed, (Mflop / (elapsed * 1E9) ) ) ;

	//thrMM_col_major(A,B,D);
	//cmpMatrix(C, D);
	//printMatrix(C);
	//printMatrix(D);
	
	if(quota != 0){
		status = cublasFree(d_A);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! error free A\n");
			return 0;
		}

		status = cublasFree(d_B);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! error free B\n");
			return 0;
		}

		status = cublasFree(d_C);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! error free C\n");
			return 0;
		}
	}

	/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! shutdown error (A)\n");
		return 0;
	}
	
	// clean up
	free(A);
	free(B);
	free(C);
	//free(D);

	return EXIT_SUCCESS;
}

