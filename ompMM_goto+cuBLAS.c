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


int main (int argc, const char * argv[]) {

	if(argc != 3){ printf("Please supply N and quota.\n"); return 0; }

	N = atoi(argv[1]);
	double Mflop = 2*powl(N,3) ;
	
	/*
	size_t fr;
	size_t total;
	
	cuMemGetInfo(&fr, &total);
	
	printf("Free: %ld, Total %ld\n", fr, total);
	*/
	
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
	int n2 = N * N;
	
	/* Allocate device memory for the matrices */
	status = cublasAlloc(N * N, sizeof(A[0]), (void**)&d_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error- alloc A\n");
		return 0;
		cublasShutdown();
	}
	status = cublasAlloc(N * N, sizeof(B[0]), (void**)&d_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error- alloc B\n");
		cublasFree(d_A);
		cublasShutdown();
		return 0;
	}
	status = cublasAlloc(N * N, sizeof(C[0]), (void**)&d_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error- alloc C\n");
		cublasFree(d_A);
		cublasFree(d_B);
		cublasShutdown();
		return 0;
	}
	
	printf("# N\tQuota\tRuntime\t GFlop/s\n");
	
	// #threads = #cores
	float quota = atof(argv[2]);
	if(quota>1){ printf("Quota must be < 1\n"); return 0;}
	omp_set_num_threads( omp_get_num_procs() );

	initMatrix(A);
	initMatrix(B);
	clearMatrix(C);
	clearMatrix(D);

	int attempt=0;
	double ktests[K];
	int domoretests=1;

	//while (domoretests){
		// Returns a value in seconds of the time elapsed
		double start = omp_get_wtime(); 

		//              row, col, elemsize, *a, lda, *b, ldb
		status = cublasSetMatrix(N, N, sizeof(A[0]), A, N, d_A, N);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! CUBLAS initialization error- set matrix A\n");
			cublasShutdown();
			return 0;
		}
		status = cublasSetMatrix(N, (int)(N*quota), sizeof(B[0]), B, N, d_B, N);
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

		/* Performs operation using cublas */
		cublasSgemm('t', 't', 	// transpose because cublas assumes column major
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

		/* Performs operation using GotoBLAS */
		cblas_sgemm (
			CblasRowMajor, // order
			CblasNoTrans,  // transpose or not?
			CblasNoTrans,  // or CblasTrans
			N, (int)(N*(1-quota)), N,       // since A, B, C are square, #rows==#col in each
			alpha,
			A, N,          // matrix A and size of its row
			B+(int)(N*quota), N,          // matrix B and size of its row
			beta,
			C+(int)(N*quota), N );     // matrix C and size of its row
	
		/* Read the result back */
		cublasGetMatrix(N, (int)(N*quota), sizeof(C[0]), d_C, N, C, N);

		double end = omp_get_wtime();
		double elapsed = end - start;
		

		/*
		printf("%d\t%.1f\t%f\t%f\n", N, quota, elapsed, (Mflop / (elapsed * 1E9) ) ) ;
		if (attempt<K){ktests[attempt]=elapsed;}
	    else {insertTest(elapsed,ktests);}
	    if ( attempt==MAX_TESTS) domoretests=0;
	    if (attempt>=MIN_TESTS && withinTol(ktests)) 
			domoretests=0;
	    attempt++;
	}*/

	thrMM(A,B,D);
	cmpMatrix(C, D);
	
	printMatrix(C);
	printMatrix(D);
	
	// clean up
	free(A);
	free(B);
	free(C);
	free(D);
	
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

	/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! shutdown error (A)\n");
		return 0;
	}

	return EXIT_SUCCESS;
}

