#include <omp.h>
#include "GotoBLAS2/common.h"
#include "GotoBLAS2/cblas.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

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


int main (int argc, const char * argv[]) {

	if(argc != 3){ printf("Please supply N and number of cores.\n"); return 0; }

	N = atoi(argv[1]);
	double Mflop = 2*pow(N,3) ;

	float *A = (float*)malloc(sizeof(float)*N*N);
	float *B = (float*)malloc(sizeof(float)*N*N);
	float *C = (float*)malloc(sizeof(float)*N*N);
	
	// shall run for half the # cores, # cores and double # cores
	int nthreads = atoi(argv[2])/2;
	int nthreadsMax = nthreads*4;

	printf("# N\tThreads\tRuntime\t GFlop/s\n");

	for ( ; nthreads <= nthreadsMax ; nthreads*=2)
	{
		omp_set_num_threads(nthreads);

		initMatrix(A);
		initMatrix(B);
		clearMatrix(C);
	
		for (int i=0; i<5; i++) {
			double start = omp_get_wtime(); // Returns a value in seconds of the time elapsed

			// single precision, C = alpha*AB + beta*C
			cblas_sgemm (
				CblasRowMajor, // order
				CblasNoTrans,  // transpose or not?
				CblasNoTrans,  // or CblasTrans
				N, N, N,       // since A, B, C are square, #rows==#col in each
				1.0,             // alpha
				A, N,          // matrix A and size of its row
				B, N,          // matrix B and size of its row
				1.0,             // beta
				C, N           // matrix C and size of its row
			);			

			double end = omp_get_wtime();
			double elapsed = end - start;
			printf("%d\t%d\t%f\t%f\n", N, nthreads, elapsed, (Mflop / (elapsed * 1E9) ) ) ;
		}
	}
}
