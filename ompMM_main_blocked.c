#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <float.h> //necessario para o DBL_MAX e MIN

#define K 3
#define TOL 0.05
#define MAX_TESTS 20
#define MIN_TESTS 5

int N; 
int Nb = 64;

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

void cmpMatrix(float* A,float* B) {
        for (int ii=0; ii<N; ii++)
                for (int jj=0; jj<N; jj++){
                	if(ii == jj) printf("!!!%f vs %f \n", A[ii*N+jj], B[ii*N+jj]);
	}
}

inline int min(int a, int b)
{ return a < b ? a : b; }

void thrMMB(float* A, float* B, float* C) {
	
	#pragma omp parallel for
	for (int ii=0; ii<N; ii+=Nb){
		for (int kk=0; kk<N; kk+=Nb){
			for (int jj=0; jj<N; jj+=Nb){
				
				for (int i=ii; i<min(ii+Nb,N); i++){
					for(int k=kk; k<min(kk+Nb,N); k++){
						float r=A[i*N+k];
						for (int j=jj; j<min(jj+Nb,N);j++){
							C[i*N+j] += r * B[k*N+j];
						}
					}
				}
			}
		}
	}
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

   int pos=0; double tmp=DBL_MIN;
   for(int i=0;i<K;i++){
      if (array[i]>tmp) {tmp=array[i];pos=i;}
   }
   if (time<tmp) array[pos]=time;
}

void thrMMB2(float* A, float* B, float* C) {

	omp_set_dynamic(0);
	omp_set_nested(1);
	#pragma omp parallel for ordered schedule(static, Nb) collapse(2)
	for (int jj = 0; jj < N; jj = jj+Nb)
		for (int kk = 0; kk < N; kk = kk+Nb)
			for (int i = 0; i < N; i++)
		   		for (int j = jj; j < min(jj+Nb,N); ++j) {
					float r = 0;
		           	for (int k = kk; k < min(kk+Nb,N); k++)  
		       	    	r +=  A[i*N+k] * B[k*N+j];
				    C[i*N+j] += r;
				}	
}

int main (int argc, const char * argv[]) {

	if(argc != 3){ printf("Please supply N and number of cores.\n"); return 0; }

	N = atoi(argv[1]);
	double Mflop = 2*pow(N,3) ;

	float *A = (float*)malloc(sizeof(float)*N*N);
	float *B = (float*)malloc(sizeof(float)*N*N);
	float *C = (float*)malloc(sizeof(float)*N*N);
	float *D = (float*)malloc(sizeof(float)*N*N);
	
	// shall run for half the # cores, # cores and double # cores
	int nthreads = atoi(argv[2])/2;
	int nthreadsMax = nthreads*4;

	// set Nb according to cpu used
	if(nthreads == 8) Nb = 60; //for intel
	else Nb = 256; // for amd

	printf("# N\tThreads\tRuntime\t GFlop/s\n");

	for ( ; nthreads <= nthreadsMax ; nthreads*=2)
	{
		omp_set_num_threads(nthreads);
		initMatrix(A);
		initMatrix(B);
		clearMatrix(C);
		clearMatrix(D);

		int attempt=0;
		double ktests[K];
		int domoretests=1;
	
		while (domoretests){
			double start = omp_get_wtime(); // Returns a value in seconds of the time elapsed

			//thrMMB2(A, B, C);
			thrMMB(A, B, D);

			double end = omp_get_wtime();
			double elapsed = end - start;

			printf("%d\t%d\t%f\t%f\n", N, nthreads, elapsed, (Mflop / (elapsed * 1E9) ) ) ;
			/*
			cmpMatrix(C, D);
			
			printf("---\n");
			start = omp_get_wtime(); // Returns a value in seconds of the time elapsed
			
			thrMMB2(A, B, C);

			end = omp_get_wtime();
			elapsed = end - start;

			printf("%d\t%d\t%f\t%f\n", N, nthreads, elapsed, (Mflop / (elapsed * 1E9) ) ) ;
		*/
			
			if (attempt<K){ktests[attempt]=elapsed;}
			    else {insertTest(elapsed,ktests);}
			    if ( attempt==MAX_TESTS) domoretests=0;
			    if (attempt>=MIN_TESTS && withinTol(ktests)) 
				domoretests=0;
			    attempt++;
		}
		
		for(int i=0;i<K;i++) printf("# %f\n",ktests[i]);
  		printf("# attempts: %d\n",attempt);
	}
}

