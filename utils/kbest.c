#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>  // necessária a função time para a seed do random
#include <float.h> //necessario para o DBL_MAX e MIN
#include <math.h>
#include <unistd.h> //necessário para o sleep


#define K 3
#define TOL 0.05
#define MAX_TESTS 50
#define MIN_TESTS 5

int withinTol(double * array){
  
  int flag=1;
  double median=0;
  for (int i=0;i<K;i++){
    median += array[K];
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


int main() {

  int attempt=0;
  double ktests[K];
  int domoretests=1;
  int r;
  srand(time( 0 ));

  while (domoretests){
    
  
    double start = omp_get_wtime(); // Returns a value in seconds of the time
    /*Fazer o teste aqui*/
    printf("vou dormir\n"); 
    r=(int) rand()%4;
    sleep(r);
    printf("acordei %d\n",r);
    
    
    double end = omp_get_wtime(); // Returns a value in seconds of the time
    double time= end-start; //medição do tempo do teste
    printf("time: %f\n",time);
    if (attempt<K){ktests[attempt]=time;}
    else {insertTest(time,ktests);}
    if ( attempt==MAX_TESTS) domoretests=0;
    if (attempt>=MIN_TESTS && withinTol(ktests)) 
        domoretests=0;
    attempt++;
  }
  
  for(int i=0;i<K;i++) printf("%f\n",ktests[i]);
  printf("attempts: %d\n",attempt);

}
