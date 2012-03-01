
#sceneA
if [ $1 -eq 1 ]
then
	icc ompMM_main.c -std=c99 -openmp -msse3 -O2 -o amd.out
	icc ompMM_main.c -std=c99 -openmp -xSSE3 -O2 -o intel.out

	#icc ompMM_main.c -std=c99 -openmp -msse3 -O2 -fno-alias -o amd.out
	#icc ompMM_main.c -std=c99 -openmp -xSSE3 -O2 -fno-alias -o intel.out

	icc ompMM_main_blocked.c -std=c99 -openmp -msse3 -O2 -o amd_block.out
	icc ompMM_main_blocked.c -std=c99 -openmp -xSSE3 -O2 -o intel_block.out

	if [ $2 -eq 1 ]
	then
		qsub sub.intel.sh -N omp_intel_main
		qsub sub.amd.sh -N omp_amd_main
		
		qsub sub.intel.block.sh -N omp_intel_block
		qsub sub.amd.block.sh -N omp_amd_block
	fi
fi

#sceneB
if [ $1 -eq 2 ]
then
	#icc ompMM_BLAS.c -o amd.blas.out -std=c99 -openmp -O2 -lpthread -lgfortran -lgoto2 -LGotoBLAS2/
	#nvcc ompMM_CUBLAS.cpp -o cublas.out -arch=sm_21 -Xcompiler -fopenmp -lcublas 
	nvcc ompMM_goto+cuBLAS.c -o gotocublas.out -arch=sm_21 -lcublas -ccbin icc --compiler-options -openmp, -lgoto2, -L GotoBLAS2/, -lgfortran -Xcompiler -std=c99
	
	if [ $2 -eq 1 ]
	then
		#qsub sub.amd.blas.sh -N amd_blas
		#qsub sub.cublas.sh -N cublas
		qsub sub.goto+cublas.sh -N goto+cublas
	fi

fi

#sceneC
if [ $1 -eq 3 ]
then
	nvcc cuda.cu -o cuda.out -arch=sm_21 -ccbin icc -Xcompiler -openmp -Xcompiler -O3 -Xcompiler -msse3
	nvcc cuda_multiGPU.cu -o cuda_multiGPU.out -arch=sm_21 -ccbin icc -Xcompiler -openmp -Xcompiler -O3 -Xcompiler -msse3

	if [ $2 -eq 1 ]
	then
		qsub sub.cuda.sh -N cuda
		#qsub...
	fi

fi

