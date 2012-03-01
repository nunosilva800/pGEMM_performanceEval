#!/bin/sh
#PBS -V
#PBS -l walltime=1:00,nodes=compute-511-2
#PBS -q fermi
#PBS -j oe
#PBS -N multi

cd $PBS_O_WORKDIR


i=1024
while [ $i -ne 16384 ]
do
	./multi.out $i
	let i=$i+1024
done
