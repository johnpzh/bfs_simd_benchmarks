#!/usr/bin/bash

#cd amplxe-report/
#./ddr_amplxe.sh 2 64 l
#./ddr_amplxe.sh 2 64 o
#./ddr_amplxe.sh 2 256 l
#./ddr_amplxe.sh 2 256 o
#./mcdr_amplxe.sh 2 64 l
#./mcdr_amplxe.sh 2 64 o
#./mcdr_amplxe.sh 2 256 l
#./mcdr_amplxe.sh 2 256 o

cd hatch/
for ((i = 0; i < 2; ++i))
do
	./run.sh 16M
done

for ((i = 0; i < 2; ++i))
do
	numactl -m 0 ./run.sh 16M
done

for ((i = 0; i < 2; ++i))
do
	numactl -m 1 ./run.sh 16M
done

for ((i = 0; i < 2; ++i))
do
	./run.sh 128M
done

for ((i = 0; i < 2; ++i))
do
	numactl -m 0 ./run.sh 128M
done

for ((i = 0; i < 2; ++i))
do
	numactl -m 1 ./run.sh 128M
done

