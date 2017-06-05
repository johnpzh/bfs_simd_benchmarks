#!/usr/bin/bash
# Determine the data file
case $1 in
4096)
	data_file="graph4096"
	;;
128M)
	data_file="graph128M"
	;;
*)
	data_file="graph16M"
	;;
esac
timestamp=$(date +%Y%m%d-%H%M%S)
data_addr="/home/zpeng/benchmarks/rodinia_3.1/data/bfs"
no_core=64
result_file="result_${timestamp}_${data_file}.txt"
touch $result_file

# Rodinia Original
version="rodinia"
bin_addr="/home/zpeng/benchmarks/rodinia_3.1/openmp/bfs"
echo -e "Threads Rodinia_Origin" >> $result_file
echo -n "$version running"
tno=1
while [ $tno -le $no_core ]
do
	${bin_addr}/bfs ${tno} ${data_addr}/${data_file}.txt >> $result_file
	echo -n .
	tno=$((tno*2))
done
echo done.

# Rodinia with SIMD
version="rodinia-vec"
#bin_addr="/home/zpeng/benchmarks/rodinia_v"
bin_addr="/home/zpeng/benchmarks/test/rodinia_v_double_loop"
echo -e "Threads Rodinia_SIMD" >> $result_file
echo -n "$version running"
tno=1
while [ $tno -le $no_core ]
do
	${bin_addr}/bfs ${tno} ${data_addr}/${data_file}.txt >> $result_file
	echo -n .
	tno=$((tno*2))
done
echo done.

# Inspector/Executor Sequential Version
version="ins-exe-seq"
bin_addr="/home/zpeng/benchmarks/ins_exe_seq"
echo -e "Threads I/E_Seq" >> $result_file
echo -n "$version running"
tno=1
while [ $tno -le $no_core ]
do
	${bin_addr}/bfs ${tno} ${data_addr}/${data_file}.txt >> $result_file
	echo -n .
	tno=$((tno*2))
done
echo done.

## Inspector/Executor Vectorization Version
#version="ins-exe-vec"
#bin_addr="/home/zpeng/benchmarks/ins_exe_vec"
#echo -e "Threads\tInspector-Executor_Vectorization" >> $result_file
#echo -n "$version running"
#tno=1
#while [ $tno -le $no_core ]
#do
#	${bin_addr}/bfs ${tno} ${data_addr}/${data_file}.txt >> $result_file
#	echo -n .
#	tno=$((tno*2))
#done
#echo done.

# Clean up
rm path.txt
