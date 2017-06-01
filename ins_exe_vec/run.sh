#!/usr/bin/bash

version="ins-exe-vec"
bin_addr="/home/zpeng/benchmarks/test/ins_exe_vec"
data_addr="/home/zpeng/benchmarks/rodinia_3.1/data/bfs"
#data_file="graph4096"
data_file="graph16M"
#data_file="graph128M"
no_core=64
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
tno=1
while [ $tno -le $no_core ]
do
	${bin_addr}/bfs ${tno} ${data_addr}/${data_file}.txt >> $result_file
	echo -n .
	tno=$((tno*2))
done
echo done.
