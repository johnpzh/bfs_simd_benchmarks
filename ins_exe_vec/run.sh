#!/usr/bin/bash

version="ins-exe-vec"
bin_addr="/home/zpeng/benchmarks/ins_exe_vec"
data_addr="/home/zpeng/benchmarks/rodinia_3.1/data/bfs"
data_file="graph4096"
#data_file="graph16M"
#data_file="graph128M"
#no_core=64
buffer_max=24
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
echo "Buffer_Size Time" >> $result_file
#tno=1
power=4
size=$((2 ** $power))
#while [ $tno -le $no_core ]
while [ $power -le $buffer_max ]
do
#${bin_addr}/bfs ${tno} ${data_addr}/${data_file}.txt $size >> $result_file
	${bin_addr}/bfs 1 ${data_addr}/${data_file}.txt ${size} >> $result_file
	echo -n .
#tno=$((tno*2))
	power=$((power + 4))
	size=$((2 ** $power))
done
echo done.
