#!/usr/bin/bash
# Determine the data file
case $1 in
4096)
	data_file="graph4096"
	;;
128M)
	data_file="graph128M"
	;;
16M)
	data_file="graph16M"
	;;
*)
	data_file=$1
	;;
esac
#data_file="local_$data_file"
version="ins-exe-vec"
bin_addr="."
data_addr="/home/zpeng/benchmarks/data"
#data_addr="/home/zpeng/benchmarks/test/localized_graph"
#power_max=20
power_max=8
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
echo "Threads I/E-Vec" >> $result_file
#echo "Buffer_Size Time" >> $result_file
#echo "Chunk_size Time" >> $result_file
#power=4
power=0
size=$((2 ** power))
#while [ $tno -le $no_core ]
while [	$power -le $power_max ]
do
	${bin_addr}/bfs ${size} ${data_addr}/${data_file} 256 32768 >> $result_file
#${bin_addr}/bfs 256 ${data_addr}/${data_file} ${size} 32768 >> $result_file
#${bin_addr}/bfs 256 ${data_addr}/${data_file} 1024 ${size} >> $result_file
#${bin_addr}/bfs 64 ${data_addr}/${data_file}.txt 16384 ${size} >> $result_file
	echo -n .
	power=$((power + 1))
	size=$((2 ** power))
done
echo done.
