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
version="ins-exe-vec"
bin_addr="/home/zpeng/benchmarks/test/one-d_buffer_ins_exe_vec/hatch"
data_addr="/home/zpeng/benchmarks/rodinia_3.1/data/bfs"
#no_core=64
power_max=16
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
#echo "Threads I/E-Vec" >> $result_file
#echo "Buffer_Size Time" >> $result_file
echo "Chunk_size Time" >> $result_file
tno=1
power=4
size=$((2 ** power))
#while [ $tno -le $no_core ]
while [ $power -le $power_max ]
do
#${bin_addr}/bfs ${tno} ${data_addr}/${data_file}.txt 64 >> $result_file
#${bin_addr}/bfs 64 ${data_addr}/${data_file}.txt ${size} 8192 >> $result_file
#${bin_addr}/bfs 64 ${data_addr}/${data_file}.txt 65536 ${size} >> $result_file
	${bin_addr}/bfs ${tno} ${data_addr}/${data_file}.txt 16384 ${size} >> $result_file
	echo -n .
#tno=$((tno*2))
	power=$((power + 1))
	size=$((2 ** power))
#size=$((size * 2))
done
echo done.
