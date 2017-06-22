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
data_file="local_$data_file"
version="ins-exe-vec"
bin_addr="/home/zpeng/benchmarks/test/function_ins_exe_vec/hatch"
#data_addr="/home/zpeng/benchmarks/rodinia_3.1/data/bfs"
data_addr="/home/zpeng/benchmarks/test/localized_graph"
power_max=16
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file

echo '==== 64 threads ===='
echo '==== 64 threads ====' >> $result_file
echo 'bfs:' >> $result_file
power=4
size=$((2 ** power))
while [	$power -le $power_max ]
do
	${bin_addr}/bfs 64 ${data_addr}/${data_file}.txt 32768 ${size} >> $result_file
	echo -n .
	power=$((power + 1))
	size=$((2 ** power))
done
echo ''
echo 'numactl:' >> $result_file
power=4
size=$((2 ** power))
while [	$power -le $power_max ]
do
	numactl -m 1 ${bin_addr}/bfs 64 ${data_addr}/${data_file}.txt 32768 ${size} >> $result_file
	echo -n .
	power=$((power + 1))
	size=$((2 ** power))
done
echo ''

echo '==== 256 threads ===='
echo '==== 256 threads ====' >> $result_file
echo 'bfs:' >> $result_file
power=4
size=$((2 ** power))
while [	$power -le $power_max ]
do
	${bin_addr}/bfs 256 ${data_addr}/${data_file}.txt 32768 ${size} >> $result_file
	echo -n .
	power=$((power + 1))
	size=$((2 ** power))
done
echo ''
echo 'numactl:' >> $result_file
power=4
size=$((2 ** power))
while [	$power -le $power_max ]
do
	numactl -m 1 ${bin_addr}/bfs 256 ${data_addr}/${data_file}.txt 32768 ${size} >> $result_file
	echo -n .
	power=$((power + 1))
	size=$((2 ** power))
done
echo done.
