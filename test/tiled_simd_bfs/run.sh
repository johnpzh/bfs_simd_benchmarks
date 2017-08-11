#!/usr/bin/bash
make
# Determine the data file
if [[ $# -eq 0 ]]; then
	data_file="out.twitter"
else
	data_file=$1
fi
version="simd-bfs"
bin_addr="."
data_addr="/home/zpeng/benchmarks/data/twt"
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

power_max=13
for((power=4; power < power_max; ++power)); do
	size=$((2 ** power))
	echo "chunk size:${size}" >> $result_file
	echo "DDR" >> $result_file
	(set -x; numactl -m 0 ${bin_addr}/bfs ${data_addr}/${data_file} 256 ${size} >> $result_file)
	echo "MCDRAM" >> $result_file
(set -x; numactl -m 1 ${bin_addr}/bfs ${data_addr}/${data_file} 256 ${size} >> $result_file)
done
#echo "DDR" >> $result_file
#(set -x; numactl -m 0 ${bin_addr}/bfs ${data_addr}/${data_file} 256 256 >> $result_file)
#echo "MCDRAM" >> $result_file
#(set -x; numactl -m 1 ${bin_addr}/bfs ${data_addr}/${data_file} 256 256 >> $result_file)
echo done.
