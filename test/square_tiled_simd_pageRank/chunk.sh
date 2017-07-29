#!/usr/bin/bash
make
# Determine the data file
version="tiled-simd-pageRank"
bin_addr="."
data_addr="/home/zpeng/benchmarks/data/twt"
data_file="out.twitter"
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"
echo "data_file: ${data_addr}/${data_file}"

touch $result_file
echo "Threads Time" >> $result_file
power=0
power_max=16
size=$((2 ** power))
while [[ $power -le $power_max ]]; do
	echo "CHUNK_SIZE: ${size}" >> $result_file
	(set -x; ${bin_addr}/page_rank ${data_addr}/${data_file} 1 4096 $size >> $result_file)
	((++power))
	size=$((2 ** power))
done
echo done.
