#!/usr/bin/bash
make
# Determine the data file
if [[ $# -eq 0 ]]; then
	data_file="soc-pokec-relationships.txt"
else
	data_file=$1
fi
version="naive-bfs"
bin_addr="."
data_addr="/home/zpeng/benchmarks/data/pokec/untiled_bak"
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
echo "DDR" >> $result_file
(set -x; numactl -m 0 ${bin_addr}/bfs ${data_addr}/${data_file} >> $result_file)
echo "MCDRAM" >> $result_file
(set -x; numactl -m 1 ${bin_addr}/bfs ${data_addr}/${data_file} >> $result_file)
echo done.
