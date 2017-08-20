#!/usr/bin/bash
make
# Determine the data file
if [[ $# -lt 2 ]]; then
	data_file="soc-pokec"
	tile_width=1024
else
	data_file=$1
	tile_width=$2
fi
version="tiled-naive-bfs"
bin_addr="."
data_addr="/home/zpeng/benchmarks/data/pokec"
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
echo "DDR" >> $result_file
(set -x; numactl -m 0 ${bin_addr}/bfs ${data_addr}/${data_file} ${tile_width} >> $result_file)
echo "MCDRAM" >> $result_file
(set -x; numactl -m 1 ${bin_addr}/bfs ${data_addr}/${data_file} ${tile_width} >> $result_file)
echo done.
