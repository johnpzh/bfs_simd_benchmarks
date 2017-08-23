#!/usr/bin/bash
make
# Determine the data file
if [[ $# -eq 0 ]]; then
	data_file="graph256MD4"
else
	case $1 in
	4096)
		data_file="graph4096"
		;;
	256M)
		data_file="graph256MD4"
		;;
	*)
		data_file=$1
		;;
	esac
fi
version="simd-bfs-edges"
bin_addr="."
data_addr="/home/zpeng/benchmarks/data/rodinia_gen"
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
echo "Edges-SIMD-BFS" >> $result_file
(set -x; ${bin_addr}/bfs 1 ${data_addr}/${data_file} 256 32768 >> $result_file)
echo done.
