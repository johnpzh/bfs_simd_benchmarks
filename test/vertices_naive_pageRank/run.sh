#!/usr/bin/bash
make
# Determine the data file
data_addr="/home/zpeng/benchmarks/data"
if [[ $# -eq 0 ]]; then
	data_addr=${data_addr}/twt
	data_file="out.twitter"
	tile_width=4096
else
	case $1 in
	pokec)
		data_addr=${data_addr}/pokec
		data_file="soc-pokec-relationships.txt"
		tile_width=1024
		;;
	twt)
		data_addr=${data_addr}/twt
		data_file="out.twitter"
		tile_width=4096
		;;
	*)
		data_file=$1
		tile_width=$2
		;;
	esac
fi
version="naive-pageRank-vertices"
bin_addr="."
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
echo "Vertices-Naive-PageRank" >> $result_file
(set -x; ${bin_addr}/page_rank ${data_addr}/${data_file} 1 >> $result_file)
echo done.
