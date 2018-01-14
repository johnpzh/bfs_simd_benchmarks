#!/usr/bin/bash
make
## Determine the data file
#data_addr="/home/zpeng/benchmarks/data"
#if [[ $# -eq 0 ]]; then
##data_addr=${data_addr}/twt
#	data_addr="/home/zpeng/benchmarks/data/twt/coo_tiled_bak"
#	data_file="out.twitter"
#	tile_width=8192
#else
#	case $1 in
#	pokec)
#		data_addr=${data_addr}/pokec
#		data_file="soc-pokec-relationships.txt"
#		tile_width=1024
#		;;
#	twt)
#		data_addr=${data_addr}/twt
#		data_file="out.twitter"
#		tile_width=4096
#		;;
#	*)
#		data_file=$1
#		tile_width=$2
#		;;
#	esac
#fi
#version="naive-pageRank"
#bin_addr="."
#result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"
#
#echo "DDR" >> $result_file
#(set -x; numactl -m 0 ${bin_addr}/page_rank ${data_addr}/${data_file} ${tile_width} >> $result_file)
##echo "MCDRAM" >> $result_file
##(set -x; numactl -m 1 ${bin_addr}/page_rank ${data_addr}/${data_file} 1 >> $result_file)
#echo done.

#(set -x; ./page_rank ../../data/twt/coo_tiled_bak/out.twitter 2048 > output_t2048.txt)
#(set -x; ./page_rank ../../data/twt/coo_tiled_bak/out.twitter 4096 > output_t4096.txt)
results_file=output_t8192_$(date +%Y%m%d-%H%M%S).txt
(set -x; ./bfs ../../data/twt/coo_tiled_bak/out.twitter 8192 > ${results_file})
