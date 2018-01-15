#!/usr/bin/bash
make clean && make
# Determine the data file
data_addr="/home/zpeng/benchmarks/data"
if [[ $# -eq 0 ]]; then
	data_addr=${data_addr}/twt
	data_file="out.twitter"
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
version="naive-sssp"
bin_addr="."
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

:> $result_file
#echo "DDR" >> $result_file
(set -x; numactl -m 0 ${bin_addr}/sssp ${data_addr}/${data_file} >> $result_file)
#echo "MCDRAM" >> $result_file
#(set -x; numactl -m 1 ${bin_addr}/page_rank ${data_addr}/${data_file} >> $result_file)
echo done.
