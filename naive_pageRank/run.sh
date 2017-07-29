#!/usr/bin/bash
make
version="naive_pageRank"
bin_addr="."
data_addr="/home/zpeng/benchmarks/data/twt"
data_file="out.twitter"
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
echo "Threads Time" >> $result_file
power=0
power_max=8
size=$((2 ** power))
#(set -x; ${bin_addr}/page_rank ${data_addr}/${data_file} ${size} >> $result_file)
(set -x; ${bin_addr}/page_rank ${data_addr}/${data_file} ${size} >> $result_file)
(set -x; numactl -m 1 ${bin_addr}/page_rank ${data_addr}/${data_file} ${size} >> $result_file)
echo done.
