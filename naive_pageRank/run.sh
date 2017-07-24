#!/usr/bin/bash
make
# Determine the data file
case $1 in
pokec)
	data_file="soc-pokec-relationships.txt"
	;;
*)
	data_file=$1
	;;
esac
#data_file="local_$data_file"
version="naive_pageRank"
bin_addr="."
data_addr="/home/zpeng/benchmarks/data"
power_max=8
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
echo "Threads Time" >> $result_file
power=0
size=$((2 ** power))
#while [	$power -le $power_max ]
#do
#	${bin_addr}/page_rank ${data_addr}/${data_file} ${size} >> $result_file
#	echo -n .
#	power=$((power + 1))
#	size=$((2 ** power))
#done
(set -x; ${bin_addr}/page_rank ${data_addr}/${data_file} ${size} >> $result_file)
echo done.
