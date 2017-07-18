#!/usr/bin/bash
# Determine the data file
if [[ $# -eq 0 ]]; then
	data_file="soc-pokec-relationships.txt"
else
	case $1 in
	pokec)
		data_file="soc-pokec-relationships.txt"
		;;
	*)
		data_file=$1
		;;
	esac
fi
#echo "data_file: $data_file"
#data_file="local_$data_file"
version="tiled-simd-pageRank"
bin_addr="."
data_addr="/home/zpeng/benchmarks/data"
result_file="result_${version}_${data_file}_$(date +%Y%m%d-%H%M%S).txt"

touch $result_file
echo "Threads Time" >> $result_file
power=0
power_max=8
#power=4
#power_max=14
size=$((2 ** power))
while [	$power -le $power_max ]
do
	${bin_addr}/page_rank ${data_addr}/${data_file} ${size} 32 >> $result_file
#${bin_addr}/page_rank ${data_addr}/${data_file} 256 ${size} >> $result_file
	echo -n .
	power=$((power + 1))
	size=$((2 ** power))
done
echo done.
