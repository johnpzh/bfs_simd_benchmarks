#!/usr/bin/bash
make
# Determine the data file
if [[ $# -eq 0 ]]; then
	data_file="soc-pokec-relationships.txt"
else
	case $1 in
	pokec)
		data_file="soc-pokec-relationships.txt"
		;;
	twt)
		data_file="out.twitter_mpi"
		;;
	*)
		data_file=$1
		;;
	esac
fi
bin_addr="."
data_addr="/home/zpeng/benchmarks/data"

power=10
power_max=8
size=$((2 ** power))
while [	$power -le $power_max ]
do
	(set -x; ${bin_addr}/page_rank ${data_addr}/${data_file} ${size} 1024 2048 >> $result_file)
	power=$((power + 1))
	size=$((2 ** power))
done
echo done.
