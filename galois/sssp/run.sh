#!/usr/bin/bash
if [[ $# -lt 1 ]]; then
	echo "Usage: ./run.sh <weighted_data_file>"
	exit
fi
data_file=$1
app="/home/zpeng/code/galois_set/build/release/apps/sssp/sssp"
output_file="output_twt_$(date +%Y%m%d-%H%M%S).txt"
:> $output_file
for ((i = 6; i < 9; ++i)); do
	power=$((2**${i}))
	(set -x; numactl -m 0 $app $data_file -startNode=0 -t=${power} -algo=asyncPP -noverify >> $output_file)
done
