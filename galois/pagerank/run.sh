#!/usr/bin/bash
if [[ $# -lt 1 ]]; then
	echo "Usage: ./run.sh <weighted_data_file> <weighted_data_file_transpose>"
	exit
fi
data_file=$1
data_trans=$2
app="/home/zpeng/code/galois_set/build/release/apps/pagerank/pagerank"
output_file="output_twt_$(date +%Y%m%d-%H%M%S).txt"
:> $output_file
for ((i = 6; i < 9; ++i)); do
	power=$((2**${i}))
	(set -x; $app $data_file -graphTranspose=$data_trans -maxIterations=1 -t=${power} -algo=pull -noverify >> $output_file)
done
