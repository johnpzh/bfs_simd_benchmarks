#!/usr/bin/bash
if [[ $# -lt 1 ]]; then
	echo "Usage: ./run.sh <data_file>"
	exit
fi
data_file=$1
app="/home/zpeng/code/galois_set/build/release/apps/betweennesscentrality/betweennesscentrality-outer"
output_file="output_$(date +%Y%m%d-%H%M%S).txt"
:> $output_file
for ((i = 6; i < 9; ++i)); do
	power=$((2**${i}))
	for ((k = 0; k < 10; ++k)); do
		(set -x; $app ${data_file} -startNode=0 -t=${power} -limit=1 -noverify >> $output_file)
	done
done
