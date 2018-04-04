#!/usr/bin/bash
if [[ $# -lt 1 ]]; then
	echo "Usage: ./run.sh <data_file>"
	exit
fi
data_file=$1
app="/home/zpeng/code/galois/build/release/apps/independentset/independentset"
output_file="output_$(date +%Y%m%d-%H%M%S).txt"
:> $output_file
for ((i = 6; i < 9; ++i)); do
	power=$((2**${i}))
	for ((k = 0; k < 10; ++k)); do
		(set -x; $app ${data_file}.vgr -startNode=0 -t=${power} -nondet -noverify >> $output_file)
	done
done
