#!/usr/bin/bash
make
#results_file="output.txt"
#:> $results_file
for ((i=4;i<14;++i)); do
	row_step=$((2**i))
#echo "STEP: $row_step" >> $results_file
	results_file="output_step${row_step}_$(date +%Y%m%d-%H%M%S).txt"
	(set -x; numactl -m 1 -- ./bfs ~/benchmarks/data/twt_col/out.twitter 4096 ${row_step} > ${results_file})
done
