#!/usr/bin/bash
make
for ((i=1;i<7;++i)); do
	row_step=$((2**i))
	results_file="output_step${row_step}_$(date +%Y%m%d-%H%M%S).txt"
	(set -x; ./bfs ~/benchmarks/data/twt_col/out.twitter 4096 ${row_step} > ${results_file})
done
