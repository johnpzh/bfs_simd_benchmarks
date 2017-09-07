#!/usr/bin/bash
bin_address="/home/zpeng/code/Galois/build/release/apps/pagerank"
#data_address="/home/zpeng/benchmarks/data/pokec"
data_address="/home/zpeng/benchmarks/data/twt"
#output_file="output.txt"
output_file="output_twt_$(date +%Y%m%d-%H%M%S).txt"
:> $output_file
for ((i = 0; i < 9; ++i)); do
	power=$((2**${i}))
#(set -x; ${bin_address}/pagerank ${data_address}/soc-pokec_nohead.gr -graphTranspose=${data_address}/soc-pokec_nohead.tgr -maxIterations=1 -t=${power} >> $output_file)
	(set -x; ${bin_address}/pagerank ${data_address}/out.twitter_nohead.gr -graphTranspose=${data_address}/out.twitter_nohead.tgr -maxIterations=1 -t=${power} >> $output_file)
done
