#!/usr/bin/bash
bin_address="/home/zpeng/code/Galois/build/release/apps/connectedcomponents"
#data_address="/home/zpeng/benchmarks/data/pokec"
data_address="/home/zpeng/benchmarks/data/twt"
output_file="output_twt_$(date +%Y%m%d-%H%M%S).txt"
:> $output_file
for ((i = 0; i < 9; ++i)); do
	power=$((2**${i}))
#	(set -x; numactl -m 0 ${bin_address}/connectedcomponents ${data_address}/soc-pokec_nohead.gr  -t=${power} >> $output_file)
	(set -x; numactl -m 0 ${bin_address}/connectedcomponents ${data_address}/out.twitter_nohead.gr  -t=${power} >> $output_file)
done
