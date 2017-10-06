#!/usr/bin/bash
bin_address="/home/zpeng/code/ligra/apps"
#data_address="/home/zpeng/benchmarks/data/pokec"
data_address="/home/zpeng/benchmarks/data/twt"
output_file="output_$(date +%Y%m%d-%H%M%S).txt"
:> $output_file
#for ((i = 0; i < 9; ++i)); do
#	power=$((2**${i}))
#(set -x; ${bin_address}/bfs ${data_address}/soc-pokec_nohead.gr  -startNode=0 -t=${power} >> $output_file)
(set -x; ${bin_address}/KCore -s ${data_address}/out.twitter_nohead.adj >> $output_file)
#done
