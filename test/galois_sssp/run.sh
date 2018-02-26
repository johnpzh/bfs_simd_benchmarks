#!/usr/bin/bash
bin="/home/zpeng/code/galois_set/build/release/apps/sssp/sssp"
#data="/home/zpeng/benchmarks/data/pokec/soc-pokec_nohead.gr"
data="/home/zpeng/benchmarks/data/twt/out.twitter_nohead.gr"
output="output_$(date +%Y%m%d-%H%M%S).txt"
:> $output
#for ((i = 0; i < 9; ++i)); do
#	power=$((2**${i}))
##(set -x; ${bin} ${data_address}/soc-pokec_nohead.gr  -startNode=0 -t=${power} >> $output_file)
#	(set -x; ${bin} ${data} -startNode=0 -t=${power} >> $output_file)
#done

thd=64
set -x
echo "async-------" >> $output
${bin} ${data} -startNode=0 -t=${thd} -noverify -algo=async >> $output
echo "asyncPP-------" >> $output
${bin} ${data} -startNode=0 -t=${thd} -noverify -algo=asyncPP >> $output
echo "asyncWithCas-------" >> $output
${bin} ${data} -startNode=0 -t=${thd} -noverify -algo=asyncWithCas >> $output
echo "serial-------" >> $output
${bin} ${data} -startNode=0 -t=${thd} -noverify -algo=serial >> $output
echo "graphlab-------" >> $output
${bin} ${data} -startNode=0 -t=${thd} -noverify -algo=graphlab >> $output
echo "ligraChi-------" >> $output
${bin} ${data} -startNode=0 -t=${thd} -noverify -algo=ligraChi >> $output
echo "ligra-------" >> $output
${bin} ${data} -startNode=0 -t=${thd} -noverify -algo=ligra >> $output
set +x
