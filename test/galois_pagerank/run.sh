#!/usr/bin/bash
bin="/home/zpeng/code/galois_set/build/release/apps/pagerank/pagerank"
#data="/home/zpeng/benchmarks/data/pokec/soc-pokec_nohead.gr"
data="/home/zpeng/benchmarks/data/twt/out.twitter_nohead.gr"
#data_tran="/home/zpeng/benchmarks/data/pokec/soc-pokec_nohead.tvgr"
data_tran="/home/zpeng/benchmarks/data/twt/out.twitter_nohead.tgr"
output="output_$(date +%Y%m%d-%H%M%S).txt"
:> $output
#for ((i = 0; i < 9; ++i)); do
#	power=$((2**${i}))
##(set -x; ${bin} ${data_address}/soc-pokec_nohead.gr  -startNode=0 -t=${power} >> $output_file)
#	(set -x; ${bin} ${data} -startNode=0 -t=${power} >> $output_file)
#done

thd=64
set -x
echo "pull-------" >> $output
${bin} ${data} -graphTranspose=${data_tran} -maxIterations=1 -t=${thd} -noverify -algo=pull >> $output
#echo "detBase-------" >> $output
#${bin} ${data} -startNode=0 -t=${thd} -algo=detBase -noverify >> $output
#echo "detDisjoint-------" >> $output
#${bin} ${data} -startNode=0 -t=${thd} -algo=detDisjoint -noverify >> $output
#echo "highCentrality-------" >> $output
#${bin} ${data} -startNode=0 -t=${thd} -algo=highCentrality -noverify >> $output
#echo "hybrid-------" >> $output
#${bin} ${data} -startNode=0 -t=${thd} -algo=hybrid -noverify >> $output
#echo "serial-------" >> $output
#${bin} ${data} -startNode=0 -t=${thd} -algo=serial -noverify >> $output

#./betweennesscentrality-inner ~/benchmarks/data/twt/out.twitter_nohead.tvgr -algo=async -graphTranspose=/home/zpeng/benchmarks/data/twt/out.twitter_nohead.tvgr -noverify -startNode=0 -t=64
set +x
