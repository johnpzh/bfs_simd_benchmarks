#!/usr/bin/bash
bin="/home/zpeng/code/galois_set/build/release/apps/connectedcomponents/connectedcomponents"
#data="/home/zpeng/benchmarks/data/pokec/soc-pokec_nohead.vgr"
data="/home/zpeng/benchmarks/data/twt/out.twitter_nohead.vgr"
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
${bin} ${data} -t=${thd} -noverify -algo=async >> $output
echo "blockedasync-------" >> $output
${bin} ${data} -t=${thd} -noverify -algo=blockedasync >> $output
echo "asyncOc-------" >> $output
${bin} ${data} -t=${thd} -noverify -algo=asyncOc >> $output
echo "labelProp-------" >> $output
${bin} ${data} -t=${thd} -noverify -algo=labelProp >> $output
echo "serial-------" >> $output
${bin} ${data} -t=${thd} -noverify -algo=serial >> $output
echo "sync-------" >> $output
${bin} ${data} -t=${thd} -noverify -algo=sync >> $output
set +x
