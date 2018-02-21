#!/usr/bin/bash
bin_address="/home/zpeng/code/Galois/build/release/apps/bfs"
#data="/home/zpeng/benchmarks/data/pokec/soc-pokec_nohead.vgr"
data="/home/zpeng/benchmarks/data/twt/out.twitter_nohead.vgr"
output="output_$(date +%Y%m%d-%H%M%S).txt"
:> $output
#for ((i = 0; i < 9; ++i)); do
#	power=$((2**${i}))
##(set -x; ${bin_address}/bfs ${data_address}/soc-pokec_nohead.gr  -startNode=0 -t=${power} >> $output_file)
#	(set -x; ${bin_address}/bfs ${data} -startNode=0 -t=${power} >> $output_file)
#done

thd=64
set -x
echo "async-------" >> $output
${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=async -noverify >> $output
echo "barrier-------" >> $output
${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=barrier -noverify >> $output
echo "barrierWithCas-------" >> $output
${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=barrierWithCas -noverify >> $output
#echo "detBase-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=detBase -noverify >> $output
#echo "detDisjoint-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=detDisjoint -noverify >> $output
#echo "highCentrality-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=highCentrality -noverify >> $output
#echo "hybrid-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=hybrid -noverify >> $output
#echo "serial-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=serial -noverify >> $output

#echo "async detBase-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=async -debBase
#echo "barrier detBase-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=barrier -debBase
#echo "barrierWithCas detBase-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=barrierWithCas -debBase
#echo "detBase detBase-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=detBase -debBase
#echo "detDisjoint detBase-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=detDisjoint -debBase
#echo "highCentrality detBase-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=highCentrality -debBase
#echo "hybrid detBase-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=hybrid -debBase
#echo "serial detBase-------" >> $output
#${bin_address}/bfs ${data} -startNode=0 -t=${thd} -algo=serial -debBase
set +x
