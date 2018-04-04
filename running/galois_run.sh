#! /usr/bin/bash

if [[ $# -lt 2 ]]; then
	echo "Usage: ./galois_run.sh <mark> <data_file> [-symmetricGraph]"
	exit
fi

mark=$1
data_file=$2
opt=$3
galois_dir="/home/zpeng/code/galois/build/release/apps"
rounds=10
threads=9


output="output_$(date +%Y%m%d-%H%M%S)_galois_${mark}.txt"
:> $output

# BFS
echo "BFS:" >> $output
for ((i = 6; i < threads; ++i)); do
	power=$((2**${i}))
	for ((k = 0; k < rounds; ++k)); do
		(set -x; ${galois_dir}/bfs/bfs ${data_file}.vgr -startNode=0 -t=${power} -algo=barrierWithCas -noverify $opt >> $output)
	done
done
echo "" >> $output

# PageRank
echo "PageRank:" >> $output
for ((i = 6; i < threads; ++i)); do
	power=$((2**${i}))
	for ((k = 0; k < rounds; ++k)); do
		(set -x; ${galois_dir}/pagerank/pagerank ${data_file}.gr -graphTranspose=${data_file}.tgr -maxIterations=1 -t=${power} -algo=pull -noverify $opt >> $output)
	done
done
echo "" >> $output

# SSSP
echo "SSSP:" >> $output
for ((i = 6; i < threads; ++i)); do
	power=$((2**${i}))
	for ((k = 0; k < rounds; ++k)); do
		(set -x; ${galois_dir}/sssp/sssp ${data_file}.gr -startNode=0 -t=${power} -algo=asyncPP -noverify $opt >> $output)
	done
done
echo "" >> $output

# CC
echo "CC:" >> $output
for ((i = 6; i < threads; ++i)); do
	power=$((2**${i}))
	for ((k = 0; k < rounds; ++k)); do
		(set -x; ${galois_dir}/connectedcomponents/connectedcomponents ${data_file}.vgr  -t=${power} -algo=async -noverify $opt >> $output)
	done
done
echo "" >> $output

# BC
echo "BC:" >> $output
for ((i = 6; i < threads; ++i)); do
	power=$((2**${i}))
	for ((k = 0; k < rounds; ++k)); do
		(set -x; ${galois_dir}/betweennesscentrality/betweennesscentrality-inner ${data_file}.vgr -algo=async -graphTranspose=${data_file}.tvgr -noverify -startNode=0 -t=${power} $opt >> $output)
	done
done
echo "" >> $output

## MIS
#echo "MIS:" >> $output
#for ((i = 6; i < threads; ++i)); do
#	power=$((2**${i}))
#	for ((k = 0; k < rounds; ++k)); do
#		(set -x; ${galois_dir}/independentset/independentset ${data_file}.vgr -t=${power} -nondet -noverify >> $output)
#	done
#done
#echo "" >> $output
