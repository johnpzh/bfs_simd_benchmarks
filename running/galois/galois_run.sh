#! /usr/bin/bash

if [[ $# -lt 2 ]]; then
	echo "Usage: ./galois_run.sh <mark> <data_file>"
	exit
fi

mark=$1
data_file=$2

output="output_$(date +%Y%m%d-%H%M%S)_galois_${mark}.txt"
:> $output

# BFS
echo "BFS:" >> $output
cd /home/zpeng/benchmarks/running/galois/bfs
./run.sh ${data_file}.vgr >> $output
echo "" >> $output

# PageRank
echo "PageRank:" >> $output
cd /home/zpeng/benchmarks/running/galois/pageRank
./run.sh ${data_file}.gr ${data_file}.tgr >> $output
echo "" >> $output

# SSSP
echo "SSSP:" >> $output
cd /home/zpeng/benchmarks/running/galois/sssp
./run.sh ${data_file}.gr >> $output
echo "" >> $output

# CC
echo "CC:" >> $output
cd /home/zpeng/benchmarks/running/galois/cc
./run.sh ${data_file}.vgr >> $output
echo "" >> $output

# BC
echo "BC:" >> $output
cd /home/zpeng/benchmarks/running/galois/bc
./run.sh ${data_file}.vgr >> $output
echo "" >> $output

# MIS
echo "MIS:" >> $output
cd /home/zpeng/benchmarks/running/galois/mis
./run.sh ${data_file}.vgr >> $output
echo "" >> $output
