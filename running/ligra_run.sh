#! /usr/bin/bash
if [[ $# -lt 2 ]]; then
	echo "Usage: ./ligra_run.sh <makr> <data_file> [-s]"
	exit
fi

mark=$1
data_file=$2
opt=$3
apps_dir=/home/zpeng/code/ligra/apps
rounds=10

output=output_$(date +%Y%m%d-%H%M%S)_ligra_${mark}.txt
:> $output

set -x

# BFS
echo "BFS:" >> $output
${apps_dir}/BFS $opt -rounds ${rounds} ${data_file}.adj >> $output
echo "" >> $output

# PageRank
echo "PageRank:" >> $output
${apps_dir}/PageRank $opt -rounds ${rounds} ${data_file}.adj  >> $output
echo "" >> $output

# SSSP
echo "SSSP:" >> $output
${apps_dir}/BellmanFord $opt -rounds ${rounds} ${data_file}_weighted.adj >> $output
echo "" >> $output

# CC
echo "CC:" >> $output
${apps_dir}/Components $opt -rounds ${rounds} ${data_file}.adj >> $output
echo "" >> $output

# BC
echo "BC:" >> $output
${apps_dir}/BC $opt -rounds ${rounds} ${data_file}.adj >> $output
echo "" >> $output

# MIS
echo "MIS:" >> $output
${apps_dir}/MIS $opt -rounds ${rounds} ${data_file}.adj >> $output
echo "" >> $output

set +x
