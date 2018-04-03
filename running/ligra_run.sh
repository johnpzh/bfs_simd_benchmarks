#! /usr/bin/bash
if [[ $# -lt 2 ]]; then
	echo "Usage: ./ligra_run.sh <makr> <data_file> <opt>"
	exit
fi

apps_dir=/home/zpeng/code/ligra/apps
mark=$1
data_file=$2
opt=$3

output=output_$(date +%Y%m%d-%H%M%S)_ligra_${mark}.txt
:> $output

set -x

# BFS
echo "BFS:" >> $output
${apps_dir}/BFS $opt ${data_file}.adj >> $output
echo "" >> $output

# PageRank
echo "PageRank:" >> $output
${apps_dir}/PageRank $opt ${data_file}.adj  >> $output
echo "" >> $output

# SSSP
echo "SSSP:" >> $output
${apps_dir}/BellmanFord $opt ${data_file}_weighted.adj >> $output
echo "" >> $output

# CC
echo "CC:" >> $output
${apps_dir}/Components $opt ${data_file}.adj >> $output
echo "" >> $output

# BC
echo "BC:" >> $output
${apps_dir}/BC $opt ${data_file}.adj >> $output
echo "" >> $output

# MIS
echo "MIS:" >> $output
${apps_dir}/MIS $opt ${data_file}.adj >> $output
echo "" >> $output

set +x
