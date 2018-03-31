#!/usr/bin/bash
if [[ $# -lt 1 ]]; then
	echo "Usage: ./run.sh <data_file>"
	exit
fi
data_file=$1
app="/home/zpeng/code/ligra_set/apps/BFS"
output_file="output_$(date +%Y%m%d-%H%M%S).txt"
:> $output_file

(set -x; $app $data_file >> $output_file)
