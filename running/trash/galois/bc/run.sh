#!/usr/bin/bash
if [[ $# -lt 2 ]]; then
	echo "Usage: ./run.sh <data_file> [-symmetricGraph]"
	exit
fi
data_file=$1
opt=$2
app="/home/zpeng/code/galois/build/release/apps/betweennesscentrality/betweennesscentrality-inner"
output_file="output_$(date +%Y%m%d-%H%M%S).txt"
:> $output_file
for ((i = 6; i < 9; ++i)); do
	power=$((2**${i}))
	for ((k = 0; k < 10; ++k)); do
		(set -x; $app ${data_file}.vgr -algo=async -graphTranspose=${data_file}.tvgr -noverify -startNode=0 -t=${power} $opt)
	done
done
