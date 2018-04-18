#!/usr/bin/bash
if [[ $# -lt 5 ]]; then
	echo "Usage: ./process.sh <command> <data_file> <min_tile_size> <max_tile_size> <stripe_length> [-w]"
	exit
fi

app=$1
data_file=$2
min_tile_size=$3
max_tile_size=$4
stripe_length=$5
weighted=$6

set -x

# Tile SIze
output="output_$(date +%Y%m%d-%H%M%S).txt"
:> $output
for ((i = min_tile_size; i <= max_tile_size; i *= 2)); do
	$app ${data_file} ${i} $stripe_length $weighted >> $output
done

# Twt
#data_file="/data/zpeng/twt_combine/tiled_2-power/out.twitter"
#output="output_$(date +%Y%m%d-%H%M%S)_twt.txt"
#:> $output
#for ((i = 16384; i < 524289; i *= 2)); do
#	./bfs ${data_file} ${i} 40 &>> $output
#done
set +x
