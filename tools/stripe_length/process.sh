#!/usr/bin/bash
if [[ $# -lt 5 ]]; then
	echo "Usage: ./process.sh <command> <data_file> <tile_size> <min_stripe_length> <max_stripe_length> [-w]"
	exit
fi

app=$1
data_file=$2
tile_size=$3
min_stripe_length=$4
max_stripe_length=$5
weighted=$6

set -x

# Stripe Length
output="output_$(date +%Y%m%d-%H%M%S).txt"
:> $output
for ((i = min_stripe_length; i <= max_stripe_length; i *= 2)); do
	$app ${data_file} ${tile_size} $i $weighted &>> $output
done

set +x
