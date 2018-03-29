#!/usr/bin/bash
make clean
make

data_file="/data/zpeng/twt_combine/tiled_2-power/out.twitter"
# Tile Size
#output=output_$(date +%Y%m%d-%H%M%S)_tile-size.txt
#:> $output
#for ((i = 1024; i < 524289; i *= 2)); do
#	numactl -m 0 ./page_rank ${data_file} ${i} 40 &>> $output
#done

# Stripe Length
output=output_$(date +%Y%m%d-%H%M%S)_stripe-length.txt
:> $output
for ((i = 1; i < 2049; i *= 2)); do
	numactl -m 0 ./page_rank ${data_file} 4096 ${i} &>> $output
done
