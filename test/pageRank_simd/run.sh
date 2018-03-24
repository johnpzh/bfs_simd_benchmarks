#!/usr/bin/bash
make

# Pokec
data_file="/data/zpeng/pokec_combine/tiled_2-power/soc-pokec"
output="output_$(date +%Y%m%d-%H%M%S)_pokec.txt"
:> $output
for ((i = 8192; i < 65537; i *= 2)); do
	./page_rank ${data_file} ${i} 16 &>> $output
done

# Twt
data_file="/data/zpeng/twt_combine/tiled_2-power/out.twitter"
output="output_$(date +%Y%m%d-%H%M%S)_twt.txt"
:> $output
for ((i = 16384; i < 524289; i *= 2)); do
	./page_rank ${data_file} ${i} 40 &>> $output
done
