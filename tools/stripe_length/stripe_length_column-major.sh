#!/usr/bin/bash
if [[ $# -lt 4 ]]; then
	echo "Usage: ./stripe_length_column-major.sh <data_file> <tile_size> <min_stripe_length> <max_stripe_length>"
	exit
fi

data_file=$1
tile_size=$2
min_stripe_length=$3
max_stripe_length=$4
# CSR tiling
#cd /home/zpeng/benchmarks/test/csr_tiling
#make clean
#make untile=1
#./page_rank $data_file 

# COO tiling
#cd /sciclone/home2/zpeng01/benchmarks/test/coo_tiling
#make clean
#make
#for((tile_size = min_tile_size; tile_size <= max_tile_size; tile_size *= 2)); do
#	./page_rank $data_file $tile_size
#done

# Column-major
cd /sciclone/home2/zpeng01/benchmarks/tools/column_major_tile
make clean
make
#for((stripe_length = min_stripe_length; stripe_length <= max_stripe_length; stripe_length *= 2)); do
#	./kcore $data_file $tile_size $stripe_length
#done
./kcore $data_file $tile_size $min_stripe_length $max_stripe_length

# CSR Tiling reverse

# COO Tiling reverse

# Column-major
