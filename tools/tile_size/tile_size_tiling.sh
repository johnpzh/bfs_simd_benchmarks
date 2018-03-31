#!/usr/bin/bash
if [[ $# -lt 4 ]]; then
	echo "Usage: ./process.sh <data_file> <min_tile_size> <max_tile_size> <stripe_length>"
	exit
fi

data_file=$1
min_tile_size=$2
max_tile_size=$3
stripe_length=$4
# CSR tiling
#cd /home/zpeng/benchmarks/test/csr_tiling
#make clean
#make untile=1
#./page_rank $data_file 

# COO tiling
#cd /sciclone/home2/zpeng01/benchmarks/tools/coo_tiling
cd /home/zpeng/benchmarks/tools/coo_tiling
make clean
make
#for((tile_size = min_tile_size; tile_size <= max_tile_size; tile_size *= 2)); do
#	./page_rank $data_file $tile_size $stripe_length
#done
./page_rank $data_file $min_tile_size $max_tile_size

# Column-major
#cd /sciclone/home2/zpeng01/benchmarks/tools/column_major_tile
cd /home/zpeng/benchmarks/tools/column_major_tile
make clean
make
for((tile_size = min_tile_size; tile_size <= max_tile_size; tile_size *= 2)); do
	./kcore $data_file $tile_size $stripe_length $stripe_length
done

# CSR Tiling reverse

# COO Tiling reverse

# Column-major
