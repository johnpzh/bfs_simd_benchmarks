#!/usr/bin/bash
if [[ $# -lt 3 ]]; then
	echo "Usage: ./process.sh <data_file> <tile_size> <stripe_length> [weighted=1]"
	exit
fi

data_file=$1
tile_size=$2
stripe_length=$3
weighted=$4

tool_dir=/sciclone/home2/zpeng01/benchmarks/tools

set -x

# CSR tiling
cd ${tool_dir}/csr_tiling
make clean
make $weighted
./page_rank $data_file 

# COO tiling
cd ${tool_dir}/coo_tiling
make clean
make $weighted
./page_rank $data_file $tile_size $tile_size

# Column-major
cd ${tool_dir}/column_major_tile
make clean
make $weighted
./kcore $data_file $tile_size $stripe_length $stripe_length


## CSR Tiling reverse
#cd ${tool_dir}/csr_tiling_reverse
#make clean
#make 
#./page_rank $data_file 
#
## COO Tiling reverse
#cd ${tool_dir}/coo_tiling_reverse
#make clean
#make 
#./page_rank $data_file $tile_size
#
## Column-major
#cd ${tool_dir}/column_major_tile_reverse
#make clean
#make 
#./kcore $data_file $tile_size $stripe_length
set +x
