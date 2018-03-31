#!/usr/bin/bash
if [[ $# -lt 3 ]]; then
	echo "Usage: ./process.sh <data_file> <tile_size> <stripe_length>"
	exit
fi

data_file=$1
tile_size=$2
stripe_length=$3
## CSR tiling
#cd /home/zpeng/benchmarks/tools/csr_tiling
#make clean
#make
#./page_rank $data_file 

# COO tiling
cd /home/zpeng/benchmarks/tools/coo_tiling
make clean
make 
./page_rank $data_file $tile_size $tile_size

# Column-major
cd /home/zpeng/benchmarks/tools/column_major_tile
make clean
make 
./kcore $data_file $tile_size $stripe_length $stripe_length

# Reorder
cd /home/zpeng/benchmarks/tools/vetex_id_remap
make clean
make 
./bc $data_file $tile_size $stripe_length

## CSR Tiling reverse
#cd /home/zpeng/benchmarks/tools/csr_tiling_reverse
#make clean
#make untile=1
#./page_rank $data_file 
#
## COO Tiling reverse
#cd /home/zpeng/benchmarks/tools/coo_tiling_reverse
#make clean
#make untile=1
#./page_rank $data_file $tile_size
#
## Column-major
#cd /home/zpeng/benchmarks/tools/column_major_tile_reverse
#make clean
#make untile=1
#./kcore $data_file $tile_size $stripe_length
