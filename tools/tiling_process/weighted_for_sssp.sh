#!/usr/bin/bash
if [[ $# -lt 3 ]]; then
	echo "Usage: ./process.sh <data_file> <tile_size> <stripe_length> [undirected=1]"
	exit
fi

data_file=$1
tile_size=$2
stripe_length=$3
undirected=$4
set -x
## CSR tiling
#cd /home/zpeng/benchmarks/tools/csr_tiling
#make clean
#make weighted=1
#./page_rank $data_file 
#
## Reorder
#cd /home/zpeng/benchmarks/tools/vertex_id_remap
#make clean
#make weighted=1
#./bfs $data_file $tile_size $stripe_length

reordered_data=${data_file}_reorder

# CSR tiling
cd /home/zpeng/benchmarks/tools/csr_tiling
make clean
make weighted=1 $undirected
./page_rank $reordered_data 

# COO tiling
cd /home/zpeng/benchmarks/tools/coo_tiling
make clean
make weighted=1
./page_rank $reordered_data $tile_size $tile_size

# Column-major
cd /home/zpeng/benchmarks/tools/column_major_tile
make clean
make weighted=1
./kcore $reordered_data $tile_size $stripe_length $stripe_length


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
set +x
